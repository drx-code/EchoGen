"""
Definition of EchoGen transformer model.
"""

import math
import random
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
import numpy as np

import echogen.utils.dist as dist
from echogen.utils.dist import for_visualize
from echogen.models.basic import flash_attn_func, flash_fused_op_installed, AdaLNBeforeHead, \
        SelfAttnBlock, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid, EchoGenBlock
from echogen.models.flex_attn import FlexAttn
from echogen.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from echogen.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

def patchify(x):
    bsz, c, h, w = x.shape
    p = 2
    h_, w_ = h // p, w // p

    x = x.reshape(bsz, c, h_, p, w_, p)
    x = torch.einsum('nchpwq->ncpqhw', x)
    x = x.reshape(bsz, c * p ** 2, h_, w_)
    return x  # [n, l, d]

def expand_attn_mask(old_mask: torch.Tensor, extra_len: int):
    """
    old_mask: (L, L) 
    extra_len: Lc
    return    : (L+Lc, L+Lc)
    """
    L = old_mask.size(-1)
    new_L = L + extra_len

    if old_mask.dtype == torch.bool:
        allow_val = False
    else: 
        allow_val = 0.0

    new_mask = old_mask.new_full((new_L, new_L), allow_val)

    new_mask[:L, :L] = old_mask
    return new_mask


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads, cos_attn=False)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B,1,6,C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, cond_BD_content, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, semantic_feats=None, content_feats=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h, content_feats = torch.utils.checkpoint.checkpoint(m, h, cond_BD, cond_BD_content, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False, semantic_feats=semantic_feats, content_feats=content_feats)
            else:
                h, content_feats = m(h, cond_BD, cond_BD_content, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, semantic_feats=semantic_feats, content_feats=content_feats)
        return h, content_feats

class EchoGen(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,  
        selecting_idx=None,               
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,
        drop_rate=0., drop_path_rate=0.,   
        norm_eps=1e-6, rms_norm=False,     
        shared_aln=False, head_aln=True,  
        cond_drop_rate=0.1,               
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,

        semantic_feats_len=257,
        semantic_feats_dim=768,
        control_scales=None,
        image_cond_droprate=0.1,
        content_feats_len=257,
        content_feats_dim=768,
    ):
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales

        self.num_block_chunks = block_chunks or 1 # 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth

        self.semantic_feats_len = semantic_feats_len
        self.semantic_feats_dim = semantic_feats_dim
        self.control_scales = control_scales if control_scales is not None else [1.0] * self.depth
        self.content_feats_len = content_feats_len
        self.content_feats_dim = content_feats_dim
            
        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.use_bit_label: {self.use_bit_label}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        customized_kernel_installed = any('EchoGen' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed

        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            file_path = inspect.getsourcefile(flash_attn_func)
            line_number = inspect.getsourcelines(flash_attn_func)[1]
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'>>>>>> {flash_attn_func.__code__.co_varnames=} <<<<<<\n'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule 
        self.first_l = 1

        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()

        self.selecting_idx = None
        self.num_classes = 0
        self.D = self.C
        
        # unconditional dropped text embedding for cfg
        cfg_uncond = torch.empty(self.text_maxlen, self.Ct5) # empty condition 
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.Ct5 ** 0.5
        if rand_uncond:
            self.register_buffer('cfg_uncond', cfg_uncond)
        else:
            self.cfg_uncond = nn.Parameter(cfg_uncond)

        # unconditional dropped semantic conditions for cfg
        control_features_uncond = torch.empty(self.semantic_feats_len, self.semantic_feats_dim)
        rng_dino = torch.Generator(device='cpu')
        rng_dino.manual_seed(0)
        torch.nn.init.trunc_normal_(control_features_uncond, std=1.2, generator=rng_dino)
        control_features_uncond /= self.semantic_feats_dim ** 0.5
        self.control_features_uncond = nn.Parameter(control_features_uncond)

         # unconditional dropped content conditions for cfg
        content_feats_dim_uncond = self.content_feats_dim
        content_feats_len_uncond = self.content_feats_len
        control_features_refer_uncond = torch.empty(content_feats_len_uncond, content_feats_dim_uncond)
        rng_img = torch.Generator(device='cpu')
        rng_img.manual_seed(0)
        torch.nn.init.trunc_normal_(control_features_refer_uncond, std=1.2, generator=rng_img)
        control_features_refer_uncond /= content_feats_dim_uncond ** 0.5
        self.control_features_refer_uncond = nn.Parameter(control_features_refer_uncond)

        # text embedding modules
        self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
        self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
        self.text_proj_for_ca = nn.Sequential(
            nn.Linear(self.Ct5, self.D),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.D, self.D),
        )
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            # this way
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        
        # adding level_embedding
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        # newly-introduced modules for content and semantic image features condition
        # semantic features input 
        self.control_proj_for_ca = nn.Sequential(
            nn.Linear(self.semantic_feats_dim, self.D),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.D, self.D),
        )

        # content features input
        self.control_proj_for_ca_refer = nn.Sequential(
            nn.Linear(self.content_feats_dim, self.D),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.D, self.D),
        )

        # global semantic features input
        self.control_proj_for_sos = nn.Linear(self.semantic_feats_dim, self.D)

        self.control_shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        self.control_pos_start = nn.Parameter(torch.zeros(1, self.first_l, self.C))

        self.control_shared_ada_lin_refer = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        self.control_shared_ada_lin_refer_dino = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                # fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        self.image_cond_droprate = image_cond_droprate
        print(f"Reference image condition drop rate: {self.image_cond_droprate}")
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 

        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = EchoGenBlock(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)

        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
                
        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum((b.sa.using_flash if self.t2i else b.attn.using_flash) for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [EchoGen config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n',
            end='\n\n', flush=True
        )

        # freeze the parameters in T2I backbone
        for name, para in self.named_parameters():
            if 'control' in name: continue
            para.requires_grad_(False)
    
    def control_load_state_dict(self):
        self.control_shared_ada_lin.load_state_dict(self.shared_ada_lin.state_dict())
        nn.init.constant_(self.control_shared_ada_lin[1].bias, 0.)
        for name, module in self.named_modules():
            if "control_mat_kv" in name or 'control_mat_kv_refer' in name:
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.constant_(module.weight, 0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "echogen_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        """Get level embedding for x_BLC in `scale_ind` scale level"""
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        """Get level embedding for x_BLC"""
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)

        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False, semantic_feats=None, content_feats=None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]: 
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        content_feats: content features of reference image, shaped as [B, content_feats_len, content_feats_dim]
        semantic_feats: semantic features of reference image
        :return: logits BLV, V is vocab_size
        """

        x_BLC_wo_prefix = x_BLC_wo_prefix.float()
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            """Condition Processing
            kv_compact: shaped as (sum(lens), KV_dim)
            lens: list of lens
            cu_seqlens_k: list of cumulated sum of lens
            max_seqlen_k: int, max(lens)
            """
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le

            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous() # Normalization
            # projection for sos, concatnent in the first position, shape [B, self.C]
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()   # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous() # projection for cross-attntion
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k     
        
            # conditional images
            semantic_global_feats, semantic_patch_feats = semantic_feats["x_norm_clstoken"].clone(), semantic_feats["x_norm_patchtokens"].clone()
            content_patch_feats = content_feats.reshape(*content_feats.shape[:-2],-1).transpose(2,1).clone()

            # for classfier-free guidance, drop the content of semantic features            
            for b in range(B):
                if random.random() < self.image_cond_droprate:
                    content_patch_feats[b:b+1] = self.control_features_refer_uncond[:].unsqueeze(dim=0).repeat(1,1,1)

            for b in range(B):
                if random.random() < self.image_cond_droprate:
                    semantic_global_feats[b:b+1] = self.control_features_uncond[:1].unsqueeze(dim=0).repeat(1,1,1).squeeze(1)
                    semantic_patch_feats[b:b+1] = self.control_features_uncond[1:].unsqueeze(dim=0).repeat(1,1,1)
            
            # for sos
            cond_BD_semantic = sos_semamtic = self.control_proj_for_sos(semantic_global_feats)

            # for cross-attention
            patch_cond_semantic = self.control_proj_for_ca(semantic_patch_feats).contiguous()
            patch_cond_semantic[0, 0] += must_on_graph

            patch_cond_content = self.control_proj_for_ca_refer(content_patch_feats).contiguous()
            patch_cond_content[0, 0] += must_on_graph

            # for modulating the multi-scale image tokens in EchoGen block by ada-ln 
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            cond_BD_or_gss_semantic = self.control_shared_ada_lin(cond_BD_semantic).contiguous()
            cond_BD_or_gss = cond_BD_or_gss + cond_BD_or_gss_semantic
        
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            sos_semamtic = sos_semamtic.unsqueeze(1).expand(B, 1, -1) + self.control_pos_start.expand(B, 1, -1)
            # concat dino global feature on the head of sequence
            x_BLC = torch.cat((sos_semamtic, sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            rope_ = self.rope2d_freqs_grid[str(tuple(scale_schedule))].clone()
            scale_schedule_train = scale_schedule.copy()
            scale_schedule_train[0] = (1,1,2)
            self.rope2d_freqs_grid[str(tuple(scale_schedule_train))] = torch.concat([rope_[:,:,:,:,0:1],rope_], dim=-2)

            # for modulating the content features in EchoGen block by ada-ln 
            cond_BD_or_gss_content  = self.control_shared_ada_lin_refer(cond_BD).contiguous()
            cond_BD_or_gss_content_semantic = self.control_shared_ada_lin_refer_dino(cond_BD_semantic).contiguous()
            cond_BD_or_gss_content = cond_BD_or_gss_content + cond_BD_or_gss_content_semantic
            
            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.customized_flash_attn:
                raise NotImplementedError
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))              
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule_train)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0:2] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))

                L_x = x_BLC.shape[-2]
                attn_bias = F.pad(attn_bias, (0, self.content_feats_len, 0, self.content_feats_len), value=0.)
                attn_bias[:,:,L_x:,:L_x] = -torch.inf
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule_train)]
        else:
            attn_fn = None

        # [2. block loop]
        SelfAttnBlock.forward, EchoGenBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        cond = patch_cond_content.clone()
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule_train, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule_train, need_to_pad)
                if checkpointing_full_block:
                    x_BLC, cond = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, cond_BD_or_gss_content, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule_train, self.rope2d_freqs_grid, use_reentrant=False, semantic_feats=patch_cond_semantic, content_feats=cond)
                else:
                    x_BLC, cond = b(x=x_BLC, cond_BD=cond_BD_or_gss, cond_BD_content=cond_BD_or_gss_content, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule_train, rope2d_freqs_grid=self.rope2d_freqs_grid, semantic_feats=patch_cond_semantic, content_feats=cond)
        else:
            for i, chunk in enumerate(self.block_chunks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule_train, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule_train, need_to_pad)
                x_BLC, cond = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, cond_BD_content=cond_BD_or_gss_content, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule_train, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid, semantic_feats=patch_cond_semantic, content_feats=cond)

        # [4. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, 1:l_end], cond_BD)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_t_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        semantic_feats=None,
        content_feats=None,
        cfg_i_list=None,
        gt_x_BLC_wo_prefix=None,
    ):   # returns List[idx_Bl]
        """
        :param cfg_t_list: the list of classifier-free guidance parameters of text condition
        :param cfg_i_list: the list of classifier-free guidance parameters of image condition
        :param semantic_feats: semantic features of reference images
        :param content_feats: content features of reference images
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_t_list) >= len(scale_schedule)
        assert len(cfg_i_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        if gt_x_BLC_wo_prefix is not None: 
            gt_x_BLC_wo_prefix = gt_x_BLC_wo_prefix.float() 
            B = gt_x_BLC_wo_prefix.shape[0]
            ti = 0

        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

        bs = 3*B
        if not negative_label_B_or_BLT:
            """No user-defined negative prompts"""
            kv_compact_un = kv_compact.clone()
            total = 0
            for le in lens:
                kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                total += le
            kv_compact = torch.cat((kv_compact, kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1], cu_seqlens_k[1:]+cu_seqlens_k[-1]*2), dim=0)
        else:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1], cu_seqlens_k_un[1:]+cu_seqlens_k[-1]*2), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)

        # control
        semantic_global_feats, semantic_patch_feats = semantic_feats["x_norm_clstoken"], semantic_feats["x_norm_patchtokens"]
        uncond_global_semantic_feats = self.control_features_uncond[:1].unsqueeze(dim=0).repeat(B,1,1).squeeze(1)
        uncond_patch_semantic_feats = self.control_features_uncond[1:].unsqueeze(dim=0).repeat(B,1,1)
        semantic_global_feats = torch.cat((semantic_global_feats, uncond_global_semantic_feats, uncond_global_semantic_feats), dim=0)
        semantic_patch_feats = torch.cat((semantic_patch_feats, uncond_patch_semantic_feats, uncond_patch_semantic_feats), dim=0)
        
        content_feats = content_feats.reshape(*content_feats.shape[:-2],-1).transpose(2,1)
        uncond_patch_content_feats = self.control_features_refer_uncond[:].unsqueeze(dim=0).repeat(B,1,1)
        content_feats = torch.cat((content_feats, uncond_patch_content_feats, uncond_patch_content_feats), dim=0)

        with torch.amp.autocast('cuda', enabled=False):
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous() # sos shape: [2, 4096]
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous() # kv_compact shape: [304, 4096]
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k

            cond_BD_semantic = sos_semamtic = self.control_proj_for_sos(semantic_global_feats)
            patch_cond_semantic = self.control_proj_for_ca(semantic_patch_feats).contiguous()
            patch_cond_content = self.control_proj_for_ca_refer(content_feats).contiguous()

            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
            cond_BD_or_gss_semantic = self.control_shared_ada_lin(cond_BD_semantic.float()).float().contiguous()
            cond_BD_or_gss = cond_BD_or_gss + cond_BD_or_gss_semantic

            cond_BD_or_gss_content  = self.control_shared_ada_lin_refer(cond_BD).contiguous()
            cond_BD_or_gss_content_semantic = self.control_shared_ada_lin_refer_dino(cond_BD_semantic).contiguous()
            cond_BD_or_gss_content = cond_BD_or_gss_content + cond_BD_or_gss_content_semantic
            
            last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)
            last_stage_sos_semantic = sos_semamtic.unsqueeze(1).expand(bs, 1, -1) + self.control_pos_start.expand(bs, 1, -1)
            last_stage = torch.cat((last_stage_sos_semantic, last_stage), dim=1)

            rope_ = self.rope2d_freqs_grid[str(tuple(scale_schedule))].clone()
            scale_schedule_test = scale_schedule.copy()
            scale_schedule_test[0] = (1,1,2)
            self.rope2d_freqs_grid[str(tuple(scale_schedule_test))] = torch.concat([rope_[:,:,:,:,0:1], rope_], dim=-2)

            accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
            idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, EchoGenBlock)  else b.attn).kv_caching(True)
            for b in self.unregistered_blocks: 
                if isinstance(b, EchoGenBlock):
                    b.kv_caching(True)
                    b.ca.kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(b, EchoGenBlock) else module.attn).kv_caching(True)
                    if isinstance(b, EchoGenBlock):
                        b.kv_caching(True)
                        b.ca.kv_caching(True)

        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            """0 by default"""
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
        num_stages_minus_1 = len(scale_schedule_test)-1
        summed_codes = 0
        d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule_test)]).view(1, -1, 1)
        l_end = d.shape[-2]
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
        attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL

        attn_bias = F.pad(attn_bias, (0, self.content_feats_len, 0, self.content_feats_len), value=0)
        attn_bias[:,:,l_end:,:l_end] = -torch.inf
        attn_bias_or_two_vector = attn_bias.type_as(patch_cond_semantic).to(patch_cond_semantic.device)
        for si, pn in enumerate(scale_schedule_test):   # si: i-th segment
            cond = patch_cond_content.clone()
            if si == 0: pn = (1,1,1)
            cfg_t = cfg_t_list[si]
            cfg_i = cfg_i_list[si]
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule_test[:(si+1)]), None)

            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule_test, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule_test, need_to_pad=need_to_pad)

                for m in b.module:
                    last_stage, cond = m(x=last_stage, cond_BD=cond_BD_or_gss, cond_BD_content=cond_BD_or_gss_content, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule_test, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si, semantic_feats=patch_cond_semantic, content_feats=cond)
                    layer_idx += 1

            if si == 0:
                # pop first token
                last_stage = last_stage[:,1:]

            logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
            logits_BlV = logits_BlV[2*B:3*B] + cfg_t * (logits_BlV[B:2*B] - logits_BlV[2*B:3*B]) + cfg_i * (logits_BlV[:B] - logits_BlV[B:2*B])
            
            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]

                # obtain \tildewide{F}_k for next scake generation
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule_test[si][0], scale_schedule_test[si][1], scale_schedule_test[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule_test)
            
            if si != num_stages_minus_1:
                """Word embed"""
                if gt_x_BLC_wo_prefix is not None:
                    last_stage = gt_x_BLC_wo_prefix[:, ti:ti+vae_scale_schedule[si+1][1]*vae_scale_schedule[si+1][2]]
                    ti = ti+vae_scale_schedule[si+1][1]*vae_scale_schedule[si+1][2]
                with torch.amp.autocast('cuda', enabled=False):
                    last_stage = self.word_embed(self.norm0_ve(last_stage))

                last_stage = last_stage.repeat(bs//B, 1, 1) # repeat for cfg

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, EchoGenBlock) else b.attn).kv_caching(False)
            for b in self.unregistered_blocks:
                if isinstance(b, EchoGenBlock):
                    b.kv_caching(False)
                    b.ca.kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(b, EchoGenBlock) else module.attn).kv_caching(False)
                    if isinstance(b, EchoGenBlock):
                        b.kv_caching(False)
                        b.ca.kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule_test, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
            
            if 'control_features_uncond' in k:
                old, new = state_dict[k], self.control_features_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]

            if 'control_features_refer_uncond' in k:
                old, new = state_dict[k], self.control_features_refer_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
        
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: Union[SelfAttnBlock, EchoGenBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift

        self.control_proj_for_sos.weight.data.mul_(0.)
        self.control_proj_for_sos.bias.data.mul_(0.)
 
        for name, module in self.named_modules():
            if "control_attn" in name:
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.constant_(module.weight, 0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)   

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        # actually no remove
        top_k = min(top_k, V)
        # logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1): first for sorted, second for index
        # logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim-1, keepdim=True): return the top_k of value
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        # mask items which has small probability
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def echogen_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return EchoGen(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def echogen_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return EchoGen(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
