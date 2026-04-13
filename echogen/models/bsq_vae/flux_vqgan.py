import argparse
import os
import imageio
import torch
import numpy as np
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from safetensors.torch import load_file
import torch.utils.checkpoint as checkpoint

from .conv import Conv
from .multiscale_bsq import MultiScaleBSQ

ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}

class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type, norm_axis="spatial"):
        super().__init__()
        self.norm_axis = norm_axis
        assert norm_type in ['group', 'batch', "no"]
        if norm_type == 'group':
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
            elif in_channels % 24 == 0: 
                self.norm = nn.GroupNorm(num_groups=24, num_channels=in_channels, eps=1e-6, affine=True)
            else:
                raise NotImplementedError
        elif norm_type == 'batch':
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False) # Runtime Error: grad inplace if set track_running_stats to True
        elif norm_type == 'no':
            self.norm = nn.Identity()
    
    def forward(self, x):
        if self.norm_axis == "spatial":
            if x.ndim == 4:
                x = self.norm(x)
            else:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.norm(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        elif self.norm_axis == "spatial-temporal":
            x = self.norm(x)
        else:
            raise NotImplementedError
        return x

def swish(x: Tensor) -> Tensor:
    try:
        return x * torch.sigmoid(x)
    except:
        device = x.device
        x = x.cpu().pin_memory()
        return (x*torch.sigmoid(x)).to(device=device)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group', cnn_param=None):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])

        self.q = Conv(in_channels, in_channels, kernel_size=1)
        self.k = Conv(in_channels, in_channels, kernel_size=1)
        self.v = Conv(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        B, _, T, _, _ = h_.shape
        h_ = self.norm(h_)
        h_ = rearrange(h_, "B C T H W -> (B T) C H W") # spatial attention only
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "(b t) 1 (h w) c -> b c t h w", h=h, w=w, c=c, b=B, t=T)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type='group', cnn_param=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["res_conv_2d"] in ["half", "full"]:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])
        self.norm2 = Normalize(out_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["res_conv_2d"] in ["full"]:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, cnn_type="2d", spatial_down=False, temporal_down=False):
        super().__init__()
        assert spatial_down == True
        if cnn_type == "2d":
            self.pad = (0,1,0,1)
        if cnn_type == "3d":
            self.pad = (0,1,0,1,0,0) # add padding to the right for h-axis and w-axis. No padding for t-axis 
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0, cnn_type=cnn_type, temporal_down=temporal_down)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, self.pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, cnn_type="2d", spatial_up=False, temporal_up=False, use_pxsl=False):
        super().__init__()
        if cnn_type == "2d":
            self.scale_factor = 2
            self.causal_offset = 0
        else:
            assert spatial_up == True
            if temporal_up:
                self.scale_factor = (2,2,2)
                self.causal_offset = -1
            else:
                self.scale_factor = (1,2,2)
                self.causal_offset = 0
        self.use_pxsl = use_pxsl
        if self.use_pxsl:
            self.conv = Conv(in_channels, in_channels*4, kernel_size=3, stride=1, padding=1, cnn_type=cnn_type, causal_offset=self.causal_offset)
            self.pxsl = nn.PixelShuffle(2)
        else:
            self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_type, causal_offset=self.causal_offset)

    def forward(self, x: Tensor):
        if self.use_pxsl:
            x = self.conv(x)
            x = self.pxsl(x)
        else:
            try:
                x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            except:
                # shard across channel
                _xs = []
                for i in range(x.shape[1]):
                    _x = F.interpolate(x[:,i:i+1,...], scale_factor=self.scale_factor, mode="nearest")
                    _xs.append(_x)
                x = torch.cat(_xs, dim=1)
            x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        in_channels = 3,
        patch_size=8, temporal_patch_size=4, 
        norm_type='group', cnn_param=None,
        use_checkpoint=False,
        use_vae=True,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.temporal_max_down = np.log2(temporal_patch_size)
        self.temporal_down_offset = self.max_down - self.temporal_max_down
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        # downsampling
        # self.conv_in = Conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        # cnn_param["cnn_type"] = "2d" for images, cnn_param["cnn_type"] = "3d" for videos
        if cnn_param["conv_in_out_2d"] == "yes": # "yes" for video
            self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type, cnn_param=cnn_param))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            # downsample, stride=1, stride=2, stride=2 for 4x8x8 Video VAE
            spatial_down = True if i_level < self.max_down else False
            temporal_down = True if i_level < self.max_down and i_level >= self.temporal_down_offset else False
            if spatial_down or temporal_down:
                down.downsample = Downsample(block_in, cnn_type=cnn_param["cnn_type"], spatial_down=spatial_down, temporal_down=temporal_down)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)

        # end
        self.norm_out = Normalize(block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["conv_inner_2d"] == "yes":
            self.conv_out = Conv(block_in, (int(use_vae) + 1) * z_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_out = Conv(block_in, (int(use_vae) + 1) * z_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        else:
            return checkpoint.checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x: Tensor, return_hidden=False) -> Tensor:
        # downsampling
        h0 = self.conv_in(x)
        hs = [h0]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if return_hidden:
            return h, hs, hs_mid
        else:
            return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch = 3, 
        patch_size=8, temporal_patch_size=4, 
        norm_type="group", cnn_param=None,
        use_checkpoint=False,
        use_freq_dec=False, # use frequency features for decoder
        use_pxsf=False
    ):
        super().__init__()
        self.max_up = np.log2(patch_size)
        self.temporal_max_up = np.log2(temporal_patch_size)
        self.temporal_up_offset = self.max_up - self.temporal_max_up
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        self.use_freq_dec = use_freq_dec
        self.use_pxsf = use_pxsf

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        if cnn_param["conv_inner_2d"] == "yes":
            self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type=norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type, cnn_param=cnn_param))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            # upsample, stride=1, stride=2, stride=2 for 4x8x8 Video VAE, offset 1 compared with encoder
            # https://github.com/black-forest-labs/flux/blob/b4f689aaccd40de93429865793e84a734f4a6254/src/flux/modules/autoencoder.py#L228
            spatial_up = True if 1 <= i_level <= self.max_up else False
            temporal_up = True if 1 <= i_level <= self.max_up and i_level >= self.temporal_up_offset+1 else False
            if spatial_up or temporal_up:
                up.upsample = Upsample(block_in, cnn_type=cnn_param["cnn_type"], spatial_up=spatial_up, temporal_up=temporal_up, use_pxsl=self.use_pxsf)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["conv_in_out_2d"] == "yes":
            self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        else:
            return checkpoint.checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    """
        vqgan_ckpt=vqgan_ckpt,
        sd_ckpt=None,
        inference_type='image',
        save='./imagenet_val_bsq',
        save_prediction=True,
        image_recon4video=False,
        junke_old=False,
        device='cuda',
        max_steps=1000000.0,
        log_every=1,
        visu_every=1000,
        ckpt_every=1000,
        default_root_dir='',
        compile='no',
        ema='no', # ?
        lr=0.0001,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0,
        optim_type='Adam',
        disc_optim_type=None,
        lr_min=0.0,
        warmup_lr_init=0.0,
        max_grad_norm=1.0,
        max_grad_norm_disc=1.0,
        disable_sch=False,
        patch_size=patch_size,
        temporal_patch_size: 4 by default,
        embedding_dim=256,
        codebook_dim=codebook_dim,
        num_quantizers=8,
        quantizer_type='MultiScaleBSQ',
        use_vae: False by default
        use_freq_enc=False,
        use_freq_dec=False,
        preserve_norm=False,
        ln_before_quant: False by default,
        ln_init_by_sqrt=False,
        use_pxsf=False,
        new_quant: True by default,
        use_decay_factor: False by default,
        mask_out=False,
        use_stochastic_depth: False by default,
        drop_rate: 0.0 by default,
        schedule_mode=schedule_mode,
        lr_drop=None,
        lr_drop_rate=0.1,
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach: False by default,
        use_out_phi=False,
        use_out_phi_res=False,
        use_lecam_reg=False,
        lecam_weight=0.05,
        perceptual_model='vgg16',
        base_ch_disc=64,
        random_flip: False by default, no randomflip,
        flip_prob=0.5,
        flip_mode='stochastic',
        max_flip_lvl: 1 by default,
        not_load_optimizer=False,
        use_lecam_reg_zero=False,
        freeze_encoder=False,
        rm_downsample=False,
        random_flip_1lvl=False,
        flip_lvl_idx=0,
        drop_when_test: False, bo drop during inference
        drop_lvl_idx: drop output idx after idx,
        drop_lvl_num=1,
        disc_version='v1',
        magvit_disc=False,
        sigmoid_in_disc=False,
        activation_in_disc='leaky_relu',
        apply_blur=False,
        apply_noise=False,
        dis_warmup_steps=0,
        dis_lr_multiplier=1.0,
        dis_minlr_multiplier=False,
        disc_channels=64,
        disc_layers=3,
        discriminator_iter_start=0,
        disc_pretrain_iter=0,
        disc_optim_steps=1,
        disc_warmup=0,
        disc_pool='no',
        disc_pool_size=1000,
        advanced_disc=False,
        recon_loss_type='l1',
        video_perceptual_weight=0.0,
        image_gan_weight=1.0,
        video_gan_weight=1.0,
        image_disc_weight=0.0,
        video_disc_weight=0.0,
        l1_weight=4.0,
        gan_feat_weight=0.0,
        perceptual_weight=0.0,
        kl_weight=0.0,
        lfq_weight: 0.0 for evaluation,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1,
        norm_type='group',
        disc_loss_type='hinge',
        use_checkpoint=False,
        precision='fp32',
        encoder_dtype='fp32',
        upcast_attention='',
        upcast_tf32=False,
        tokenizer='flux',
        pretrained=None,
        pretrained_mode='full',
        inflation_pe=False,
        init_vgen='no',
        no_init_idis=False,
        init_idis='keep',
        init_vdis='no',
        enable_nan_detector=False,
        turn_on_profiler=False,
        profiler_scheduler_wait_steps=10,
        debug=True,
        video_logger=False,
        bytenas='',
        username='',
        seed=1234,
        vq_to_vae=False,
        load_not_strict=False,
        zero=0,
        bucket_cap_mb=40,
        manual_gc_interval=1000,
        data_path=[''],
        data_type=[''],
        dataset_list=['imagenet'],
        fps=-1,
        dataaug='resizecrop',
        multi_resolution=False,
        random_bucket_ratio=0.0,
        sequence_length=16,
        resolution=[256, 256],
        batch_size=[1],
        num_workers=0,
        image_channels=3,
        codebook_size=codebook_size,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        commit_loss_beta=0.25,
        entropy_loss_ratio=0.0,
        base_ch=128,
        num_res_blocks: 2 by default,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        dropout_p=0.0,
        cnn_type: '2d' by default, no video,
        cnn_version='v1',
        conv_in_out_2d: 'no' by default, 'yes' for video which use a 2D conv for spatial downnsampling.
        conv_inner_2d: 'no' by default, 'yes' for video which use a 2D conv for spatial filtering.
        res_conv_2d: 'no' by default,
        cnn_attention: 'no' by default, no attention.
        cnn_norm_axis: 'spatial' by default, 'spatial-temporal' for video.
        flux_weight=0,
        cycle_weight=0,
        cycle_feat_weight=0,
        cycle_gan_weight=0,
        cycle_loop=0,
        z_drop=0.0
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        cnn_param = dict(
            cnn_type=args.cnn_type,
            conv_in_out_2d=args.conv_in_out_2d,
            res_conv_2d=args.res_conv_2d,
            cnn_attention=args.cnn_attention,
            cnn_norm_axis=args.cnn_norm_axis,
            conv_inner_2d=args.conv_inner_2d,
        )
        self.encoder = Encoder(
            ch=args.base_ch,
            ch_mult=args.encoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=args.use_checkpoint,
            use_vae=args.use_vae,
        )
        self.decoder = Decoder(
            ch=args.base_ch,
            ch_mult=args.decoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=args.use_checkpoint,
            use_freq_dec=args.use_freq_dec,
            use_pxsf=args.use_pxsf # pixelshuffle for upsampling
        )
        self.z_drop = nn.Dropout(args.z_drop)
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.codebook_dim = self.embed_dim = args.codebook_dim

        self.gan_feat_weight = args.gan_feat_weight
        self.video_perceptual_weight = args.video_perceptual_weight
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.use_vae = args.use_vae
        self.kl_weight = args.kl_weight
        self.lfq_weight = args.lfq_weight
        self.image_gan_weight = args.image_gan_weight # image GAN loss weight
        self.video_gan_weight = args.video_gan_weight # video GAN loss weight
        self.perceptual_weight = args.perceptual_weight
        self.flux_weight = args.flux_weight
        self.cycle_weight = args.cycle_weight
        self.cycle_feat_weight = args.cycle_feat_weight
        self.cycle_gan_weight = args.cycle_gan_weight

        self.flux_image_encoder = None
        
        if not args.use_vae:
            if args.quantizer_type == 'MultiScaleBSQ':
                self.quantizer = MultiScaleBSQ(
                    dim = args.codebook_dim,                        # this is the input feature dimension, defaults to log2(codebook_size) if not defined  
                    codebook_size = args.codebook_size,             # codebook size, must be a power of 2
                    entropy_loss_weight = args.entropy_loss_weight, # how much weight to place on entropy loss
                    diversity_gamma = args.diversity_gamma,         # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
                    preserve_norm=args.preserve_norm,               # preserve norm of the input for BSQ
                    ln_before_quant=args.ln_before_quant,           # use layer norm before quantization
                    ln_init_by_sqrt=args.ln_init_by_sqrt,           # layer norm init value 1/sqrt(d)
                    commitment_loss_weight=args.commitment_loss_weight, # loss weight of commitment loss
                    new_quant=args.new_quant,
                    use_decay_factor=args.use_decay_factor,
                    mask_out=args.mask_out,
                    use_stochastic_depth=args.use_stochastic_depth,
                    drop_rate=args.drop_rate,
                    schedule_mode=args.schedule_mode,
                    keep_first_quant=args.keep_first_quant,
                    keep_last_quant=args.keep_last_quant,
                    remove_residual_detach=args.remove_residual_detach,
                    use_out_phi=args.use_out_phi,
                    use_out_phi_res=args.use_out_phi_res,
                    random_flip = args.random_flip,
                    flip_prob = args.flip_prob,
                    flip_mode = args.flip_mode,
                    max_flip_lvl = args.max_flip_lvl,
                    random_flip_1lvl = args.random_flip_1lvl,
                    flip_lvl_idx = args.flip_lvl_idx,
                    drop_when_test = args.drop_when_test,
                    drop_lvl_idx = args.drop_lvl_idx,
                    drop_lvl_num = args.drop_lvl_num,
                )
                self.quantize = self.quantizer
                self.vocab_size = args.codebook_size
            else:
                raise NotImplementedError(f"{args.quantizer_type} not supported")


    def forward(self, x):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
        enc_dtype = ptdtype[self.args.encoder_dtype]

        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h, hs, hs_mid = self.encoder(x, return_hidden=True) # B C H W or B C T H W
        
        # detach 
        hs = [_h.detach() for _h in hs]
        hs_mid = [_h.detach() for _h in hs_mid]
        h = h.to(dtype=torch.float32)
        # print(z.shape)
        # Multiscale LFQ            
        z, all_indices, _, _, all_loss, _ = self.quantizer(h)
        x_recon = self.decoder(z)
        vq_output = {
            "commitment_loss": torch.mean(all_loss) * self.lfq_weight, # here commitment loss is sum of commitment loss and entropy penalty
            "encodings": all_indices, 
        }
        return x_recon, vq_output

    def encode_for_raw_features(self, x, scale_schedule, return_residual_norm_per_scale=False):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1

        enc_dtype = ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast("cuda", dtype=enc_dtype):
            # hs, hs_mid: middle state
            h, hs, hs_mid = self.encoder(x, return_hidden=True) # B C H W or B C T H W

        # detach
        hs = [_h.detach() for _h in hs]
        hs_mid = [_h.detach() for _h in hs_mid]
        h = h.to(dtype=torch.float32)
        return h, hs, hs_mid
    
    def encode(self, x, scale_schedule, return_residual_norm_per_scale=False):
        h, hs, hs_mid = self.encode_for_raw_features(x, scale_schedule, return_residual_norm_per_scale)
        # Multiscale LFQ
        z, all_indices, all_bit_indices, residual_norm_per_scale, all_loss, var_input = self.quantizer(h, scale_schedule=scale_schedule, return_residual_norm_per_scale=return_residual_norm_per_scale)
        return h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input

    def decode(self, z):
        x_recon = self.decoder(z)
        x_recon = torch.clamp(x_recon, min=-1, max=1)
        return x_recon
    
    def decode_from_indices(self, all_indices, scale_schedule, label_type):
        summed_codes = 0
        for idx_Bl in all_indices:
            codes = self.quantizer.lfq.indices_to_codes(idx_Bl, label_type)
            summed_codes += F.interpolate(codes, size=scale_schedule[-1], mode=self.quantizer.z_interplote_up)
        assert summed_codes.shape[-3] == 1
        x_recon = self.decoder(summed_codes.squeeze(-3))
        x_recon = torch.clamp(x_recon, min=-1, max=1)
        return summed_codes, x_recon

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--flux_weight", type=float, default=0)
        parser.add_argument("--cycle_weight", type=float, default=0)
        parser.add_argument("--cycle_feat_weight", type=float, default=0)
        parser.add_argument("--cycle_gan_weight", type=float, default=0)
        parser.add_argument("--cycle_loop", type=int, default=0)
        parser.add_argument("--z_drop", type=float, default=0.)
        return parser

