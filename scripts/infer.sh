#!/bin/bash
# set arguments for inference
pn=1M
model_type=echogen_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
model_path=checkpoints/echogen_2b.pth
vae_type=32
vae_path=pretrained/infinity/infinity_vae_d32reg.pth
cfg_i=2.5
cfg_t=2.5
tau=2.0
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=pretrained/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
content_feats_len=4096
content_feats_dim=16
semantic_feats_len=1025
semantic_feats_dim=768
content_ckpt_path=pretrained


python3 tools/run_echogen.py \
--pn ${pn} \
--model_path ${model_path} \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--cfg_t ${cfg_t} \
--cfg_i ${cfg_i} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--content_feats_len ${content_feats_len} \
--content_feats_dim ${content_feats_dim} \
--semantic_feats_len ${semantic_feats_len} \
--semantic_feats_dim ${semantic_feats_dim} \
--content_ckpt_path ${content_ckpt_path} \
--prompt <cond_image_path> \
--seed 42 \
--control_img <your_prompt> \
--save_file tmp.jpg 