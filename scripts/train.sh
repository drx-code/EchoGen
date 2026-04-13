export GIT_PYTHON_REFRESH=quiet
set -x

export PYTHONFAULTHANDLER=1  # Optional: helps catch Python error

BED=experiments/checkpoints
LOCAL_OUT=experiments/local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

wandb offline
exp_name=test
bed_path=experiments/checkpoints/${exp_name}/
data_path='data/toy_data_example/splits'
video_data_path=''
local_out_path=$LOCAL_OUT/${exp_name}

torchrun \
--nproc_per_node=8 \
--nnodes=${NUM_NODES} \
--node_rank=${NODE_RANK} \
--master_addr=${MASTER_ADDR} \
train.py \
--ep=160 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=6e-5 \
--pn 1M \
--model=2bc8 \
--lbs=4 \
--workers=8 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path=pretrained/flan-t5-xl \
--vae_type 32 \
--vae_ckpt=pretrained/infinity/infinity_vae_d32reg.pth  \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=3 \
--save_model_iters_freq 10000 \
--log_freq=10 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=128 \
--rush_resume=pretrained/infinity/infinity_2b_reg.pth \
--content_feats_len=4096 \
--content_feats_dim=16 \
--semantic_feats_len=1025 \
--semantic_feats_dim=768 \
--content_ckpt_path=pretrained