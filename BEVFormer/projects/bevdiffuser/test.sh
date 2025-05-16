set -e

export CUDA_VISIBLE_DEVICES=0

BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"

CHECKPOINT_DIR="../../train/BEVDiffuser_BEVFormer_tiny/checkpoint-50000"

BEV_CHECKPOINT="../../train/BEVDiffuser_BEVFormer_tiny/checkpoint-50000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

python -m torch.distributed.launch --master_port 9999 test_bev_diffuser.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 5 \
    --denoise_timesteps 5 \
    --num_inference_steps 5 \
    # --use_classifier_guidence \


