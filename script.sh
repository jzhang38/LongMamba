accelerate launch --num_processes 8  finetune_torch.py --batch-size 1 --gradient-accumulate-every 1  --output-dir ./output/slim_delta_1.0_legnth_1024_step_400_lr_1e-5 \
--wandb longmamba  --model state-spaces/mamba-370m --dataset PY007/tokenized_slim6B_train_neox_1024  --max-train-steps 400   --learning-rate 1e-5
