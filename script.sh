accelerate launch --num_processes 8  finetune_torch.py --batch-size 1 --gradient-accumulate-every 1  --output-dir ./output/slim_delta_1.0_legnth_1024_step_400_lr_1e-5 \
--wandb longmamba  --model state-spaces/mamba-370m --dataset PY007/tokenized_slim6B_train_neox_1024  --max-train-steps 400   --learning-rate 1e-5



accelerate launch --num_processes 8  train-infinite.py --batch-size 1 --gradient-accumulate-every 2  --output-dir ./output/130m-infinite-lr-6e-5-1024-window-bs-16k-step200 \
--wandb longmamba  --model state-spaces/mamba-370m --dataset PY007/tokenized_slim6B_train_neox_1024  --max-train-steps 200   --learning-rate 6e-5


 2/400 [00:18<57:56,  8.73s/it, loss=3.05, ppl=21.2]

   | 33/400 [00:09<00:48,  7.63it/s, loss=1.88, ppl=6.52]

   22/400 [00:07<00:49,  7.57it/s, loss=3.16, ppl=23.7]



   python eval.py \
    --tokenized PY007/tokenized_proof_pile_test_neox \
    --dataset-min-tokens 65536 \
    --samples 20 \
    --output-file data/original_mamba.csv \
    --min-tokens 4096 \
    --max-tokens 65536 \
    --tokens-step 4096 \
    --truncate \
    -m state-spaces/mamba-130m