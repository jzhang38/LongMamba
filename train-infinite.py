import argparse
import copy
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, default_data_collator, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from experimental.mamba_lm import MambaLMHeadModel
from experimental.mamba_module import Block
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import json
import math
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000)) 
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block,
        },
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        fsdp_plugin=fsdp_plugin,
    )
    accelerator.init_trackers(
        project_name=args.wandb)
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")



    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]
        
    model = MambaLMHeadModel.from_pretrained(
        args.model,
        device=accelerator.device,
    )
    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    
    train_dataset = train_dataset.shard(num_shards=accelerator.num_processes, contiguous=True, index=accelerator.process_index)
    print("Dataset Size:", len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
    )



    model = accelerator.prepare(model)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optim, num_warmup_steps=args.warmup_steps)
    optim, scheduler = accelerator.prepare(
        optim, scheduler)



    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    )

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    accelerator.print(f"Total batch size: {total_batch_size}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0





    model.train()
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        loss_log = None
        with accelerator.accumulate(model):
            loss_func = CrossEntropyLoss(inplace_backward=True)
            logits = model(batch["input_ids"][..., :-1], delta_ratio=args.delta_ratio).logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), batch["input_ids"][..., 1:].view(-1))
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                loss_log = {"loss": loss.item(), "ppl": math.exp(loss.item())}
                accelerator.log(loss_log, step=completed_steps)


            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            completed_steps += 1


        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = accelerator.get_state_dict(model, unwrap=True)
        if accelerator.is_main_process:
            model_path = os.path.join(args.output_dir, 'pytorch_model.bin')
            torch.save(state_dict, model_path)
            config_path = os.path.join(args.output_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(model.config.__dict__, f)

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--model", type=str,
                      default="state-spaces/mamba-2.8b-slimpj")
    args.add_argument("--dataset", type=str,
                      default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    args.add_argument("--num-proc", type=int, default=32)
    args.add_argument("--lr-schedule", type=str,
                      choices=["linear", "constant"], default="linear")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--debug", action="store_true")
    args.add_argument("--delta_ratio", type=float, default=None)
    main(args.parse_args())