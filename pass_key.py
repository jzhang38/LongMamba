
import os
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
import pandas as pd
from modeling.mamba_lm import MambaLMHeadModel
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="PY007/LongMamba_16384_bs128_step400")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--max_tokens', type=int, default=32768, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=2000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')

    args = parser.parse_args()
    return args


def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids
    generation_output = model.generate(
        input_ids=input_ids, max_length=answer_ids.shape[-1] + input_ids.shape[-1]
    )
    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    model_answer = tokenizer.decode(model_answer).strip()
    gold_answer = tokenizer.decode(answer_ids[0]).strip()
    is_correct = (model_answer == gold_answer)
    return is_correct, len_token


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)




    # Load model and tokenizer
    model = MambaLMHeadModel.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device=device,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
    )

    total_test_points = args.max_tokens // args.interval
    all_accuries = []
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        # 10 diffierent n_garbage_prefix for each n_garbage that uniformly distributed
        avg_tokens = None
        for n_garbage_prefix in range(0, n_garbage, n_garbage // 10):
            passed_tests = 0
            total_tokens = 0
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=n_garbage, seed=k)
                passed_tests += is_correct
                total_tokens += len_tokens
            avg_tokens = total_tokens//args.num_tests if avg_tokens is None else avg_tokens
            accuracy = float(passed_tests)/args.num_tests
            depth = n_garbage_prefix/n_garbage
            print("accuracy on the token length %d, depth %f, is %f"%(avg_tokens,depth, accuracy))
            result = {"Context Length": avg_tokens, "Document Depth": round(depth*100, -1),"Score": passed_tests}
            all_accuries.append(result)
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'}
    )

    # More aesthetics
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    plt.savefig(f"data/heatmap_{args.max_tokens}.png".format(args.max_tokens))
if __name__ == "__main__":
    args = parse_config()
    main(args)