#!/usr/bin/env python
# coding=utf-8
import multiprocessing
import json
import os
from dataclasses import field, dataclass
from functools import partial
import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser
from solve import DecodingArguments, solve
from task1 import AnswerTask,ChoiceTask
from accelerate import Accelerator
import sys
import setproctitle

setproctitle.setproctitle('')



@dataclass
class MainArguments:
    data_file: str = field(default="./benchmark/gsm8k/test.jsonl")
    model_name_or_path: str = field(default="model/mistralai/Mistral-7B-Instruct-v0.3")
    batch_size: int = field(default=64)
    output_fname: str = field(default="outputs/model_predictions.jsonl")
    result_path: str = field(default="result.txt")
    gpu_id:str = field(default="3")
    model:str= field(default="Mistral-7B-Instruct-v0.3")


def encode_function(example, tokenizer, task):
    prompt = task.encode_prompt(example)
    # print(prompt)
    tokenized = tokenizer(prompt, return_tensors='pt')
    input_ids = tokenized.input_ids
    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


@torch.no_grad()
def main():
    parser = HfArgumentParser((MainArguments, DecodingArguments))
    main_args, decoding_args = parser.parse_args_into_dataclasses()

    print(main_args.output_fname)
    if os.path.exists(main_args.output_fname):
        print(f"{main_args.output_fname} exist!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(0)
    if 'gsm8k' in main_args.data_file or 'MultiArith' in main_args.data_file or 'MATH' in main_args.data_file:
        task = AnswerTask(encode_format=decoding_args.encode_format,decoding=decoding_args.decoding,data_file=main_args.data_file,model=main_args.model)
    else:
        task = ChoiceTask(encode_format=decoding_args.encode_format,decoding=decoding_args.decoding,model=main_args.model)
    # Initialize Accelerator with device_placement=True (this helps in multi-GPU configurations)
    accelerator = Accelerator(device_placement=True, split_batches=False)

    gpu_id=main_args.gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(main_args.model_name_or_path, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        main_args.model_name_or_path, torch_dtype=torch.bfloat16
    )
    
    # Manually move model to 'cuda:3'
    model = model.to(device)

    # Ensure that accelerator prepares the model for the right device (this helps in multi-GPU setups)
    model = accelerator.prepare(model)  # Automatically places the model on the correct device

    # Load dataset
    raw_dataset = load_dataset("json", data_files={'test': main_args.data_file})['test']
    encode_function_partial = partial(
        encode_function,
        tokenizer=tokenizer,
        task=task,
    )
    lm_dataset = raw_dataset.map(
        encode_function_partial,
        batched=False,
        num_proc=16,
        remove_columns=[name for name in raw_dataset.column_names if name not in ["input_ids", "attention_mask"]],
        desc="Tokenizing data",
    )

    # Generate
    dataloader = DataLoader(
        lm_dataset, shuffle=False, batch_size=main_args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    )

    outputs_all = []
    accs_all = []
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for batch in pbar:
        if batch is None:
            continue  # Skip empty batches

        # Move everything to 'cuda:6' (ensures everything is on the correct device)
        batch = {
            k: v.to(device) if v is not None else torch.zeros((1,), device=device)
            for k, v in batch.items()
        }

        # Now the model and batch are on the correct device
        outputs = solve(model, tokenizer, task, batch, args=decoding_args,device=device)

        n = len(outputs_all)

        for i, output in enumerate(outputs):
            outputs_all.append(output)
            
        pbar.set_postfix(acc=np.mean(accs_all))

    os.makedirs(os.path.dirname(main_args.output_fname), exist_ok=True)
    with open(main_args.output_fname, "w") as f:
        for output in outputs_all:
            f.write(json.dumps(output) + '\n')
    

if __name__ == "__main__":
    main()
