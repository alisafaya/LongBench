import os
import json
import random
import logging
import argparse

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from neurocache_utils import initialize_neurocache_model, prefill_neurocache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--datasets", nargs="+", type=str, default=None, help="Datasets to predict on")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length of input sequence")

    # Neurocache arguments
    parser.add_argument("--pretrained_neurocache", type=str, default=None)
    parser.add_argument("--attention_layers", type=str, default=None)
    parser.add_argument("--cache_layers", type=str, default=None)
    parser.add_argument("--cache_size", type=int, default=None)
    parser.add_argument("--cache_type", type=str, default=None, choices=["FIFO", "LRU"])
    parser.add_argument(
        "--cache_dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--context_size", type=int, default=None)
    parser.add_argument("--neighborhood_size", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--segment_size", type=int, default=1024)

    # LoRA arguments
    parser.add_argument("--add_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_modules", type=str, default="gate_proj,up_proj,down_proj")

    return parser.parse_args()

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_model_and_tokenizer(path, device, args):
    try:
        config = AutoConfig.from_pretrained(path)
        config.use_cache = True
        config._flash_attn_2_enabled = False # fp32 does not support flash_attn
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", use_fast=False)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # fp32 gives consistent results: https://github.com/huggingface/transformers/issues/25921
        if args.pretrained_neurocache:
            logging.info("Loading model with pretrained neurocache...")
            model = initialize_neurocache_model(args, device)
        else:
            logging.info("Loading model without neurocache...")
            model = AutoModelForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.float32).to(device) 
    
        model.eval()
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model/tokenizer: {e}")
        raise

def build_chat(tokenizer, prompt, model_name):
    if "llama2" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def prepare_prompts(tokenizer, json_obj, max_length, prompt_format, dataset, model_name):
    try:
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"] and "chat" in model_name.lower():
            prompt = build_chat(tokenizer, prompt, model_name)

        return prompt
    except Exception as e:
        logging.error(f"Error in prepare_inputs: {e}")
        return None

def generate_predictions(model, prompts, max_gen, tokenizer):
    try:
        if max_gen < -1:
            eos_token_id = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
        else:
            eos_token_id = tokenizer.eos_token_id

        context_length = prompts["input_ids"].shape[-1]
        with torch.no_grad():
            output = model.generate(
                **prompts,
                max_new_tokens=max_gen,
                min_new_tokens=1,
                eos_token_id=eos_token_id,
                num_beams=1,
                do_sample=False,
            )

        preds = tokenizer.batch_decode(output[:, context_length:].cpu(), skip_special_tokens=True)
        return preds
    except Exception as e:
        logging.error(f"Error in generate_predictions: {e}")
        raise e

def predict_dataset(args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, batch_size=4):

    token_length = max_length
    if args.pretrained_neurocache:
        token_length += model.neurocache_config.cache_size

    prompts = []
    for json_obj in tqdm(data, desc=f"Preparing inputs for {dataset}"):
        prompt = prepare_prompts(tokenizer, json_obj, token_length, prompt_format, dataset, model_name)
        if prompt is not None:
            prompts.append(prompt)

    # sort prompts by length to reduce padding. in descending order 
    # save indices for resorting later
    sorted_indices = np.argsort([len(prompt) for prompt in prompts])[::-1]
    prompts = [prompts[i] for i in sorted_indices]

    # generate predictions in batches
    preds = []
    input_ids = []
    attn_masks = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating predictions for {dataset}"):
        batch = prompts[i:i + batch_size]

        # pad batch. ensure equal batchsize to avoid neurocache reinitialization of cache
        if i + batch_size > len(prompts):
            batch.extend([batch[-1]] * (batch_size - len(batch)))

        prepared_input = tokenizer(batch, padding=True, truncation=True, 
                                    max_length=(token_length - max_gen), return_tensors="pt").to(device)

        # prefill cache
        if args.pretrained_neurocache:
            prepared_input = prefill_neurocache(args, model, prepared_input, max_length - max_gen)
            with model.generation_mode():
                preds.extend(generate_predictions(model, prepared_input, max_gen, tokenizer))
            import ipdb; ipdb.set_trace()
        else:
            preds.extend(generate_predictions(model, prepared_input, max_gen, tokenizer))

        input_ids.extend(prepared_input["input_ids"].tolist())
        attn_masks.extend(prepared_input["attention_mask"].tolist())

    # resort predictions
    preds = [preds[i] for i in np.argsort(sorted_indices)]
    input_ids = [input_ids[i] for i in np.argsort(sorted_indices)]
    attn_masks = [attn_masks[i] for i in np.argsort(sorted_indices)]

    json_preds = []
    for i, (json_obj, pred) in enumerate(zip(data, preds)):
        json_preds.append({
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
            "input_ids": input_ids[i],
            "attention_mask": attn_masks[i],
        })

    return json_preds


def main():
    seed_everything()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configurations
    try:
        with open("config/dataset2prompt.json") as f:
            dataset2prompt = json.load(f)
        with open("config/dataset2maxlen.json") as f:
            dataset2maxlen = json.load(f)
    except Exception as e:
        logging.error(f"Error loading configurations: {e}")
        raise

    model_name = args.model
    model, tokenizer = load_model_and_tokenizer(model_name, device, args)
    max_length = (model.config.max_position_embeddings - 2) if args.max_length is None else args.max_length

    if args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", \
                "gov_report", "multi_news", "trec", "triviaqa", "samsum",\
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    if not args.e:
        if args.datasets is None:
            datasets.extend(["narrativeqa", "musique", "qmsum"])

        output_dir = "pred"
        logging.info("Evaluating on LongBench")
    else:
        output_dir = "pred_e"
        logging.info("Evaluating on LongBench-E")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_name = model_name.replace("/", "_")

    for dataset in datasets:
        dataset_output_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

        data = load_dataset("THUDM/LongBench", f"{dataset}_e" if args.e else dataset, split="test")
        out_path = os.path.join(dataset_output_dir, f"{dataset}.jsonl")

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        logging.info(f"Predicting on dataset: {dataset}")
        preds = predict_dataset(
            args,
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            args.batch_size
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")

        logging.info(f"Predictions saved to {out_path}.")

if __name__ == "__main__":
    main()
