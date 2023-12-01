import logging

import torch

from peft import LoraConfig, inject_adapter_in_model
from neurocache import NeurocacheModelForCausalLM, OnDeviceCacheConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def initialize_neurocache_model(args, device):
    config = AutoConfig.from_pretrained(args.model)
    config.use_cache = True
    config._flash_attn_2_enabled = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=torch.float32
    )

    overrides = {
        "cache_size": args.cache_size,
        "cache_type": args.cache_type,
        "cache_dtype": args.cache_dtype,
        "neighborhood_size": args.neighborhood_size,
        "context_size": args.context_size,
        "topk": args.topk,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    neurocache_config = OnDeviceCacheConfig.from_pretrained(
        args.pretrained_neurocache, **overrides
    )

    if args.add_lora:
        # Add LoRA to the main model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules.split(","),
            lora_dropout=args.lora_dropout,
            layers_to_transform=neurocache_config.attention_layers,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logging.info(f"LoRA Config: {lora_config}")
        model = inject_adapter_in_model(lora_config, model, "neurocache")

    logging.info(f"Loading neurocache from {args.pretrained_neurocache}")
    model = NeurocacheModelForCausalLM.from_pretrained(
        model, args.pretrained_neurocache, config=neurocache_config
    )
    logging.info(f"Neurocache model initialized.")
    logging.info(f"Neurocache config: {model.neurocache_config}")

    return model.to(device)


def prefill_neurocache(args, model, inputs, last_chunk_len):
    # Pre-fill neurocache

    # First, we segment the input into chunks of size args.segment_size
    # Then, we run the model on each chunk to fill the neurocache
    # Except for the last chunk, which is run in generation mode.
    last_chunk = {k: v[:, -last_chunk_len:] for k, v in inputs.items()}

    if last_chunk_len > len(inputs["input_ids"][0]):
        # If the input is shorter than args.max_length,
        # we don't need to pre-fill the neurocache
        return last_chunk

    if last_chunk_len > 0:
        inputs = {k: v[:, :-last_chunk_len] for k, v in inputs.items()}
    else:
        last_chunk = None

    segments = segment_inputs(args, inputs)
    for i, seg in enumerate(segments):
        # reset the cache at the beginning of each document
        sos = (
            torch.tensor([int(i == 0)] * seg["input_ids"].shape[0])
            .bool()
            .to(seg["input_ids"].device)
        )
        with torch.no_grad():
            model(**seg, start_of_sequence=sos)

    return last_chunk


def segment_inputs(args, inputs):
    # Segment the input into chunks of size args.segment_size
    # and return the segments as a list of dictionaries
    # with the same keys as inputs.
    def split(tensor, size):
        residual = tensor.shape[1] % size

        segments = ()
        if residual > 2:
            segments += (tensor[:, :residual],)

        if tensor.shape[1] > residual:
            segments += torch.split(tensor[:, residual:], size, dim=1)

        return segments

    segments = {}
    for k in inputs:
        segments[k] = split(inputs[k], args.segment_size)

    return [
        {k: v[i] for k, v in segments.items()}
        for i in range(len(segments[list(segments.keys())[0]]))
    ]
