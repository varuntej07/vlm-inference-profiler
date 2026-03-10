# Reusable profiling and quantization utilities for VLM inference analysis

import torch
import csv
from dataclasses import dataclass


@dataclass
class InferenceProfile:
    vision_encoder_ms: float = 0.0
    projection_ms: float = 0.0
    decoder_ms: float = 0.0
    total_ms: float = 0.0
    peak_vram_mb: float = 0.0
    num_tokens_generated: int = 0


def profile_single_inference(model, processor, image_path, question, max_new_tokens=50):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": question},
        ],
    }]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=image_inputs,
        videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    ev_start        = torch.cuda.Event(enable_timing=True)
    ev_after_vision = torch.cuda.Event(enable_timing=True)
    ev_after_decode = torch.cuda.Event(enable_timing=True)

    hooks = []
    def make_hook(event):
        def hook(module, input, output): event.record()
        return hook

    hooks.append(model.model.visual.register_forward_hook(make_hook(ev_after_vision)))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        ev_start.record()
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        ev_after_decode.record()
        torch.cuda.synchronize()

    profile = InferenceProfile()
    profile.vision_encoder_ms = ev_start.elapsed_time(ev_after_vision)
    profile.decoder_ms        = ev_after_vision.elapsed_time(ev_after_decode)
    profile.total_ms          = ev_start.elapsed_time(ev_after_decode)
    profile.peak_vram_mb      = torch.cuda.max_memory_allocated() / 1e6
    profile.num_tokens_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]

    for h in hooks:
        h.remove()

    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )[0]

    return profile, response


def run_profiling_loop(model, processor, dataset, output_csv,
                       config_name, max_new_tokens=50, n_warmup=2):
    for _ in range(n_warmup):
        profile_single_inference(
            model, processor,
            dataset[0]["image"], dataset[0]["question"],
            max_new_tokens=20
        )

    fieldnames = [
        "config", "sample_id", "question",
        "vision_encoder_ms", "decoder_ms", "total_ms",
        "peak_vram_mb", "tokens_generated", "response_text",
        "expected", "exact_match"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, sample in enumerate(dataset):
            print(f"Sample {i+1}/{len(dataset)}...", end=" ")
            try:
                profile, response = profile_single_inference(
                    model, processor,
                    sample["image"], sample["question"],
                    max_new_tokens=max_new_tokens
                )
                exact_match = int(sample["expected"].lower() in response.lower())
                writer.writerow({
                    "config": config_name, "sample_id": i,
                    "question": sample["question"],
                    "vision_encoder_ms": round(profile.vision_encoder_ms, 2),
                    "decoder_ms":        round(profile.decoder_ms, 2),
                    "total_ms":          round(profile.total_ms, 2),
                    "peak_vram_mb":      round(profile.peak_vram_mb, 1),
                    "tokens_generated":  profile.num_tokens_generated,
                    "response_text":     response.strip(),
                    "expected":          sample["expected"],
                    "exact_match":       exact_match,
                })
                f.flush()
                print(f"done — {profile.total_ms:.0f}ms")
            except Exception as e:
                print(f"FAILED: {e}")


def dequantize_vision_encoder(model):
    import bitsandbytes as bnb
    converted = 0
    for name, module in model.named_modules():
        if "model.visual" not in name:
            continue
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        weight_fp16 = bnb.functional.dequantize_4bit(
            module.weight.data, module.weight.quant_state
        ).to(torch.float16)
        new_linear = torch.nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None
        ).to(torch.float16).to(model.device)
        new_linear.weight = torch.nn.Parameter(weight_fp16)
        if module.bias is not None:
            new_linear.bias = torch.nn.Parameter(module.bias.data.to(torch.float16))
        setattr(parent, attr, new_linear)
        converted += 1
    return model, converted
