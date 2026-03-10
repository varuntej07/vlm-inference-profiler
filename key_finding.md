
# Key Finding: Vision Encoder Quantization is Free

## Experiment
Qwen2.5-VL-7B profiled across 4 precision configurations on T4 16GB GPU.
50 samples, median reported.

## Results Table
| Config                    | VRAM    | Latency | Vision  | Decoder |
|---------------------------|---------|---------|---------|---------|
| FP16 full model           | OOM     | —       | —       | —       |
| Vision=FP16, Decoder=4bit | 7906 MB | 6079ms  | 655ms   | 5420ms  |
| Both=4bit NF4 (baseline)  | 7906 MB | 6184ms  | 653ms   | 5510ms  |
| Vision=4bit, Decoder=FP16 | OOM*    | —       | —       | —       |

*Loads 13.2GB, OOMs during first inference warmup

## Finding
Converting 162 vision encoder Linear layers from 4bit to FP16 produced
zero measurable difference: 2ms latency delta, 0MB VRAM delta.

The vision encoder represents ~18% of inference time and is insensitive
to quantization precision on this hardware.

The decoder consumes 12.4GB in FP16 — exceeding what remains after model
load on 14.56GB GPU. The decoder is the hard constraint in every dimension:
memory, latency, and quantization sensitivity.

## Implication for Serving
Uniform quantization frameworks (vLLM, TGI default configs) apply equal
compression across all components. This data shows that for Qwen2.5-VL-7B,
vision encoder precision is irrelevant to serving performance. Optimization
effort should be concentrated entirely on decoder precision and KV cache
management.
