# Progress Log

## Session 1 — Data Collection

### What I measured
- Qwen2.5-VL-7B under NF4 4-bit quantization, 50 samples
- Vision encoder: 653ms median (18% of total)
- Language decoder: 5510ms median (81% of total)
- Peak VRAM: 7906MB

### Experiments run
1. Both=4bit (NF4 baseline) — 50 samples — COMPLETE
2. Vision=FP16, Decoder=4bit — 50 samples — COMPLETE  
3. Vision=4bit, Decoder=FP16 — OOM during warmup — DOCUMENTED
4. FP16 full model — OOM on load — DOCUMENTED

### What broke
- FP16 full model: OOM, requires 14GB+, GPU has 14.56GB with no room for activations
- Vision=4bit Decoder=FP16: loads 13.2GB then OOMs on first inference allocation
- TextVQA dataset: incompatible with current datasets library version, used custom image set instead

### Key finding
Converting 162 vision encoder layers from 4bit to FP16 produced 2ms latency 
delta and 0MB VRAM delta. Vision encoder precision is irrelevant to serving 
performance. Decoder is the hard constraint in memory and latency.

### Next steps
- Build GitHub repo
- Write README
- LinkedIn post
