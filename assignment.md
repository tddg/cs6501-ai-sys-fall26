---
layout: page
title: Inference Speedrun
nav_order: 6
---

# Assignment: LLM Inference Speedrun
{: .no_toc }

**Out:** 08/26 · **Due:** 09/09, 11:59 PM ET · **Weight:** 20%

<details open markdown="block">
  <summary>Table of contents</summary>
1. TOC
{:toc}
</details>


## FAQs

*This section will be updated as questions come in on the course forum.
Check here before posting.*

## Overview

You are given a fixed language model, **Qwen/Qwen3-4B-Instruct-2507**,
and a deliberately naive OpenAI-compatible inference server. Your job is to
make serving this model on a single **NVIDIA T4 (16 GB)** as fast as you
can. You own the entire serving stack: HTTP handling, request scheduling,
batching, KV-cache management, quantization, GPU kernels — replace any of
it, or all of it.

This is the core engineering loop of every production LLM serving
system (vLLM, SGLang, TGI, TensorRT-LLM), compressed to a scale where
one person can build it in weeks on a free GPU, with the help of AI
coding tools. The classic optimizations each map to a measurable
scenario, so you will *see* each technique pay off:

| Technique | What it moves |
|---|---|
| Continuous / wave batching | burst throughput (the single biggest win: the naive server serializes concurrent requests) |
| CUDA graphs, static KV cache, kernel fusion | single-stream decode latency (the naive server wastes ~3× on launch overhead) |
| Weight quantization (int4/int8) | decode latency (T4 decode is memory-bandwidth-bound: ~25 ms/token floor at fp16) |
| Prefill engineering (attention kernels, int8 tensor cores) | time-to-first-token on long prompts |
| Speculative decoding (prompt-lookup) | single-stream latency, no draft model needed |


**You may not use existing serving engines** (vLLM, SGLang, TGI,
TensorRT-LLM, llama.cpp, ExLlama, etc.). The grading environment
contains only a pinned lightweight stack — torch (+ triton),
transformers, tokenizers, safetensors, accelerate, numpy, fastapi,
uvicorn, httpx — plus the CUDA toolchain (nvcc, ninja) to compile
**your own** kernels.  Submissions are source-only; there is no
package index at grading time.  The point is that the batching,
caching, quantization, and kernels are *your* work.



## Part 0: Environment setup

The grading hardware is one **T4 (16 GB, SM75), 8 vCPU, 32 GB RAM**. Two
free/cheap dev options match it:

- **Google Colab (free tier)** — the free T4 is the same GPU as the
  grader. Recommended starting point.
- **GCP `n1-standard-8` + T4** (spot ≈ $0.19/h) — same shape as the
  grader; use this when you outgrow Colab session limits. We provide
$50 of GCP credit to each student. 

T4 notes you will run into: fp16 only (no bf16, no FP8), no
FlashAttention-2 (Turing SM75), but int4/int8 kernels, Triton, CUDA
graphs, and torch.compile all work.

Setup:

```sh
pip install torch transformers accelerate safetensors fastapi uvicorn httpx
python3 -c "from huggingface_hub import snapshot_download; \\
  snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='models/qwen3-4b')"
```

The model is the official
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(~8 GB, ungated, no HF account needed). Use the **exact** `-Instruct-2507`
variant. **The plain `Qwen3-4B` or `-Thinking-2507` siblings are
different models and will fail the accuracy gate.**

Download the starter kit from the course dashboard (link TBD). It
contains:

| File | What it is |
|---|---|
| `SPEC.md` | the precise grading contract (API, scenarios, gate, scoring) |
| `baseline_server.py` | the naive server you are beating — **read it first**; every design decision in it is an optimization opportunity |
| `start_server.sh` | entry point; your submission must keep this contract |
| `prompts_public.json` | public examples of the graded workloads |
| `bench_client.py` | the exact measurement client the grader uses |
| `local_test.py` | runs the graded workload shapes against your server locally |



## Part 1: Run and understand the baseline

Boot the baseline and measure it with the same methodology the grader
uses:

```sh
export MODEL_DIR=~/models/qwen3-4b PORT=8000
sh start_server.sh &
python3 local_test.py --port 8000 --model-dir $MODEL_DIR
```

On a T4 you should see roughly **78 ms/token** time-per-output-token
(TPOT) for interactive decode, **8.8 s** time-to-first-token (TTFT)
on a 4K-token prompt, and **~14 tok/s** aggregate throughput under 32
concurrent requests. Before writing any code, profile where that time
goes; the baseline's three central weaknesses (a global lock,
token-by-token everything, fp16-only) each map to one graded
scenario.



## Part 2: Optimize

Rebuild the serving stack behind `start_server.sh`. Your server must keep
the API contract (see `SPEC.md`): `GET /health`, and
`POST /v1/chat/completions` with streaming and non-streaming responses,
honoring `max_tokens`, greedy at `temperature: 0`. Everything else,
architecture, scheduling, memory layout, precision, kernels, is yours.


### How you are measured

The grader fires three fixed hidden traffic patterns at your server (you
never implement or configure these; public examples of each shape are in
the kit). All timing is external, at SSE-chunk arrival; token counts are
re-tokenized from your returned text.

| Scenario | The situation | Metric |
|---|---|---|
| **Chat** | one user watching a reply stream (250-token prompts, 256-token outputs, sequential) | ms per output token ↓ |
| **Long prompt** | a 4K-token document; the user waits for the answer to start | seconds to first token ↓ |
| **Burst** | 32 requests arrive at once (128-token outputs) | total tokens/sec ↑ |

**Score = geometric mean of your three speedups over the baseline
server, measured on the same physical GPU as your run** (this cancels
hardware differences between grading machines — the baseline is
re-measured per GPU, so your ratio is hardware-independent). The
baseline scores 1.0×. The instructor reference implementation (CUDA-graph
decode + wave batching, no quantization) scores ≈ 2.4×; a strong
submission adding quantization, real prefill work, and better batching
has headroom to ≈ 5–8×.

Server startup is untimed (5-minute cap); an optional `setup.sh` build
step gets 10 untimed minutes with no network. Per-scenario wall-clock
caps exist at ~2× baseline speed — exceeding one is a TIMEOUT and the run
does not score.


### The accuracy gate

Before any performance measures, **200 hidden multiple-choice
questions** are sent through your server under a fixed few-shot,
direct-answer, greedy protocol. The unmodified model answers 100% of
them; crude int4 round-to-nearest quantization still answers 97.5%.
Your run scores only at **≥ 95%**. Quantize, fuse, and reorder math
freely — serve something that is no longer this model, and the gate
catches it. Gate-failing runs are marked INCORRECT and never reach
the leaderboard.

Forbidden regardless of the gate: proxying to another model or service
(the grading sandbox has no network), canned responses to benchmark
prompts (they are hidden), and manipulating timing rather than actual
serving speed.

## Deliverables

Submit through the course dashboard (link TBD; register with your invite
token):

```
submission.tar.gz
├── start_server.sh   # required: starts your server on 127.0.0.1:$PORT
│                     #   serving the model from $MODEL_DIR (read-only)
├── setup.sh          # optional: untimed build step (10 min, no network)
└── ...               # your engine source (source-only, no vendored wheels)
```

One run in flight at a time; a full grading run takes ~6–12 minutes and
the dashboard shows live phases (gate → chat → long-prompt → burst) plus
per-scenario results, your speedups, and a grading timeline. The
anonymous leaderboard ranks your **most recent passing run** — you cannot
keep a lucky outlier.

Also submit (via Canvas, TBD): a short report (~2 pages) describing what
you built, which optimizations moved which scenario, and one experiment
that surprised you.

## Grading

*Draft — bands TBD.*

| Component | Weight |
|---|---|
| Passing the accuracy gate with any speedup ≥ 1.0× | TBD % |
| Speedup milestones (e.g., ≥ 1.5× / ≥ 2.5× / ≥ 4×) | TBD % |
| Report | TBD % |
| Leaderboard placement (top-N bonus) | TBD % |

## Policies

- **AI assistance:** TBD.
- **Collaboration:** discuss ideas freely; code and submissions are
  individual. Do not share engine code.
- **Submission budget:** TBD (grading runs cost real GPU money; budget
  your iterations and use `local_test.py` first).
- Late policy: per syllabus.
