# DeepSeek-V2-Lite Device Attention

> **TL;DR:** On the NCCL backend, steady decode keeps attention, the KV cache, and every RMSNorm on rank 0's GPU, with one slot of a fixed bf16 KV pool per request, so rows at different positions share one decode step. Attention stays bitwise-equal to the host oracle, and TPOT on the #466 short HTTP contract drops 72-74%. Prefill still runs on the host (#452).

Last touched: 2026-10

## Scope

The device path covers NCCL steady decode for up to eight live requests. Prefill and the host-staged backend still use host attention, and host-staged remains the correctness oracle. `PEGAINFER_DSV2_LITE_DECODE_ATTENTION=host` ("the switch" below) keeps NCCL decode on host attention for A/B and emergency rollback.

Before this change, every decode layer copied the q / kv_a / kv_b projections to the host, ran RMSNorm and attention on the CPU, and copied the result back. The host batch path also needed one shared position, so rows at different positions were split into groups.

## Design

```text
after host prefill:
  take a free slot, copy the prompt's K/V into it as bf16, free the host copy

each decode step:
  upload_decode_rows: slot_ids[row], positions[row]       (once per step)
  per layer on rank 0:
    device RMSNorm
    q = gemm(q_proj), kv_a = gemm(kv_a_proj)
    compressed = kv_norm(kv_a[:512])
    kv_b = gemm(kv_b_proj, compressed)
    decode_attention: append K/V at positions[row], attend over 0..=positions[row]
  advance every slot by one token
```

`DeviceKvPool` owns per-layer K and V buffers laid out `[slot][position][head][dim]`, with K at 192 dims per head (128 nope + 64 rope) and V at 128. It stores expanded per-head K/V. The 512-wide compressed latent is only an intermediate projection. Eight slots x 4096 tokens x 27 layers is about 8.4 GiB of bf16 on rank 0. The slot count is tied to the scheduler's active-request cap.

Startup builds the pool only if the configured context fits the kernel's 4096-position limit and current usage plus the pool stays within 90% of rank 0's memory, the same default as Qwen3's `gpu_memory_utilization`; otherwise it logs a warning and leaves NCCL decode on host attention. Rank 0 weights take about 15.9 GiB, so by static estimate 24 GiB cards take the fallback and 32 GiB cards fit. The 90% margin is a convention, not a measured peak, and only 40 GiB cards have run the device path.

`DeviceKvSlot` is an RAII handle stored in the request's `DecodeCache`. Dropping the cache returns the slot to the free list, so finish, error, and cancellation share one release path. A reused slot is not cleared: the kernel only reads positions `0..=position`, and the new owner overwrites them from prefill on.

`decode_attention_kernel` launches one 256-thread block per (head, row). A block writes only its own head's K/V in its own row's slot, so blocks need no cross-block synchronization. Slot IDs come from owned handles, and the pool checks positions before launch. The kernel also guards invalid slot or position metadata by writing NaN instead of touching the pool.

## Numerical Contract

Device attention is bitwise-equal to the host attention oracle at every position up to the 4096 limit, because a last-bit difference can flip a greedy near-tie. The kernel therefore computes nothing whose result could differ from the host:

- RoPE cos/sin come from a table the host builds with its own code. Device math differed in the last bit, and large positions amplified it.
- The softmax scale comes from the host, and the softmax `exp` is a port of glibc's `expf`, which the host uses.
- Products and sums are unfused and in host order. The first version used a tree-reduced RMSNorm and flipped the `Hello` case (`19 year old` vs `20 year old`).

This contract covers attention only. The NCCL backend as a whole is not bitwise-equal to host-staged, because its device-routed MoE uses batched cuBLAS GEMMs ([device-routed-moe.md](device-routed-moe.md)). Long prompts can therefore diverge at greedy near-ties on main as well.

## Correctness Validation

Validation ran on two A100 40 GB GPUs with release builds. These runs build the pool, so they do not exercise the memory fallback.

| Check | Result |
| --- | --- |
| Library tests, including device attention vs host oracle at prefill lengths 1-64 and 1000-4080 | Bitwise-equal |
| Device RMSNorm vs host, hidden and compressed KV | Bitwise-equal |
| Slot reuse, exclusivity, and full-slot refusal | Passed |
| `e2e_ep2` case set and mixed serving: host-staged, NCCL, NCCL with the switch | Token-exact across all three |
| Long context (3990 / 4000-token prompts, 64 output tokens): NCCL vs NCCL with the switch | Token-exact |

On those long prompts, both NCCL modes diverge from host-staged at the same token, which is the MoE difference above. On NCCL, `e2e_ep2` requires every decode step on the device (`GenerationStats::device_decode_steps`) unless the switch is set, so a silent fallback fails the test.

## Performance Validation

HTTP A/B compared main (`353cebf8`) with this branch on two idle A100 40 GB GPUs, NCCL, CUDA Graph off. It used the #466 short contract: about 85 prompt tokens, 64 output tokens, 32 requests, three repeats ([benchmarking.md](benchmarking.md#retained-http-slo-profiles)). Cells are medians.

| Concurrency | Main TPOT p50 | Branch TPOT p50 | TPOT change | Main output tok/s | Branch output tok/s | Main TTFT p50 | Branch TTFT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.8 ms | 15.3 ms | -72.1% | 13.1 | 28.1 | 1413 ms | 1319 ms |
| 4 | 228.1 ms | 58.9 ms | -74.2% | 14.8 | 36.3 | 2868 ms | 3319 ms |
| 8 | 440.6 ms | 116.5 ms | -73.6% | 15.5 | 38.2 | 5674 ms | 6032 ms |

All requests completed with identical output hashes, and the #453 reliability runner passed on both binaries. The trace splits the TTFT rise at concurrency 4/8:

- Prefill is unchanged.
- The rise is all queue wait: c4 goes from 1461 to 1982 ms and c8 from 4254 to 4669 ms.
- Rows that decode together finish together, so their successors queue for serial host prefill. That belongs with #452.

The mixed and long profiles were not compared because host prefill dominates them.

Direct decode (`Hello` prompt, 16 output tokens) on one binary was also measured, with the switch alternated three times. Output hashes were identical. Every row shares one position here, so this excludes the scheduler gain.

| Decode batch | Host attention | Device attention | Change |
| ---: | ---: | ---: | ---: |
| 1 | 29.2 ms | 24.1 ms | -17.7% |
| 4 | 53.0 ms | 35.3 ms | -33.5% |
| 8 | 85.9 ms | 49.1 ms | -42.9% |

Over 512 short requests of turnover, rank 0 VRAM and server RSS stayed flat after the first round. The batch-1 graph probe verifies all replays with the new kernel. The serving decode step still runs eager, because it allocates per-layer scratch.

## Follow-Up Work

1. Store the compressed 576-wide latent instead of expanded K/V. That cuts pool memory about ninefold and lets 24 GiB cards use the device path, but needs an absorbed MLA kernel.
2. Prefill still runs on the host and uploads its cache. Device prefill attention needs a causal kernel and belongs with #452.
3. Port the serving decode step to CUDA Graph: move its per-layer scratch into preallocated buffers; the slot metadata and RoPE table already have stable pointers.
