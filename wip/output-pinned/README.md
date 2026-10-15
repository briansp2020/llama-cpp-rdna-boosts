# output-pinned: the output buffer is pageable under `-sm tensor`

One format-patch on top of `v12-097cf33d9-r62` (tree `386a6a3d`; applied tree `61f5e1d7`), for `git am`.  It touches
`src/llama-context.cpp` only (+9).

## What we saw

Under `-sm tensor`, `llama_context`'s output buffer (logits, embeddings, the MTP hidden state) is allocated with the
CPU buffer type, i.e. pageable memory:

- `output_reserve` asks the output device for its host buffer type.  Under `-sm tensor` that device is the meta device.
- `ggml_backend_meta_device_get_host_buffer_type` returns `nullptr` when the simple devices report different host
  buffer types.
- The CUDA backend gives each device its own pinned host buffer type (`ggml_backend_cuda_host_buffer_type_dev`, which
  differ only in `.device`), so they always differ, and `output_reserve` falls back to `ggml_backend_cpu_buffer_type()`.

So every per-step device-to-host read of an output goes to pageable memory and blocks the host until the GPU is done.
With temporary logging around the MTP hidden-state read (`ggml_backend_tensor_get_async` of `t_h_nextn` in `decode`),
averaged per 512 calls:

| | output buffer | `t_h_nextn` read, target | `t_h_nextn` read, draft |
|---|---|---|---|
| r62 | `CPU` | 966 - 2493 us per call | 19 us per call |
| r62 + patch | `ROCm_Host` | 1 us per call | 1 us per call |

## The patch

When the output device is a meta device with no host buffer type, `output_reserve` uses the first simple device's
host buffer type for the output buffer.  The output buffer is never scheduled on, so its `.device` does not matter,
and the pinned allocation itself is device-agnostic (the comment above `ggml_backend_cuda_host_buffer_type_dev` says
the same).  The meta device's own `get_host_buffer_type` is unchanged, so nothing else that schedules on host buffer
types is affected.  `LLAMA_OUTPUT_PINNED=0` restores the old behaviour, for A/B.

## Tests (2 x R9700 32 GB, gfx1201)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM, + MTP draft (shared Q8_0, `--spec-draft-n-max 3`), no tuning
env.  The binary is r62 + this patch + #153 (needed for `-lm none` to load); one binary, `LLAMA_OUTPUT_PINNED=0` for
"off".  Each run is a fresh server after `llama-warmup.py`, in the order off, on, off, on.

```
llama-server -m <IQ3_XXS> -c 262144 -ngl 99 -sm tensor -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -ot per_layer_token_embd.*=CPU -lm none -lzm off \
  --spec-type draft-mtp -md <mtp Q8_0> -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0
```

| t/s | off | on |
|---|---|---|
| greedy 400 tokens, 1st request | 130.6 / 130.2 | 136.0 / 131.5 |
| greedy 400 tokens, 2nd request | 134.7 / 134.7 | 136.2 / 136.3 |
| seeded chat (temperature 0.7) | 105.9 / 105.0 | 107.2 / 106.6 |
| 4 concurrent requests, aggregate | 160.4 / 143.7 | 148.0 / 146.6 |

About +1 % single-request; the 4-concurrent numbers swing more than that between runs.  Outputs are the same with it
on and off: greedy `a7a81f01b3bc` (text checked), 12 short prompts `527dd2d14a60`, seeded `21fb152ad281`, draft
acceptance 274/374 in every run.  No faults or aborts.

No urgency on our side.  Happy to re-run anything on your config, or rework it if you'd rather fix it in the meta
device's `get_host_buffer_type`.
