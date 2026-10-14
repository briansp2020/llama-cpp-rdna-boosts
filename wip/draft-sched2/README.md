# draft-sched2: a second scheduler for the MTP draft context's graphs with outputs

One format-patch on top of `v12-097cf33d9-r61` (tree `047807d7`; applied tree `5681694a`), for `git am`.  10 files,
+146 / -12.  `LLAMA_DRAFT_SCHED2=0` turns it off.

## What it changes

Every MTP cycle, the draft context runs two different graphs: a no-output ingest graph and a 1-token draft-step graph.
Graph reuse needs the scheduler's split and allocation state to belong to the graph, so with one scheduler each switch
re-splits and re-allocates (about 0.5 ms of host time per cycle, mostly `ggml_backend_sched_alloc_graph`).

- `llama_context` gets a second scheduler, created on first use, for the graphs with outputs.  It is swapped into
  `sched` (with its own `gf_res_prev_active`) only while a `decode`/`encode` runs, and swapped back by a scope guard,
  so the rest of the context always sees the main scheduler.  It only engages for a context with `ctx_other` set, i.e.
  a draft that borrows the target's tensors: the MTP draft here, but also EAGLE3, DFlash and the Gemma 4 assistant,
  which we have not tested.  `sched_reserve`, `memory_update`, `graph_reserve` and the optimizer path drop the second scheduler's
  reuse state along with the main one's.
- Meta backend: a buffer keeps its external views in a rotating pair of containers that assumes one live graph, so a
  rebuild by one scheduler cleared the other graph's views (`GGML_ASSERT(bcj.nodes[i])` in `graph_compute`).  Each
  scheduler now selects its own view slot (`ggml_backend_meta_set_view_slot`); slot 1 gets its own pair, created on
  first use.
- Slab: the second scheduler's small compute buffer is allocated outside the slab and without the chunk rounding
  (`ggml_backend_dev_slab_compute_plain`, a new optional device-interface entry appended after the existing ones, so
  backends that don't set it are zero-initialized), because narrow-2 holds the draft's one slab view.

## Tests (2 x R9700 32 GB, gfx1201)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM, + MTP draft (shared Q8_0, `--spec-draft-n-max 3`), no tuning
env.  The binary is r61 + this patch + #153 (needed for `-lm none` to load); one binary, `LLAMA_DRAFT_SCHED2=0` for
"off".  Each run is a fresh server after `llama-warmup.py`, in the order off, on, off, on.

```
llama-server -m <IQ3_XXS> -c 262144 -ngl 99 -sm tensor -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -ot per_layer_token_embd.*=CPU -lm none -lzm off \
  --spec-type draft-mtp -md <mtp Q8_0> -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0
```

| t/s | off | on | change |
|---|---|---|---|
| greedy 400 tokens, 1st request | 130.3 / 130.1 | 132.5 / 136.0 | +3.3 % |
| greedy 400 tokens, 2nd request | 134.8 / 134.5 | 137.0 / 137.1 | +1.8 % |
| seeded chat (temperature 0.7) | 105.5 / 105.1 | 107.7 / 107.2 | +2.2 % |
| 4 concurrent requests, aggregate | 146.3 / 142.8 | 146.9 / 151.1 | within noise |

Outputs are the same with it on and off: greedy `a7a81f01b3bc` (text checked), 12 short prompts `527dd2d14a60`, seeded
`21fb152ad281`, draft acceptance 274/374 in every run.  No faults or aborts.

No urgency on our side.  Happy to re-run anything on your config.
