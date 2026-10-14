# meta-async-set-fix: `-sm tensor` load assert in the meta sync setter, reached from `set_tensor_async`

One format-patch on top of `v12-097cf33d9-r59` (tree `59b9a7fa`; applied tree `83e46879`), for `git am`.  It touches
`ggml/src/ggml-backend-meta.cpp` only (+17 / -5).

## What we saw

From r54 on, loading Qwen3.8-Flash-Next IQ3_XXS with `-sm tensor -lm none` on two GPUs aborts during the model load:

```
ggml/src/ggml-backend-meta.cpp:1781: GGML_ASSERT(size % row_stride == 0) failed
```

It needs `-lm none`: only the loader's pinned async upload goes through `set_tensor_async`, so a load with the default
`-lm` does not reach this code.

A temporary log line just before the assert showed the call:

```
set_tensor axis0: blk.12.ssm_out.weight type q6_K ne 6144 x 2560 nb1 5040 offset 0 size 1048576 total 12902400
```

That is the loader's first 1 MiB chunk of a segmented, axis-0 split tensor.  r54 added a shortcut in
`ggml_backend_meta_set_tensor_async`: a range that starts on a row boundary goes straight to the sync setter
(`ggml_backend_meta_buffer_set_tensor`).  It serves the scheduler's per-op upload of a pruned expert group: whole rows
plus the `min(expert_size, 512)` MMQ tail pad from `copy_experts`.  The loader's chunks also start on a row boundary at
offset 0, but end mid-row: 1048576 mod 5040 = 256, so this one ends 256 bytes into row 208.  So it reaches the setter,
whose axis-0 branch requires whole rows.  It is the same with `--n-cpu-moe 20`: the tensor is a dense one.

One more thing we noticed while reading the same code; we did not observe it separately, because the assert fires
first on this model.  On axis 1 the same chunk would not assert.  The setter would take its partial last row as the
MMQ tail pad, the next chunk would start mid-row and begin the accumulator, and the flush from offset 0 would then
overwrite the first chunk's rows with zeros (the accumulator is zero-filled by `resize`).

## The patch

- Only whole rows go direct, plus, on axis 1, exactly the `min(row_stride, 512)` pad that `copy_experts` appends.
  Any other range is accumulated as before.
- Nothing goes direct while that tensor is accumulating.
- The accumulator records the offset of its first chunk and flushes from there.  Everything before it went direct as
  whole rows, so that offset is on a row boundary; an assert checks this.  Another assert checks that the chunks it
  receives are contiguous, so an unexpected caller aborts instead of being held in the accumulator.

The scheduler's uploads still take the r54 shortcut unchanged: they are whole experts plus the pad, or whole experts
only for the last one.

One case the check can't tell apart: an axis-1 loader chunk whose leftover happens to be exactly the pad size, i.e.
`1048576 mod row_stride == min(row_stride, 512)`, still looks like the pad.  The next chunk would then start mid-row,
and the flush assert would fail instead of writing wrong data.  We didn't hit it.

## Tests (2 x R9700 32 GB, gfx1201)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS + MTP draft (shared Q8_0, `--spec-draft-n-max 3`), no tuning env:

```
llama-server -m <IQ3_XXS> -c 262144 -ngl 99 -sm tensor -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -ot per_layer_token_embd.*=CPU -lm none -lzm off \
  --spec-type draft-mtp -md <mtp Q8_0> -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0 [extra]
```

Each cell is the combined hash of greedy outputs for 12 prompts (4-24 tokens), then the hash of one longer greedy
completion.  r53 predates the r54 shortcut and serves as the reference.

| extra | r59 | r59 + patch | r53 |
|---|---|---|---|
| (all experts in VRAM) | load abort (above) | `527dd2d14a60` / `a7a81f01b3bc` | `527dd2d14a60` / `a7a81f01b3bc` |
| `--n-cpu-moe 20` | load abort (same tensor) | `a29beeb27f18` / `43aaeff3595c` | `a29beeb27f18` / `43aaeff3595c` |
| `--n-cpu-moe 20`, `MOE_EXPERT_CACHE_MIB=0` | not run (same load) | `44ee620b6cb9` / `19c9047a6788` | `44ee620b6cb9` / `19c9047a6788` |

The last row turns off the expert cache, so routed experts are uploaded per op through `copy_experts`.  That is the
path the r54 shortcut serves.  No faults or aborts in any run that loaded.

No urgency on our side.  Happy to re-run anything on your config, or rework it if you'd rather fix it in the loader.
