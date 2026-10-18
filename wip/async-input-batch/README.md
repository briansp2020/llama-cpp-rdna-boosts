# wip/async-input-batch

Two small changes to the async graph-input path (`GGML_CUDA_ASYNC_INPUT=1`). Both remove marker and copy packets the GPU processes between graphs during decode:
- **0001:** copy a graph's queued inputs with one kernel, instead of one `cudaMemcpyAsync` plus one event record per input.
- **0002:** record the "previous graph done" event at the next upload, instead of after every graph.

Two format-patches on `v12-097cf33d9-r82` (tree `dc48b4a9`); applied tree `f75f0c2d`. Patch sha256 `c0eba29b…` (0001), `06b201c8…` (0002).

Nothing changes unless `GGML_CUDA_ASYNC_INPUT=1` is set together with `GGML_CUDA_ASYNC_INPUT_BATCH=1` (or `GGML_CUDA_ASYNC_INPUT_LAZY_DONE=1`). 0002 is meant to be used with 0001: alone it measured up to about 1 % slower on one GPU (35B-A3B; 27B −0.2 %, within noise; table below), so it defaults on only together with the batching.

## What changes

All in `ggml-cuda.cu`, inside the existing async-input code.

**0001, `GGML_CUDA_ASYNC_INPUT_BATCH=1`:**
- **Queue**: `ggml_cuda_async_input_set` still copies the input into the pinned ring. It then records `(dst, ring offset, size)` in a per-device queue instead of issuing the copy.
- **Flush**: `ggml_cuda_async_input_wait` runs at the start of each `graph_compute`. It is first ordered after any older per-copy uploads (`d.ev`) and, like each per-copy upload, waits for the last graph's `done_ev`. It then launches one `ggml_cuda_ain_batch_kernel` on the compute stream:
  - up to 64 uploads per launch, split into 64 KiB chunks;
  - reads the ring through `cudaHostGetDevicePointer`;
  - byte-exact copy, 16-byte vectors when aligned.

  A `flush_ev` is recorded after it.
- **Drain**: every other CUDA buffer or backend access first issues the queued uploads through the old per-copy path, in order. That covers get / set / memset / clear / cpy, their 2-D and async forms, `synchronize` and `free_buffer`. The drain also makes `cudaStreamPerThread` wait for `flush_ev`, so a synchronous access stays ordered after the upload, as before. An upload that overlaps a queued one, or a full queue, drains too.
- **Ring wrap**: before reusing the ring, it also waits for `flush_ev`.
- If `cudaHostGetDevicePointer` fails, uploads fall back to the per-copy path.

**0002, `GGML_CUDA_ASYNC_INPUT_LAZY_DONE`:**
- `done_ev` is recorded on the compute stream after every `graph_compute`, but only an upload reads it, while a decode step runs hundreds of graphs per device.
- The graph now only marks it pending, with its stream. The next consumer (a per-copy upload or the batch kernel) records it on that stream first: later on the same stream, so it still covers the last graph.
- A context destructor records a still-pending event before its streams are destroyed.
- Default on with `GGML_CUDA_ASYNC_INPUT_BATCH=1`, off otherwise; `GGML_CUDA_ASYNC_INPUT_LAZY_DONE=0` / `1` overrides.

## Gates

Setup:
- R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r82 against r82 + these patches, same build dir.
- Models:
  - Qwen3.8-27B Q4_K_M with its own MTP head (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`);
  - Qwen3.6-35B-A3B Q4_K_M, plain decode.
- Server settings: `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`.

**Greedy hashes** (400 tokens, prose / code): identical to stock r82 in every arm:
- stock defaults;
- stock + `ASYNC_INPUT=1`;
- each of 0001 / 0002 alone;
- both.

| model | config | prose / code |
|---|---|---|
| 27B | 1 GPU; 2 GPU `-sm layer` | `d12cdeede326` / `e8a93c816237` |
| 27B | 2 GPU `-sm tensor` | `7cfb359d5c5d` / `dcd93c8965ce` |
| 35B-A3B | 1 GPU; 2 GPU `-sm layer` | `0eedd580608a` / `365e3058e68f` |
| 35B-A3B | 2 GPU `-sm tensor` | `c0cf0739bdab` / `930bcf1f4405` |

**`test-backend-ops`** with `ASYNC_INPUT=1` + `ASYNC_INPUT_BATCH=1`: `-o ADD` 154/154, `-o GET_ROWS` 220/220, `-o CPY` 251/251, `-o MUL_MAT` 2444/2444.

**Faults**: 0 memory-access faults or asserts in any run.

## Launch counts

`rocprofv3 --kernel-trace`, 27B MTP, 1 GPU, the same greedy work, with 0001 against stock + `ASYNC_INPUT=1`:

| kernel | stock r82 + `ASYNC_INPUT=1` | + 0001 |
|---|---|---|
| `__amd_rocclr_copyBuffer` | 18,610 | 4,616 |
| `ggml_cuda_ain_batch_kernel` | — | 2,691 × 9.5 µs |

The remaining `copyBuffer` launches are copies that do not go through the async-input ring.

## Throughput (decode)

All arms are one binary with `GGML_CUDA_ASYNC_INPUT=1`; the other two switches select the arm. Each arm is 8 greedy runs (prose and code, twice, in two rounds with the arm order reversed). Within an arm the runs spread by about 0.3 %. Change is against the arm with neither patch active.

| config | neither | 0002 only | 0001 only | both |
|---|---|---|---|---|
| 35B-A3B, 2 GPU `-sm tensor` | 136.24 t/s | +3.87 % | +0.23 % | **+4.26 %** |
| 27B MTP, 2 GPU `-sm tensor` | 103.24 t/s | +1.17 % | +0.34 % | **+1.77 %** |
| 35B-A3B, 1 GPU | 127.08 t/s | −1.04 % | +1.01 % | **+0.89 %** |
| 27B MTP, 1 GPU | 68.46 t/s | −0.20 % | +0.35 % | **+0.35 %** |

- **0002 alone on one GPU:** it costs about 1 % on 35B-A3B (27B −0.2 %, within noise). We have not traced why. 0001 removes the cost, which is why 0002 defaults on only with 0001.
- **Both together:** never measurably worse than either alone (35B-A3B 1 GPU: +0.89 % against +1.01 %, inside the spread), and within about 0.2–0.9 points of the two gains added up.
- **Which build:** the hashes and this table are from a build where 0002 defaulted on regardless of 0001. Each arm sets the switches explicitly, so the code paths are the ones in the patches. The final series changes only that default and adds the destructor flush, which runs only when a context is freed. On the final series:
  - `test-backend-ops` passes;
  - the build before the destructor flush matched the "both" column (1 GPU and tensor, both models), and the "neither" column with only `ASYNC_INPUT=1` set (35B-A3B, 1 GPU), in hashes and speed.
- **Prefill:** flat for 0001 alone (±0.2 %, two rounds); not re-measured for the 2 × 2.

On our hybrid Flash-Next build (2 GPU `-sm tensor`, MTP draft; a different base, r80, and an earlier version of these changes), which carries both, 0001 on top of 0002 gave decode +4.1 % (95 % CI +3.4 to +4.8 %) in a paired 6-round A/B, with identical output.
