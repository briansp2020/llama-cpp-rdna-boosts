#!/usr/bin/env python3
"""QSA attended-set census (wip/kv-residency B1(c), TODO #118).

Reads the binary the diagnostic in `ggml/src/ggml-cuda/indexer-topk.cu` writes when
`LLAMA_QSA_STREAM_CENSUS=<path>` is set (campaign branch only), and answers the one question a residency
POOL's design hangs on: **how stable is the attended set from step to step**, i.e. what hit rate would a pool
of N cells get if it admitted what attention selected?

Record layout (little-endian int32, repeated): [nrows, k, n_kv, n_blocks, name_len, name bytes, ids (k*nrows)].
Only decode-band calls (`nrows <= 8`) are recorded, so every record is one layer of one decode step.

Usage:
  ./qsa-census.py /tmp/census.bin [--cell-bytes 1088] [--layers 12]

The arm that produces it must be run with `GGML_CUDA_DISABLE_GRAPHS=1` and the sparse decode arm
(`LLAMA_QSA_DENSE_DECODE_UNTIL=0`), otherwise no top-k tensor is ever computed.  Timings from that arm are
meaningless (an uncaptured graph); only the selection statistics are.
"""
import argparse
import struct
import sys
from collections import OrderedDict, defaultdict


def read_records(path):
    recs = []
    with open(path, "rb") as f:
        blob = f.read()
    off = 0
    while off + 5 * 4 <= len(blob):
        nrows, k, n_kv, n_blocks, name_len = struct.unpack_from("<5i", blob, off)
        off += 5 * 4
        if name_len < 0 or name_len > 4096 or off + name_len > len(blob):
            break
        name = blob[off:off + name_len].decode("utf-8", "replace")
        off += name_len
        n = k * nrows
        if off + 4 * n > len(blob):
            break
        ids = struct.unpack_from("<%di" % n, blob, off)
        off += 4 * n
        recs.append((nrows, k, n_kv, n_blocks, name, ids))
    return recs


def lru_hits(seq_per_step, capacity):
    """Simulate an LRU pool of `capacity` cells over a sequence of per-step cell sets.

    Returns (hits, misses).  A step is per-layer: one query column (nrows == 1) or a verify batch, whose
    rows are unioned (the attention reads all of them in that step).
    """
    pool = OrderedDict()
    hits = misses = 0
    for step in seq_per_step:
        for cell in step:
            if cell in pool:
                pool.move_to_end(cell)
                hits += 1
            else:
                misses += 1
                pool[cell] = None
                if len(pool) > capacity:
                    pool.popitem(last=False)
    return hits, misses


def flru_hits(seq_per_step, capacity, period=32):
    """F-LRU (the delivery's expert-cache policy shape: a decayed frequency plus an LRU tiebreak).

    `period` steps halve every counter, so the policy forgets a topic shift instead of carrying a stale
    hot set forever (the reason the arena chose F-LRU over pure LRU; AGENTS.md block 09/12).
    """
    freq = defaultdict(int)
    last = {}
    hits = misses = 0
    tick = 0
    for step in seq_per_step:
        tick += 1
        if tick % period == 0:
            for k in list(freq):
                freq[k] //= 2
                if freq[k] == 0:
                    del freq[k]
        for cell in step:
            if cell in freq:
                freq[cell] += 1
                last[cell] = tick
                hits += 1
            else:
                misses += 1
                freq[cell] = 1
                last[cell] = tick
                if len(freq) > capacity:
                    victim = min(freq, key=lambda k: (freq[k], last[k]))
                    del freq[victim]
                    del last[victim]
    return hits, misses


def static_hits(seq_per_step, capacity, warmup):
    """A PRE-CALCULATED ordering (Strata's static arena): after `warmup` steps freeze the `capacity` most
    attended cells and admit nothing afterwards.  This is the policy the campaign rejected for the expert
    cache (content beats a frozen ordering) -- measured here on the KV's attended set for the same reason."""
    counts = defaultdict(int)
    hits = misses = 0
    frozen = None
    for i, step in enumerate(seq_per_step):
        if frozen is None and i >= warmup:
            frozen = set(sorted(counts, key=lambda k: -counts[k])[:capacity])
        for cell in step:
            if frozen is None:
                counts[cell] += 1
                continue          # warmup: nothing resident yet, but also nothing to charge? charge it
            if cell in frozen:
                hits += 1
            else:
                misses += 1
    return hits, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cell-bytes", type=float, default=1088.0,
                    help="bytes per cell per layer (Flash-Next q8_0: 2 KV heads x 256 dims x K+V x ~1.06 B)")
    ap.add_argument("--layers", type=int, default=0,
                    help="expected layers per step (0 = infer from the record order)")
    args = ap.parse_args()

    recs = read_records(args.path)
    if not recs:
        sys.exit("no records -- was the arm run with GGML_CUDA_DISABLE_GRAPHS=1 and the sparse arm?")

    # order of layers = the distinct names, in first-seen order
    names = []
    for r in recs:
        if r[4] not in names:
            names.append(r[4])
    n_layers = args.layers or len(names)
    print("records: %d, layers per step: %d (%s), k per record: %s, n_kv: %s"
          % (len(recs), n_layers, ", ".join(names[:4]) + ("..." if len(names) > 4 else ""),
             sorted({r[1] for r in recs}), sorted({r[2] for r in recs})))

    # group into steps: a step is one record per layer, in the layer order
    steps = defaultdict(list)   # layer index -> list of sets
    cur = {}
    step_no = 0
    for nrows, k, n_kv, n_blocks, name, ids in recs:
        li = names.index(name)
        if li in cur:                       # the layer repeated -> a new step started without all layers
            for _, cells in sorted(cur.items()):
                steps[_].append(cells)
            cur = {}
            step_no += 1
        cur[li] = set(ids[i] for i in range(0, len(ids), max(nrows, 1)))
    if cur:
        for li, cells in sorted(cur.items()):
            steps[li].append(cells)

    n_steps = max(len(v) for v in steps.values())
    print("decode steps analysed: %d" % n_steps)

    # attended set size and step-to-step stability, per layer
    sizes = [len(s) for v in steps.values() for s in v]
    sizes.sort()
    print("attended cells per layer per step: min %d  median %d  max %d"
          % (sizes[0], sizes[len(sizes) // 2], sizes[-1]))

    jac = []
    for li, seq in steps.items():
        for a, b in zip(seq, seq[1:]):
            u = len(a | b)
            if u:
                jac.append(len(a & b) / u)
    if jac:
        jac.sort()
        print("consecutive-step Jaccard (same layer): median %.3f  p10 %.3f  p90 %.3f"
              % (jac[len(jac) // 2], jac[int(0.10 * len(jac))], jac[int(0.90 * (len(jac) - 1))]))

    # the deciding number: an LRU pool's hit rate, and the host traffic it implies
    n_att = sum(sizes) / max(len(sizes), 1)     # mean attended cells per layer per step
    print("\npool (cells/layer)   hit rate   host read MiB/token   vs an all-miss pool")
    base = None
    for cap in (2048, 4096, 8192, 16384, 32768, 65536):
        hits = misses = 0
        for li, seq in steps.items():
            h, m = lru_hits(seq, cap)
            hits += h
            misses += m
        tot = hits + misses
        rate = hits / tot if tot else 0.0
        mib = misses * args.cell_bytes / (1024.0 * 1024.0) / max(n_steps, 1)
        if base is None:
            base = mib
        print("%20d   %7.3f%%   %18.2f   %+8.1f%%" % (cap, 100.0 * rate, mib,
                                                      100.0 * (mib / base - 1.0) if base else 0.0))

    # policy comparison at equal capacity: LRU (Strata's CLOCK approximates it), the delivery's F-LRU, and a
    # pre-calculated static ordering.  The expert-cache campaign chose adaptive over static for the arena; this
    # is the same question on the KV's attended set.
    print("\npolicy comparison (hit rate, %d steps):" % n_steps)
    print("%8s %10s %10s %13s" % ("cells", "LRU", "F-LRU", "static(warm 1/2)"))
    for cap in (2048, 4096, 8192, 16384):
        tot = hit = hit_f = miss_f = 0
        for li, seq in steps.items():
            h, m = lru_hits(seq, cap)
            hit += h
            tot += h + m
            hf, mf = flru_hits(seq, cap)
            hit_f += hf
            miss_f += mf
        # static: count over the whole sequence with the second half frozen on the first half's heat
        hs = ms = 0
        for li, seq in steps.items():
            combined = list(seq)
            half = max(len(combined) // 2, 1)
            counts = defaultdict(int)
            for s2 in combined[:half]:
                for c in s2:
                    counts[c] += 1
            frozen = set(sorted(counts, key=lambda k: -counts[k])[:cap])
            for s2 in combined[half:]:
                for c in s2:
                    if c in frozen:
                        hs += 1
                    else:
                        ms += 1
        print("%8d %9.2f%% %9.2f%% %12.2f%%" % (cap,
              100.0 * hit / tot if tot else 0.0,
              100.0 * hit_f / (hit_f + miss_f) if (hit_f + miss_f) else 0.0,
              100.0 * hs / (hs + ms) if (hs + ms) else 0.0))

    print("\nall-miss (zero-hit) read for the same selection: %.2f MiB/token"
          % (sum(len(s) for v in steps.values() for s in v) * args.cell_bytes
             / (1024.0 * 1024.0) / max(n_steps, 1)))
    print("note: read volume is per layer only; multiply by the QSA layer count for the whole step, and")
    print("      compare with the measured cost (bytes / 55.7 GB/s) before believing a design.")


if __name__ == "__main__":
    main()
