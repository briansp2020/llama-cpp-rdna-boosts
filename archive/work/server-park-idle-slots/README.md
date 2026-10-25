# wip/server-park-idle-slots — park the resident idle slots at the top of a unified KV cache

One `git am` file (one commit), made on `v14-3f03b1772-r133` (tree `972701d2…`; applied: tree `aea351ae…`).
It touches:
- `src/llama-kv-cache.{cpp,h}`, `src/llama-kv-cells.h`, `src/llama-memory.cpp`, `src/llama-ext.h`;
- `common/common.{cpp,h}`;
- `tools/server/server-context.cpp`.

These eight files are identical in r131 and r133, and every measurement below was made on r131 with this change.

**Why.** #210 keeps an idle slot's context resident in a unified KV cache. A request that then runs alone no longer
starts in an empty cache:
- its cells land after the resident ones;
- its attention range (`n_kv`) covers them.

Flash attention's KV tiling and split depend on the cell indices and on `n_kv`, so near-ties resolve differently. The
same 16 requests, run one at a time in two rounds on 4 slots, agreed on 4/16 on r131. They agree on 16/16 with
`LLAMA_SERVER_LAZY_IDLE_SLOTS=0` and on r120.

**What changes.**
- **KV cache.** `llama_memory_seq_park(mem, seq_id, park)` (staging API, `llama-ext.h`) moves a sequence's cells to the
  top of a unified cache, or back down. The K/V rows are copied on the device, and the cell state moves with them.
  - New cells are only taken below the lowest parked cell, and `get_n_kv()` stops below it.
  - The parked sequences are kept together at the top.
  - `llama_memory_can_park()` tells whether a cache supports this. It requires one stream, cells that no other cache
    shares, V not transposed, and no sliding window.
- **Server.** When a task starts, the idle slots are parked and the slots that run come back to the low cells. A request
  that runs alone then sits in the same cells, and attends over the same range, as when the idle slots are saved and
  cleared.
  - An idle slot that cannot be parked (its cells are not one run, or there is no room) is saved and cleared, as
    before #210.
  - A cache that does not support parking keeps #210's behaviour unchanged.
  - The draft context is not parked.
- **Switches.** `LLAMA_SERVER_PARK_IDLE_SLOTS=0` keeps #210's behaviour. `LLAMA_KV_PARK_DEBUG=1` logs every move and
  every refusal.

**Results** (gfx1201, Qwen3.8-27B UD-Q4_K_XL + DFlash2):
- **Repeat test.** The same 16 requests in two rounds now agree on 16/16. Both rounds are also token-identical, request
  by request, to `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` (32/32).
- **Switch time.** Two 40K-token conversations taking turns take 0.90 s per switch, against 0.85 s with #210 as it is
  and 2.01 s with the save-and-restore path.

Details, side effects and limits are in the PR description.

*The patch and this note were written by Claude (AI).*
