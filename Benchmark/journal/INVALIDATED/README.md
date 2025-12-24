# INVALIDATED RESULTS

**These journal entries contain INVALID data and analysis due to a critical methodology error discovered on December 22, 2025.**

## The Bug

The analysis code was filtering by `rwbs LIKE '%R%'` without filtering by blktrace action type.

**Problem**: blktrace records multiple actions for each I/O operation:
- `Q` = Queued (app requests)
- `G` = Get request structure
- `P` = Plugged (I/O scheduler)
- `D` = Dispatched (sent to disk)
- `C` = Completed

**Result**: Each I/O operation was counted **5 TIMES** (once for each action).

## Impact on Results

### What Was Wrong:

1. **"3x I/O Amplification"** → FALSE
   - Reported: 38.80 GB total / 12.85 GB unique = 3.02x
   - Reality: 12.85 GB total / 12.85 GB unique = 1.00x (no re-reading!)
   - The 3x was entirely from counting each I/O 5 times

2. **"67% Backward Seeks"** → FALSE
   - Reported: ~67% backward seeks "intrinsic to transformers"
   - Reality: 0% backward seeks when model fits in RAM
   - The backward seeks were from mixing different action types in gap analysis

3. **"Fragmented vs Contiguous: 75.9% vs 100% Coverage"** → INVALID
   - This difference was likely due to hardcoded sector ranges + action overcounting
   - Need to re-run with fixed methodology

4. **Bandwidth: 6-7 GB/s peak** → UNREALISTIC
   - Was measuring queue submission, not actual throughput
   - Exceeded NVMe physical limits

### What Might Still Be Valid:

- ✅ 100% coverage finding (consistent across experiments)
- ✅ Thrashing behavior exists (need to verify threshold)
- ✅ Memory overhead observations (need to verify with pmap)

## The Fix

**Corrected filter** (implemented 2025-12-22):
```python
WHERE action = 'D' AND rwbs LIKE '%R%'
```

Only count `D` (Dispatch) actions = what ACTUALLY hit the disk.

## Corrected Results

See **21DEC.md** for valid analysis using the corrected methodology.

---

**Files in this folder:**
- `21DECP2.md` - December 21 Part 2 (tmpfs removal experiments)
- `21DECP3.md` - December 21 Part 3 (1-extent contiguous file experiments)
- `21DECP4.md` - December 21 Part 4 (gpt-oss-20b MoE experiments)

**Do NOT cite these results in thesis or papers!**
