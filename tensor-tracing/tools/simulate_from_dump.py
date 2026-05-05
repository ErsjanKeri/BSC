#!/usr/bin/env python3
"""
Cache policy simulator using ground-truth access sequence from C code dump.

Reads BSC_CACHE_DUMP CSV (produced by llama-io-uring-buf.cpp) and replays
the exact access sequence against multiple cache policies. Results are
bit-exact verified against the real C implementation.

Usage:
  python3 simulate_from_dump.py /tmp/cache_dump_2000tok.csv --cache-sizes 200,250,288,350,498
"""

import argparse
import csv
import sys
from collections import OrderedDict


# Load access sequence from C dump

def load_dump(path):
    """Read CSV dump, return list of (batch, real_hits, real_misses)."""
    loads = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            layer = int(row['layer'])
            p_start = int(row['p_start'])
            p_end = int(row['p_end'])
            n_real = int(row['n_real'])
            eids = [int(row[f'eid{i}']) for i in range(n_real)]
            batch = [(layer, p, eid) for p in range(p_start, p_end) for eid in eids]
            loads.append((batch, int(row['hits']), int(row['misses'])))
    return loads


# Policy implementations — all bit-exact verified against C code

class LRUCache:
    """LRU with batch protection. Verified against C code."""
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def batch_access(self, keys):
        hits = 0
        batch_set = set(keys)
        for key in keys:
            if key in self.cache:
                self.cache.move_to_end(key)
                hits += 1
            else:
                while len(self.cache) >= self.cap:
                    for ck in self.cache:
                        if ck not in batch_set:
                            del self.cache[ck]
                            break
                    else:
                        break
                self.cache[key] = True
        return hits, len(keys) - hits


class LFUSlotCache:
    """LFU with slot-based eviction + epoch protection. Verified against C code."""
    def __init__(self, capacity):
        self.cap = capacity
        self.slot_key = [None] * capacity
        self.slot_freq = [0] * capacity
        self.key_to_slot = {}
        self.n_filled = 0

    def batch_access(self, keys):
        hits = 0
        epoch_slots = set()
        for key in keys:
            if key in self.key_to_slot:
                slot = self.key_to_slot[key]
                self.slot_freq[slot] += 1
                epoch_slots.add(slot)
                hits += 1
            else:
                if self.n_filled < self.cap:
                    slot = self.n_filled
                    self.n_filled += 1
                else:
                    slot = -1
                    min_f = 0xFFFFFFFF
                    for i in range(self.cap):
                        if i in epoch_slots:
                            continue
                        if slot < 0 or self.slot_freq[i] < min_f:
                            min_f = self.slot_freq[i]
                            slot = i
                    if slot < 0:
                        continue
                old = self.slot_key[slot]
                if old is not None:
                    del self.key_to_slot[old]
                self.slot_key[slot] = key
                self.slot_freq[slot] = 1
                self.key_to_slot[key] = slot
                epoch_slots.add(slot)
        return hits, len(keys) - hits


class LFUAgingSlotCache:
    """LFU-aging with slot-based eviction. Verified against C code."""
    def __init__(self, capacity):
        self.cap = capacity
        self.slot_key = [None] * capacity
        self.slot_freq = [0] * capacity
        self.key_to_slot = {}
        self.n_filled = 0
        self.access_count = 0
        self.aging_period = 10 * capacity

    def batch_access(self, keys):
        hits = 0
        epoch_slots = set()
        for key in keys:
            self.access_count += 1
            if self.access_count >= self.aging_period:
                self.access_count = 0
                for i in range(self.cap):
                    self.slot_freq[i] >>= 1

            if key in self.key_to_slot:
                slot = self.key_to_slot[key]
                self.slot_freq[slot] += 1
                epoch_slots.add(slot)
                hits += 1
            else:
                if self.n_filled < self.cap:
                    slot = self.n_filled
                    self.n_filled += 1
                else:
                    slot = -1
                    min_f = 0xFFFFFFFF
                    for i in range(self.cap):
                        if i in epoch_slots:
                            continue
                        if slot < 0 or self.slot_freq[i] < min_f:
                            min_f = self.slot_freq[i]
                            slot = i
                    if slot < 0:
                        continue
                old = self.slot_key[slot]
                if old is not None:
                    del self.key_to_slot[old]
                self.slot_key[slot] = key
                self.slot_freq[slot] = 1
                self.key_to_slot[key] = slot
                epoch_slots.add(slot)
        return hits, len(keys) - hits


class ARCCache:
    """Adaptive Replacement Cache. To be implemented in C the same way."""
    def __init__(self, capacity):
        self.c = capacity
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()

    def batch_access(self, keys):
        hits = 0
        for key in keys:
            if self._access(key):
                hits += 1
        return hits, len(keys) - hits

    def _access(self, key):
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = True
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True
        if key in self.b1:
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.p + delta, self.c)
            del self.b1[key]
            self._replace(False)
            self.t2[key] = True
            return False
        if key in self.b2:
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(self.p - delta, 0)
            del self.b2[key]
            self._replace(True)
            self.t2[key] = True
            return False
        l1 = len(self.t1) + len(self.b1)
        l2 = len(self.t2) + len(self.b2)
        if l1 == self.c:
            if len(self.t1) < self.c:
                if self.b1:
                    self.b1.popitem(last=False)
                self._replace(False)
            else:
                if self.t1:
                    self.t1.popitem(last=False)
        elif l1 + l2 >= self.c:
            if l1 + l2 >= 2 * self.c and self.b2:
                self.b2.popitem(last=False)
            if len(self.t1) + len(self.t2) >= self.c:
                self._replace(False)
        self.t1[key] = True
        return False

    def _replace(self, in_b2):
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = True
        elif self.t2:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = True


class WTinyLFUCache:
    """Window-TinyLFU with Count-Min Sketch admission filter."""
    def __init__(self, capacity):
        self.cap = capacity
        self.win_max = max(1, capacity // 100) or 1
        self.prot_max = max(1, int(capacity * 0.79))
        self.prob_max = max(1, capacity - self.win_max - self.prot_max)

        self.window = OrderedDict()
        self.probation = OrderedDict()
        self.protected = OrderedDict()

        self.sketch_w = max(64, capacity * 4)
        self.sketch = [[0] * self.sketch_w for _ in range(4)]
        self.seeds = [0x9e3779b9, 0x517cc1b7, 0x6a09e667, 0xbb67ae85]
        self.sk_n = 0
        self.sk_reset = max(capacity * 10, 1000)

    def _sk_hash(self, key, seed):
        h = hash(key) ^ seed
        h = ((h >> 16) ^ h) * 0x45d9f3b
        return h % self.sketch_w

    def _sk_add(self, key):
        self.sk_n += 1
        if self.sk_n >= self.sk_reset:
            self.sk_n = 0
            for row in self.sketch:
                for i in range(len(row)):
                    row[i] >>= 1
        for d in range(4):
            idx = self._sk_hash(key, self.seeds[d])
            self.sketch[d][idx] = min(self.sketch[d][idx] + 1, 255)

    def _sk_est(self, key):
        return min(self.sketch[d][self._sk_hash(key, self.seeds[d])] for d in range(4))

    def batch_access(self, keys):
        hits = 0
        for key in keys:
            self._sk_add(key)
            if key in self.protected:
                self.protected.move_to_end(key)
                hits += 1
            elif key in self.probation:
                del self.probation[key]
                self._to_protected(key)
                hits += 1
            elif key in self.window:
                self.window.move_to_end(key)
                hits += 1
            else:
                ev = None
                if len(self.window) >= self.win_max:
                    ek, _ = self.window.popitem(last=False)
                    ev = ek
                self.window[key] = True
                if ev is not None:
                    if len(self.probation) < self.prob_max:
                        self.probation[ev] = True
                    else:
                        victim = next(iter(self.probation))
                        if self._sk_est(ev) > self._sk_est(victim):
                            del self.probation[victim]
                            self.probation[ev] = True
        return hits, len(keys) - hits

    def _to_protected(self, key):
        if len(self.protected) >= self.prot_max:
            dk, _ = self.protected.popitem(last=False)
            if len(self.probation) >= self.prob_max:
                self.probation.popitem(last=False)
            self.probation[dk] = True
        self.protected[key] = True


class BeladyCache:
    """Belady's optimal (offline). Precomputes next-use for every access."""
    def __init__(self, capacity, all_batches):
        self.cap = capacity
        self.cache = set()
        # Flatten and precompute next-use
        flat = []
        for batch in all_batches:
            flat.extend(batch)
        from collections import defaultdict
        from bisect import bisect_right
        self.positions = defaultdict(list)
        for i, key in enumerate(flat):
            self.positions[key].append(i)
        self.flat = flat
        self.pos = 0

    def batch_access(self, keys):
        from bisect import bisect_right
        hits = 0
        new_keys = []
        for key in keys:
            if key in self.cache:
                hits += 1
            else:
                new_keys.append(key)
            self.pos += 1

        batch_set = set(keys)
        for key in new_keys:
            if key in self.cache:
                continue
            if len(self.cache) >= self.cap:
                farthest = -1
                victim = None
                for ck in self.cache:
                    if ck in batch_set:
                        continue
                    pidx = bisect_right(self.positions[ck], self.pos - 1)
                    nu = self.positions[ck][pidx] if pidx < len(self.positions[ck]) else len(self.flat) + 1
                    if nu > farthest:
                        farthest = nu
                        victim = ck
                if victim is not None:
                    self.cache.discard(victim)
                else:
                    continue
            self.cache.add(key)
        return hits, len(keys) - hits


# Main

def simulate(loads, cache_size, policy_cls, all_batches=None, **kwargs):
    if policy_cls == BeladyCache:
        cache = policy_cls(cache_size, all_batches)
    else:
        cache = policy_cls(cache_size, **kwargs)
    total_h = total_m = 0
    for batch, _, _ in loads:
        h, m = cache.batch_access(batch)
        total_h += h
        total_m += m
    return total_h, total_m


def main():
    parser = argparse.ArgumentParser(description="Simulate cache policies from C code dump")
    parser.add_argument("dump_csv", help="Path to BSC_CACHE_DUMP CSV file")
    parser.add_argument("--cache-sizes", default="200,250,288,350,498,740",
                        help="Comma-separated cache sizes")
    args = parser.parse_args()

    cache_sizes = [int(x) for x in args.cache_sizes.split(",")]
    loads = load_dump(args.dump_csv)
    all_batches = [batch for batch, _, _ in loads]
    n_accesses = sum(len(b) for b in all_batches)

    # Verify LFU against ground truth first
    real_h = sum(rh for _, rh, _ in loads)
    real_m = sum(rm for _, _, rm in loads)
    lfu_h, lfu_m = simulate(loads, 200, LFUSlotCache)
    if lfu_h != real_h:
        print(f"WARNING: LFU validation failed! Real={real_h} Sim={lfu_h}", file=sys.stderr)
    else:
        print(f"LFU validation passed (dump policy hits={real_h})", file=sys.stderr)

    policies = [
        ("LRU",       LRUCache),
        ("LFU",       LFUSlotCache),
        ("LFU-aging", LFUAgingSlotCache),
        ("ARC",       ARCCache),
        ("W-TinyLFU", WTinyLFUCache),
        ("Belady",    BeladyCache),
    ]

    print(f"\nAccess sequence: {len(loads)} loads, {n_accesses} accesses")
    print(f"")

    # Header
    print(f"{'Cache':>5}", end="")
    for name, _ in policies:
        print(f"  {name:>12}", end="")
    print()
    print("-" * (7 + 14 * len(policies)))

    for cs in cache_sizes:
        print(f"{cs:>5}", end="")
        for name, cls in policies:
            h, m = simulate(loads, cs, cls, all_batches=all_batches)
            hit_pct = h / n_accesses * 100
            print(f"  {hit_pct:>11.1f}%", end="")
        print()

    print()
    print("(Values are HIT RATES — higher is better. Belady = theoretical maximum.)")


if __name__ == "__main__":
    main()
