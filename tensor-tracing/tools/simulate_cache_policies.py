#!/usr/bin/env python3
"""
Cache policy simulator for MoE expert access patterns.

Replays recorded expert access sequences against multiple cache eviction
policies and cache sizes. Each simulation counts exact hits/misses without
running actual inference — the access sequence is deterministic.

CRITICAL: accesses are batched per-layer (12 keys per batch = 3 proj × 4 experts)
to match the real load() call behavior. Within a batch, eviction cannot touch
items that were just inserted/hit in the same batch (epoch protection).

Usage:
  python3 simulate_cache_policies.py experiments/expert-traces-2k/
"""

import argparse
import json
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path


# =============================================================================
# Access sequence loader — returns batches, not flat list
# =============================================================================

def load_access_batches(trace_dir: str) -> list[list[tuple[int, int, int]]]:
    """Load expert accesses grouped into per-layer batches.

    Returns list of batches. Each batch = one load() call = 12 keys
    [(layer, proj, expert), ...] for one layer of one token.
    """
    trace_dir = Path(trace_dir)
    batches = []

    for f in sorted(trace_dir.glob("token-*.json")):
        with open(f) as fh:
            data = json.load(fh)

        layer_experts = {}
        for entry in data["entries"]:
            if entry["operation_type"] == "MUL_MAT_ID" and entry["num_experts"] > 0:
                lid = entry["layer_id"]
                if lid not in layer_experts:
                    layer_experts[lid] = entry["expert_ids"][:entry["num_experts"]]

        for lid in sorted(layer_experts.keys()):
            batch = []
            for proj in range(3):
                for eid in layer_experts[lid]:
                    batch.append((lid, proj, eid))
            batches.append(batch)

    return batches


# =============================================================================
# Cache policy implementations — all use batch_access(keys) interface
# =============================================================================

class LRUCache:
    """Classic LRU. Batch: all keys get MRU placement, eviction only from
    items NOT in the current batch."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def batch_access(self, keys):
        hits = 0
        batch_set = set(keys)
        for key in keys:
            if key in self.cache:
                self.cache.move_to_end(key)
                hits += 1
            else:
                # Evict LRU that's NOT in current batch
                while len(self.cache) >= self.capacity:
                    # Find LRU item not in batch
                    evicted = False
                    for ck in self.cache:  # iterates oldest first
                        if ck not in batch_set:
                            del self.cache[ck]
                            evicted = True
                            break
                    if not evicted:
                        break  # all cache items are in batch (shouldn't happen if cap >= 12)
                self.cache[key] = True
        return hits, len(keys) - hits


class LFUCache:
    """LFU with slot-based eviction matching the real C implementation.

    Uses fixed slot array (not dict). Victim = lowest slot index with min freq,
    skipping slots touched in current batch (epoch). This matches lfu_pick_victim()
    in llama-io-uring-buf.cpp which scans slot 0..N-1 and picks the first min.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.slot_key = [None] * capacity   # slot index -> key
        self.slot_freq = [0] * capacity     # slot index -> freq
        self.key_to_slot = {}               # key -> slot index
        self.n_filled = 0                   # next free slot during initial fill

    def batch_access(self, keys):
        hits = 0
        epoch_slots = set()  # slot indices touched in this batch
        for key in keys:
            if key in self.key_to_slot:
                # HIT
                slot = self.key_to_slot[key]
                self.slot_freq[slot] += 1
                epoch_slots.add(slot)
                hits += 1
            else:
                # MISS — find victim slot
                if self.n_filled < self.capacity:
                    slot = self.n_filled
                    self.n_filled += 1
                else:
                    # Scan for min freq, skip epoch-protected slots
                    slot = -1
                    min_f = float('inf')
                    for i in range(self.capacity):
                        if i in epoch_slots:
                            continue
                        if self.slot_freq[i] < min_f:
                            min_f = self.slot_freq[i]
                            slot = i
                    if slot < 0:
                        continue  # can't evict (shouldn't happen)

                # Evict old key from slot
                old_key = self.slot_key[slot]
                if old_key is not None:
                    del self.key_to_slot[old_key]

                # Install new key
                self.slot_key[slot] = key
                self.slot_freq[slot] = 1
                self.key_to_slot[key] = slot
                epoch_slots.add(slot)

        return hits, len(keys) - hits


class LFUAgingCache:
    """LFU-aging with slot-based eviction matching the real C implementation."""
    def __init__(self, capacity):
        self.capacity = capacity
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
                for i in range(self.capacity):
                    self.slot_freq[i] >>= 1

            if key in self.key_to_slot:
                slot = self.key_to_slot[key]
                self.slot_freq[slot] += 1
                epoch_slots.add(slot)
                hits += 1
            else:
                if self.n_filled < self.capacity:
                    slot = self.n_filled
                    self.n_filled += 1
                else:
                    slot = -1
                    min_f = float('inf')
                    for i in range(self.capacity):
                        if i in epoch_slots:
                            continue
                        if self.slot_freq[i] < min_f:
                            min_f = self.slot_freq[i]
                            slot = i
                    if slot < 0:
                        continue

                old_key = self.slot_key[slot]
                if old_key is not None:
                    del self.key_to_slot[old_key]

                self.slot_key[slot] = key
                self.slot_freq[slot] = 1
                self.key_to_slot[key] = slot
                epoch_slots.add(slot)

        return hits, len(keys) - hits


class TwoQueueCache:
    """2Q with epoch-aware eviction."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.a1in_max = max(1, capacity // 4)
        self.am_max = capacity - self.a1in_max
        self.a1out_max = capacity // 2
        self.a1in = OrderedDict()
        self.am = OrderedDict()
        self.a1out = OrderedDict()

    def batch_access(self, keys):
        hits = 0
        for key in keys:
            if key in self.am:
                self.am.move_to_end(key)
                hits += 1
            elif key in self.a1in:
                del self.a1in[key]
                self._admit_to_am(key)
                hits += 1
            else:
                if key in self.a1out:
                    del self.a1out[key]
                    self._admit_to_am(key)
                else:
                    if len(self.a1in) >= self.a1in_max:
                        ek, _ = self.a1in.popitem(last=False)
                        if len(self.a1out) >= self.a1out_max:
                            self.a1out.popitem(last=False)
                        self.a1out[ek] = True
                    self.a1in[key] = True
        return hits, len(keys) - hits

    def _admit_to_am(self, key):
        if len(self.am) >= self.am_max:
            self.am.popitem(last=False)
        self.am[key] = True


class ARCCache:
    """Adaptive Replacement Cache with batch interface."""
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
            if self._access_one(key):
                hits += 1
        return hits, len(keys) - hits

    def _access_one(self, key):
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


class LRU2Cache:
    """LRU-2 with epoch protection."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> [last, second_last]
        self.clock = 0

    def batch_access(self, keys):
        hits = 0
        protected = set()
        for key in keys:
            self.clock += 1
            if key in self.cache:
                self.cache[key][1] = self.cache[key][0]
                self.cache[key][0] = self.clock
                hits += 1
            else:
                if len(self.cache) >= self.capacity:
                    victim = None
                    min_sl = float('inf')
                    for k, (_, sl) in self.cache.items():
                        if k not in protected and sl < min_sl:
                            min_sl = sl
                            victim = k
                    if victim is not None:
                        del self.cache[victim]
                self.cache[key] = [self.clock, 0]
            protected.add(key)
        return hits, len(keys) - hits


class StaticPinLRUCache:
    """Pin top-N popular slots after warmup, LRU for the rest."""
    def __init__(self, capacity, warmup_batches=100):
        self.capacity = capacity
        self.warmup_batches = warmup_batches
        self.warmup_freq = defaultdict(int)
        self.warmup_done = False
        self.batch_count = 0
        self.pinned = set()
        self.lru = OrderedDict()
        self.lru_cap = capacity

    def batch_access(self, keys):
        self.batch_count += 1

        if not self.warmup_done:
            for key in keys:
                self.warmup_freq[key] += 1
            if self.batch_count >= self.warmup_batches:
                self._finish_warmup()
            # During warmup, use plain LRU
            return self._lru_batch(keys, self.capacity)

        hits = 0
        for key in keys:
            if key in self.pinned:
                hits += 1
            elif key in self.lru:
                self.lru.move_to_end(key)
                hits += 1
            else:
                if len(self.lru) >= self.lru_cap:
                    self.lru.popitem(last=False)
                self.lru[key] = True
        return hits, len(keys) - hits

    def _finish_warmup(self):
        n_pin = min(self.capacity // 3, len(self.warmup_freq))
        sorted_keys = sorted(self.warmup_freq.items(), key=lambda x: -x[1])
        self.pinned = set(k for k, _ in sorted_keys[:n_pin])
        self.lru_cap = self.capacity - len(self.pinned)
        self.lru = OrderedDict()
        self.warmup_done = True

    def _lru_batch(self, keys, cap):
        hits = 0
        batch_set = set(keys)
        for key in keys:
            if key in self.lru:
                self.lru.move_to_end(key)
                hits += 1
            else:
                while len(self.lru) >= cap:
                    for ck in self.lru:
                        if ck not in batch_set:
                            del self.lru[ck]
                            break
                    else:
                        break
                self.lru[key] = True
        return hits, len(keys) - hits


class WTinyLFUCache:
    """Window-TinyLFU with Count-Min Sketch admission filter."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.win_max = max(1, capacity // 100) or 1
        self.prot_max = max(1, int(capacity * 0.79))
        self.prob_max = capacity - self.win_max - self.prot_max
        if self.prob_max < 1:
            self.prob_max = 1
            self.prot_max = capacity - self.win_max - self.prob_max

        self.window = OrderedDict()
        self.probation = OrderedDict()
        self.protected = OrderedDict()

        self.sketch_w = max(64, capacity * 4)
        self.sketch = [[0] * self.sketch_w for _ in range(4)]
        self.seeds = [0x9e3779b9, 0x517cc1b7, 0x6a09e667, 0xbb67ae85]
        self.sk_count = 0
        self.sk_reset = max(capacity * 10, 1000)

    def _sk_hash(self, key, seed):
        h = hash(key) ^ seed
        h = ((h >> 16) ^ h) * 0x45d9f3b
        return h % self.sketch_w

    def _sk_add(self, key):
        self.sk_count += 1
        if self.sk_count >= self.sk_reset:
            self.sk_count = 0
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
                self._add_protected(key)
                hits += 1
            elif key in self.window:
                self.window.move_to_end(key)
                hits += 1
            else:
                evicted = None
                if len(self.window) >= self.win_max:
                    ek, _ = self.window.popitem(last=False)
                    evicted = ek
                self.window[key] = True
                if evicted is not None:
                    if len(self.probation) < self.prob_max:
                        self.probation[evicted] = True
                    else:
                        victim = next(iter(self.probation))
                        if self._sk_est(evicted) > self._sk_est(victim):
                            del self.probation[victim]
                            self.probation[evicted] = True
        return hits, len(keys) - hits

    def _add_protected(self, key):
        if len(self.protected) >= self.prot_max:
            dk, _ = self.protected.popitem(last=False)
            if len(self.probation) >= self.prob_max:
                self.probation.popitem(last=False)
            self.probation[dk] = True
        self.protected[key] = True


# =============================================================================
# Belady's optimal (standalone, processes flat sequence)
# =============================================================================

def simulate_belady_batched(batches, capacity):
    """Belady's optimal with batch-aware processing."""
    from bisect import bisect_right

    # Flatten for position tracking
    flat = []
    for batch in batches:
        flat.extend(batch)

    positions = defaultdict(list)
    for i, key in enumerate(flat):
        positions[key].append(i)

    cache = set()
    total_hits = 0
    total_misses = 0
    pos = 0

    for batch in batches:
        batch_hits = 0
        batch_misses = 0
        new_entries = []

        for key in batch:
            if key in cache:
                batch_hits += 1
            else:
                batch_misses += 1
                new_entries.append(key)
            pos += 1

        # Add all new entries, evicting as needed (but never evict batch members)
        batch_set = set(batch)
        for key in new_entries:
            if key in cache:
                continue  # already added by earlier miss in same batch
            if len(cache) >= capacity:
                farthest = -1
                victim = None
                for ck in cache:
                    if ck in batch_set:
                        continue
                    pidx = bisect_right(positions[ck], pos - 1)
                    nu = positions[ck][pidx] if pidx < len(positions[ck]) else len(flat) + 1
                    if nu > farthest:
                        farthest = nu
                        victim = ck
                if victim is not None:
                    cache.discard(victim)
                else:
                    break  # can't evict anything (all in batch)
            cache.add(key)

        total_hits += batch_hits
        total_misses += batch_misses

    return total_hits, total_misses


# =============================================================================
# Simulation runner
# =============================================================================

def simulate_policy(policy_class, batches, capacity, **kwargs):
    cache = policy_class(capacity, **kwargs)
    total_hits = 0
    total_misses = 0
    for batch in batches:
        h, m = cache.batch_access(batch)
        total_hits += h
        total_misses += m
    return total_hits, total_misses


def main():
    parser = argparse.ArgumentParser(description="Simulate cache policies on expert access traces")
    parser.add_argument("experiment_dir", help="Directory containing domain-*/traces/ subdirs")
    parser.add_argument("--cache-sizes", default="100,150,200,220,240,250,260,270,280,288,300,350,400,498,600,740",
                        help="Comma-separated cache sizes")
    parser.add_argument("--domains", default=None, help="Comma-separated domain names (default: all)")
    parser.add_argument("--policies", default="lru,lfu,lfu_aging,2q,arc,lru2,static_pin,w_tinylfu,belady",
                        help="Comma-separated policy names")
    args = parser.parse_args()

    cache_sizes = [int(x) for x in args.cache_sizes.split(",")]
    requested_policies = set(args.policies.split(","))

    exp_dir = Path(args.experiment_dir)
    if args.domains:
        domain_names = args.domains.split(",")
    else:
        domain_names = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and (d / "traces").exists()])

    policy_map = {
        "lru":        (LRUCache, {}),
        "lfu":        (LFUCache, {}),
        "lfu_aging":  (LFUAgingCache, {}),
        "2q":         (TwoQueueCache, {}),
        "arc":        (ARCCache, {}),
        "lru2":       (LRU2Cache, {}),
        "static_pin": (StaticPinLRUCache, {"warmup_batches": 24 * 50}),
        "w_tinylfu":  (WTinyLFUCache, {}),
    }

    policies_to_run = [p for p in ["lru", "lfu", "lfu_aging", "2q", "arc", "lru2", "static_pin", "w_tinylfu", "belady"]
                       if p in requested_policies]

    print(f"{'Domain':<20} {'Cache':>5}", end="")
    for p in policies_to_run:
        print(f" {p:>12}", end="")
    print()
    print("-" * (27 + 13 * len(policies_to_run)))

    all_results = {}

    for domain in domain_names:
        trace_dir = exp_dir / domain / "traces"
        if not trace_dir.exists():
            print(f"WARNING: {trace_dir} not found, skipping", file=sys.stderr)
            continue

        print(f"\nLoading {domain}...", file=sys.stderr, end=" ", flush=True)
        batches = load_access_batches(str(trace_dir))
        n_accesses = sum(len(b) for b in batches)
        print(f"{len(batches)} batches, {n_accesses} accesses", file=sys.stderr, flush=True)

        domain_results = {}

        for cs in cache_sizes:
            row = {}
            for pname in policies_to_run:
                if pname == "belady":
                    hits, misses = simulate_belady_batched(batches, cs)
                else:
                    cls, kwargs = policy_map[pname]
                    hits, misses = simulate_policy(cls, batches, cs, **kwargs)

                miss_rate = misses / n_accesses * 100
                row[pname] = miss_rate

            domain_results[cs] = row

            print(f"{domain:<20} {cs:>5}", end="")
            for pname in policies_to_run:
                print(f" {row[pname]:>11.1f}%", end="")
            print()

        all_results[domain] = domain_results

    # Print average across domains
    if len(all_results) > 1:
        print(f"\n{'--- AVERAGE ---':<20} {'':>5}", end="")
        for p in policies_to_run:
            print(f" {p:>12}", end="")
        print()
        print("-" * (27 + 13 * len(policies_to_run)))

        for cs in cache_sizes:
            print(f"{'AVERAGE':<20} {cs:>5}", end="")
            for pname in policies_to_run:
                vals = [all_results[d][cs][pname] for d in all_results if cs in all_results[d]]
                avg = sum(vals) / len(vals)
                print(f" {avg:>11.1f}%", end="")
            print()


if __name__ == "__main__":
    main()
