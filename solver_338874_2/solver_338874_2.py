import math
import copy
import random
import time as _time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from .abstract_solver import AbstractSolver
from solver_338874.solver_338874 import solver_338874 as _SolverBase

INFEASIBLE = 100000

# ============================================================================
# Performance-tuning constants
# ============================================================================
# Cap on simultaneously-tracked EMS per bin.  Drastically reduces the O(|EMS|^2)
# growth in Bin.update; pruning keeps the largest volumes (most useful).
DEFAULT_MAX_EMS_PER_BIN = 64

# DFTRC_2 early exit: stop scanning EMS once we found one whose distance score
# >= EARLY_EXIT_FRACTION * theoretical_max.  Set to 1.0 to disable.
EARLY_EXIT_FRACTION = 1.0


def _bo_orient(box, BO):
    """Apply BRKGA orientation BO (1–6) to a (depth, width, height) tuple."""
    d, w, h = box
    if   BO == 1: return (d, w, h)
    elif BO == 2: return (d, h, w)
    elif BO == 3: return (w, d, h)
    elif BO == 4: return (w, h, d)
    elif BO == 5: return (h, d, w)
    elif BO == 6: return (h, w, d)


# ============================================================================
# Bin: EMS bookkeeping
# OPT: vectorised update(), EMS cap, vectorised gravity_ok, vectorised load()
# ============================================================================
class Bin():
    def __init__(self, V, max_value=float('inf'), gravity_strength=0.0,
                 max_ems=DEFAULT_MAX_EMS_PER_BIN, verbose=False):
        self.dimensions = tuple(V)
        # Store EMSs as np.int64 (2,3) arrays: row 0 = min, row 1 = max.
        # A python list of these gives O(1) indexing AND lets us np.stack for
        # block-vectorised ops in update()/DFTRC_2.
        self.EMSs = [np.array([(0, 0, 0), V], dtype=np.int64)]
        # Cached stacked view (rebuilt on demand)
        self._ems_block = None
        self._ems_dirty = True
        # Items as np.int64 (2,3) min/max arrays for vectorised gravity tests.
        self.load_items = []
        self.current_value = 0.0
        self.max_value = max_value
        self.gravity_strength = gravity_strength
        self.max_ems = int(max_ems) if max_ems else None
        self._bin_volume = int(V[0]) * int(V[1]) * int(V[2])

        if verbose:
            print('Init EMSs:', self.EMSs)

    def __getitem__(self, index):
        return self.EMSs[index]

    def __len__(self):
        return len(self.EMSs)

    # ----- internal helpers -------------------------------------------------
    def _invalidate_cache(self):
        self._ems_dirty = True

    def _refresh_cache(self):
        if not self._ems_dirty:
            return
        if not self.EMSs:
            self._ems_block = np.zeros((0, 2, 3), dtype=np.int64)
        else:
            self._ems_block = np.stack(self.EMSs, axis=0)
        self._ems_dirty = False

    def _prune_ems(self):
        """Cap EMS list size, keeping the highest-volume entries (vectorised)."""
        if self.max_ems is None or len(self.EMSs) <= self.max_ems:
            return
        block = np.stack(self.EMSs, axis=0)  # (K,2,3)
        sizes = block[:, 1, :] - block[:, 0, :]
        vols = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
        keep_idx = np.argpartition(-vols, self.max_ems - 1)[: self.max_ems]
        keep_idx.sort()
        self.EMSs = [self.EMSs[i] for i in keep_idx]
        self._invalidate_cache()

    # ----- main update (vectorised) -----------------------------------------
    def update(self, box, selected_EMS, min_vol=1, min_dim=1, verbose=False):
        # 1. Place box
        boxToPlace = np.asarray(box, dtype=np.int64)
        selected_min = np.asarray(selected_EMS[0], dtype=np.int64)
        ems_min = selected_min
        ems_max = selected_min + boxToPlace
        placed = np.stack([ems_min, ems_max], axis=0)
        self.load_items.append(placed)

        if not self.EMSs:
            return

        # 2. Vectorised overlap detection: ems overlaps iff
        #    (ems.max > placed.min) all axes AND (ems.min < placed.max) all axes
        ems_arr = np.stack(self.EMSs, axis=0)  # (K,2,3)
        overlap_mask = np.all(ems_arr[:, 1, :] > ems_min, axis=1) & \
                       np.all(ems_arr[:, 0, :] < ems_max, axis=1)
        overlap_indices = np.where(overlap_mask)[0]

        if overlap_indices.size == 0:
            self._invalidate_cache()
            self._prune_ems()
            return

        kept = [self.EMSs[i] for i in range(len(self.EMSs)) if not overlap_mask[i]]

        x3, y3, z3 = ems_min.tolist()
        x4, y4, z4 = ems_max.tolist()

        # 3. Generate candidate cuts (3 per overlapped EMS) — small Python loop
        new_candidates = []
        for idx in overlap_indices:
            EMS = self.EMSs[idx]
            x1, y1, z1 = EMS[0].tolist()
            x2, y2, z2 = EMS[1].tolist()
            new_candidates.append(np.array([(x4, y1, z1), (x2, y2, z2)], dtype=np.int64))
            new_candidates.append(np.array([(x1, y4, z1), (x2, y2, z2)], dtype=np.int64))
            new_candidates.append(np.array([(x1, y1, z4), (x2, y2, z2)], dtype=np.int64))

        if not new_candidates:
            self.EMSs = kept
            self._invalidate_cache()
            self._prune_ems()
            return

        cand_block = np.stack(new_candidates, axis=0)
        sizes = cand_block[:, 1, :] - cand_block[:, 0, :]

        # 4. Filter degenerate/too-small candidates (vectorised)
        min_dims_per_cand = sizes.min(axis=1)
        vol_per_cand = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
        valid_mask = (
            (min_dims_per_cand >= int(min_dim)) &
            (vol_per_cand >= int(min_vol)) &
            (sizes > 0).all(axis=1)
        )
        if not valid_mask.any():
            self.EMSs = kept
            self._invalidate_cache()
            self._prune_ems()
            return
        valid_cands = cand_block[valid_mask]

        # 5. Drop candidates that are totally inscribed by some kept EMS OR by
        # another candidate.  Vectorised via broadcasting.
        if kept:
            kept_block = np.stack(kept, axis=0)
        else:
            kept_block = np.zeros((0, 2, 3), dtype=np.int64)
        ref_block = np.concatenate([kept_block, valid_cands], axis=0)

        n_cand = valid_cands.shape[0]

        cand_min = valid_cands[:, 0, :][:, None, :]      # (n_cand,1,3)
        cand_max = valid_cands[:, 1, :][:, None, :]
        ref_min  = ref_block[:, 0, :][None, :, :]        # (1,n_ref,3)
        ref_max  = ref_block[:, 1, :][None, :, :]

        inscribed = np.all(ref_min <= cand_min, axis=2) & \
                    np.all(cand_max <= ref_max, axis=2)
        # Skip self-comparison (each candidate is also at row kept_block.size+i in ref_block)
        self_idx = np.arange(n_cand) + kept_block.shape[0]
        inscribed[np.arange(n_cand), self_idx] = False
        is_inscribed_anywhere = inscribed.any(axis=1)

        survivors = valid_cands[~is_inscribed_anywhere]

        self.EMSs = kept + [survivors[i] for i in range(survivors.shape[0])]
        self._invalidate_cache()
        self._prune_ems()

    # ----- legacy / compat --------------------------------------------------
    def overlapped(self, ems, EMS):
        ems_min, ems_max = ems[0], ems[1]
        EMS_min, EMS_max = EMS[0], EMS[1]
        return bool(np.all(ems_max > EMS_min) and np.all(ems_min < EMS_max))

    def inscribed(self, ems, EMS):
        return bool(np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]))

    def eliminate(self, ems):
        ems_t = (tuple(ems[0]), tuple(ems[1]))
        for i, E in enumerate(self.EMSs):
            if (tuple(E[0]), tuple(E[1])) == ems_t:
                self.EMSs.pop(i)
                self._invalidate_cache()
                return

    def get_EMSs(self):
        return [(tuple(E[0]), tuple(E[1])) for E in self.EMSs]

    def gravity_ok(self, ems_min, box_dims):
        if self.gravity_strength <= 0.0:
            return True
        x, y, z = int(ems_min[0]), int(ems_min[1]), int(ems_min[2])
        if z == 0:
            return True
        if not self.load_items:
            return False
        dx, dy = int(box_dims[0]), int(box_dims[1])

        items = np.stack(self.load_items, axis=0)
        z_mask = items[:, 1, 2] == z
        if not z_mask.any():
            return False
        cand = items[z_mask]
        ix0 = cand[:, 0, 0]; iy0 = cand[:, 0, 1]
        ix1 = cand[:, 1, 0]; iy1 = cand[:, 1, 1]
        ox = np.maximum(0, np.minimum(x + dx, ix1) - np.maximum(x, ix0))
        oy = np.maximum(0, np.minimum(y + dy, iy1) - np.maximum(y, iy0))
        supported = int((ox * oy).sum())
        return supported >= self.gravity_strength * dx * dy - 0.001

    def load(self):
        if not self.load_items:
            return 0.0
        items = np.stack(self.load_items, axis=0)
        sizes = items[:, 1, :] - items[:, 0, :]
        return float((sizes[:, 0] * sizes[:, 1] * sizes[:, 2]).sum()) / self._bin_volume


# ============================================================================
# PlacementProcedure
# OPT: pre-filter EMS by volume, early exit in DFTRC_2, vectorised
#      elimination_rule, cached oriented dims
# ============================================================================
class PlacementProcedure():
    def __init__(self, inputs, solution, verbose=False):
        n_types  = len(inputs['V'])
        n_items  = len(inputs['v'])
        max_bins = inputs.get('max_bins', n_items)

        # Store inputs so _open_bin can read type pool and per-type params
        self._inputs   = inputs
        self._n_types  = n_types
        self._max_bins = max_bins
        self._type_cdf = inputs.get('type_cdf', [(i + 1) / n_types for i in range(n_types)])

        # Bins and slot→type mapping allocated lazily as bins are opened
        self.Bins         = []
        self._slot_vtypes = []

        self.boxes             = inputs['v']
        self.allowed_rotations = inputs.get('allowed_rotations', [[1, 2, 3, 4, 5, 6]] * n_items)
        self.item_values       = inputs.get('item_values',       [0.0] * n_items)
        self.BPS = np.argsort(solution[:n_items])
        self.VBO = solution[n_items:2 * n_items]
        btg_raw  = np.asarray(solution[2 * n_items:], dtype=np.float64)
        if len(btg_raw) >= max_bins:
            self.BTG = np.clip(btg_raw[:max_bins], 0.0, 1.0 - 1e-9)
        else:
            self.BTG = np.random.uniform(0.0, 1.0 - 1e-9, max_bins)
        self.num_opend_bins = 0

        self.verbose   = verbose
        self.infisible = False
        self.placement()

    def _select_bin_type(self, gene):
        """Map gene [0,1) to a type index via cost-weighted CDF (cheap types get larger buckets)."""
        idx = int(np.searchsorted(self._type_cdf, float(gene), side='right'))
        return min(idx, self._n_types - 1)

    def _open_bin(self, slot):
        """Create a new Bin for slot using the BTG gene; append to self.Bins and self._slot_vtypes."""
        gene     = float(self.BTG[slot]) if slot < len(self.BTG) else random.random()
        type_idx = self._select_bin_type(gene)
        inp      = self._inputs
        b = Bin(
            inp['V'][type_idx],
            max_value=inp['max_values'][type_idx],
            gravity_strength=inp['gravity_strengths'][type_idx],
            max_ems=inp.get('max_ems_per_bin', DEFAULT_MAX_EMS_PER_BIN),
        )
        self.Bins.append(b)
        self._slot_vtypes.append(inp['vehicle_type_keys'][type_idx])

    def placement(self):
        items_sorted   = [self.boxes[i]             for i in self.BPS]
        allowed_sorted = [self.allowed_rotations[i] for i in self.BPS]
        values_sorted  = [self.item_values[i]       for i in self.BPS]

        for i, box in enumerate(items_sorted):
            allowed    = allowed_sorted[i]
            item_value = values_sorted[i]
            box_vol = box[0] * box[1] * box[2]

            selected_bin = None
            selected_EMS = None
            for k in range(self.num_opend_bins):
                if self.Bins[k].current_value + item_value > self.Bins[k].max_value:
                    continue
                EMS = self.DFTRC_2(box, k, allowed_directions=allowed, box_vol=box_vol)
                if EMS is not None:
                    selected_bin = k
                    selected_EMS = EMS
                    break

            if selected_bin is None:
                if self.num_opend_bins >= self._max_bins:
                    self.infisible = True
                    return
                self._open_bin(self.num_opend_bins)
                selected_bin = self.num_opend_bins
                self.num_opend_bins += 1
                selected_EMS = self.Bins[selected_bin].EMSs[0]

            BO = self.selecte_box_orientaion(
                self.VBO[i], box, selected_EMS,
                allowed_directions=allowed, bin_idx=selected_bin
            )
            min_vol, min_dim = self.elimination_rule(items_sorted[i + 1:])
            self.Bins[selected_bin].current_value += item_value
            self.Bins[selected_bin].update(self.orient(box, BO), selected_EMS, min_vol, min_dim)

    # -----------------------------------------------------------------------
    # DFTRC_2 with pre-filter (vol >= box_vol) + early exit (distance threshold)
    # -----------------------------------------------------------------------
    def DFTRC_2(self, box, k, allowed_directions=None, box_vol=None):
        if allowed_directions is None:
            allowed_directions = [1, 2, 3, 4, 5, 6]
        bin_obj = self.Bins[k]
        if not bin_obj.EMSs:
            return None
        if box_vol is None:
            box_vol = box[0] * box[1] * box[2]

        D, W, H = bin_obj.dimensions

        # OPT: Pre-filter EMS block by volume — skip EMSs that can't fit the box
        bin_obj._refresh_cache()
        ems_block = bin_obj._ems_block
        sizes = ems_block[:, 1, :] - ems_block[:, 0, :]
        vols = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
        feasible_idx = np.where(vols >= box_vol)[0]
        if feasible_idx.size == 0:
            return None

        # OPT: Cache oriented dims so we don't recompute per EMS
        oriented = {dirn: self.orient(box, dirn) for dirn in allowed_directions}

        # Heuristic scan order: small (x+y+z) first — far-back-top-right candidates
        # tend to score high quickly, helping the early-exit fire sooner.
        origins = ems_block[feasible_idx, 0, :]
        scan_order = feasible_idx[np.argsort(origins[:, 0] + origins[:, 1] + origins[:, 2])]

        # Early-exit threshold: a "good enough" EMS gets us out of the scan loop
        min_d = min(box)
        theoretical_max = (D - min_d) ** 2 + (W - min_d) ** 2 + (H - min_d) ** 2
        early_threshold = EARLY_EXIT_FRACTION * theoretical_max

        maxDist = -1
        selectedEMS = None

        for ems_idx in scan_order:
            EMS = bin_obj.EMSs[ems_idx]
            ex0 = int(EMS[0][0]); ey0 = int(EMS[0][1]); ez0 = int(EMS[0][2])
            edx = int(EMS[1][0]) - ex0
            edy = int(EMS[1][1]) - ey0
            edz = int(EMS[1][2]) - ez0

            for direction in allowed_directions:
                d, w, h = oriented[direction]
                if d > edx or w > edy or h > edz:
                    continue
                if not bin_obj.gravity_ok(EMS[0], (d, w, h)):
                    continue
                distance = (D - ex0 - d) ** 2 + (W - ey0 - w) ** 2 + (H - ez0 - h) ** 2
                if distance > maxDist:
                    maxDist = distance
                    selectedEMS = EMS
                    if distance >= early_threshold:
                        return selectedEMS
        return selectedEMS

    def orient(self, box, BO=1):
        d, w, h = box
        if   BO == 1: return (d, w, h)
        elif BO == 2: return (d, h, w)
        elif BO == 3: return (w, d, h)
        elif BO == 4: return (w, h, d)
        elif BO == 5: return (h, d, w)
        elif BO == 6: return (h, w, d)

    def selecte_box_orientaion(self, VBO, box, EMS, allowed_directions=None, bin_idx=None):
        if allowed_directions is None:
            allowed_directions = [1, 2, 3, 4, 5, 6]
        bin_ = self.Bins[bin_idx] if bin_idx is not None else None

        BOs = []
        for direction in allowed_directions:
            dims = self.orient(box, direction)
            if self.fitin(dims, EMS):
                if bin_ is not None and not bin_.gravity_ok(EMS[0], dims):
                    continue
                BOs.append(direction)
        if not BOs:
            for direction in allowed_directions:
                if self.fitin(self.orient(box, direction), EMS):
                    BOs.append(direction)
        return BOs[math.ceil(VBO * len(BOs)) - 1]

    def fitin(self, box, EMS):
        return (box[0] <= EMS[1][0] - EMS[0][0] and
                box[1] <= EMS[1][1] - EMS[0][1] and
                box[2] <= EMS[1][2] - EMS[0][2])

    def elimination_rule(self, remaining_boxes):
        if not remaining_boxes:
            return 0, 0
        # OPT: one numpy reduction instead of two python loops
        arr = np.asarray(remaining_boxes, dtype=np.int64)
        min_dim = int(arr.min())
        min_vol = int((arr[:, 0] * arr[:, 1] * arr[:, 2]).min())
        return min_vol, min_dim

    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        leastLoad = 1
        for k in range(self.num_opend_bins):
            load = self.Bins[k].load()
            if load < leastLoad:
                leastLoad = load
        return self.num_opend_bins + leastLoad % 1


# ============================================================================
# PermutationGA: permutation-based GA with OX1 crossover
# ============================================================================
class PermutationGA():
    def __init__(self, inputs, num_generations=200, num_individuals=120,
                 multiProcess=False, n_workers=1):
        self.inputs = inputs
        self.N = len(inputs['v'])
        self.max_bins = inputs.get('max_bins', self.N)
        self.num_generations = int(num_generations)
        self.num_individuals = int(num_individuals)
        self.n_workers = max(1, int(n_workers))

        self.used_bins    = -1
        self.solution     = None
        self.best_fitness = -1
        self.history = {'mean': [], 'min': []}

    def _to_brkga_format(self, seq, rots, btg=None):
        """Convert (sequence, rotations, bin-type genes) to float array for PlacementProcedure."""
        floats = np.zeros(2 * self.N + self.max_bins)
        ranks = np.zeros(self.N)
        ranks[seq] = np.arange(self.N)
        floats[:self.N] = ranks / float(self.N)
        floats[self.N:2 * self.N] = rots
        if btg is not None:
            btg_arr = np.asarray(btg, dtype=np.float64)
            L = min(len(btg_arr), self.max_bins)
            floats[2 * self.N : 2 * self.N + L] = btg_arr[:L]
        return floats

    def _eval_individual(self, ind):
        floats = self._to_brkga_format(ind[0], ind[1], ind[2])
        return PlacementProcedure(self.inputs, floats).evaluate()

    def cal_fitness(self, population):
        if self.n_workers > 1 and len(population) >= self.n_workers * 2:
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                return list(ex.map(self._eval_individual, population))
        return [self._eval_individual(ind) for ind in population]

    def _ox1_crossover(self, seq1, seq2):
        """Order Crossover (OX1) for permutation sequences."""
        N = self.N
        if N < 2:
            return seq1.copy()
        c1, c2 = sorted(random.sample(range(N), 2))
        child = np.empty(N, dtype=np.int64)
        child[c1:c2 + 1] = seq1[c1:c2 + 1]
        in_child = set(int(x) for x in child[c1:c2 + 1])
        fill_vals = [int(x) for x in seq2 if int(x) not in in_child]
        j = 0
        for i in list(range(c2 + 1, N)) + list(range(0, c1)):
            child[i] = fill_vals[j]
            j += 1
        return child

    def _uniform_crossover_rots(self, rots1, rots2):
        """Uniform crossover (50/50) for rotation genes."""
        mask = np.random.random(self.N) < 0.5
        return np.where(mask, rots1, rots2)

    def _uniform_crossover_btg(self, btg1, btg2):
        """Uniform crossover (50/50) for bin-type genes."""
        mask = np.random.random(self.max_bins) < 0.5
        return np.where(mask, btg1, btg2)

    def _mutate(self, seq, rots, btg, mut_prob=0.10):
        """Swap mutation for sequence; per-gene random replacement for rotations and BTG."""
        seq  = seq.copy()
        rots = rots.copy()
        btg  = btg.copy()
        if self.N >= 2 and random.random() < mut_prob:
            i, j = random.sample(range(self.N), 2)
            seq[[i, j]] = seq[[j, i]]
        mut_mask = np.random.random(self.N) < mut_prob
        if mut_mask.any():
            rots[mut_mask] = np.random.uniform(0.0, 1.0, size=int(mut_mask.sum()))
        btg_mask = np.random.random(self.max_bins) < mut_prob
        if btg_mask.any():
            btg[btg_mask] = np.random.uniform(0.0, 1.0, size=int(btg_mask.sum()))
        return seq, rots, btg

    def _tournament_select(self, population, fitness_list, k=3):
        k = min(k, len(population))
        candidates = random.sample(range(len(population)), k)
        best = min(candidates, key=lambda i: fitness_list[i])
        return population[best]

    def fit(self, patient=4, verbose=False, seed_pool=None,
            time_budget=None, stagnation_ratio=None):
        t_start = _time.time()
        N = self.N
        n_elite = max(1, int(self.num_individuals * 0.10))

        # Build initial population from GRASP seeds + random permutations
        population = []
        if seed_pool:
            for ind in seed_pool[:self.num_individuals]:
                population.append((np.asarray(ind[0], dtype=np.int64).copy(),
                                   np.asarray(ind[1], dtype=np.float64).copy(),
                                   np.asarray(ind[2], dtype=np.float64).copy()))
        while len(population) < self.num_individuals:
            seq  = np.random.permutation(N).astype(np.int64)
            rots = np.random.uniform(0.0, 1.0, size=N)
            btg  = np.random.uniform(0.0, 1.0, size=self.max_bins)
            population.append((seq, rots, btg))

        fitness_list = self.cal_fitness(population)
        best_idx = int(np.argmin(fitness_list))
        best_fitness = float(fitness_list[best_idx])
        best_solution = (population[best_idx][0].copy(), population[best_idx][1].copy(),
                         population[best_idx][2].copy())

        self.history['min'].append(best_fitness)
        self.history['mean'].append(float(np.mean(fitness_list)))

        best_iter = 0
        last_improvement_time = _time.time()

        for g in range(self.num_generations):
            if time_budget is not None and (_time.time() - t_start) > time_budget:
                if verbose:
                    print(f'[PermGA] Time budget {time_budget}s reached at gen {g}.')
                break
            if g - best_iter > patient:
                if verbose:
                    print(f'[PermGA] Patience exhausted at gen {g}.')
                break
            if stagnation_ratio is not None and self.num_generations > 0:
                if (g / max(1, self.num_generations)) > stagnation_ratio:
                    elapsed = _time.time() - t_start
                    idle = _time.time() - last_improvement_time
                    if elapsed > 0 and idle / elapsed > 0.25:
                        if verbose:
                            print(f'[PermGA] Adaptive stagnation at gen {g}.')
                        break

            # Elitism: top 10% pass unchanged
            sorted_idx = np.argsort(fitness_list)
            elite_pop = [population[i] for i in sorted_idx[:n_elite]]
            elite_fit = [fitness_list[i] for i in sorted_idx[:n_elite]]

            # Fill remaining 90% with OX1 crossover + mutation
            offspring = []
            while len(offspring) < self.num_individuals - n_elite:
                p1 = self._tournament_select(population, fitness_list)
                p2 = self._tournament_select(population, fitness_list)
                child_seq  = self._ox1_crossover(p1[0], p2[0])
                child_rots = self._uniform_crossover_rots(p1[1], p2[1])
                child_btg  = self._uniform_crossover_btg(p1[2], p2[2])
                child_seq, child_rots, child_btg = self._mutate(child_seq, child_rots, child_btg)
                offspring.append((child_seq, child_rots, child_btg))

            offspring_fitness = self.cal_fitness(offspring)

            population = elite_pop + offspring
            fitness_list = elite_fit + list(offspring_fitness)

            current_min = float(np.min(fitness_list))
            if current_min < best_fitness:
                best_iter = g
                best_fitness = current_min
                best_idx = int(np.argmin(fitness_list))
                best_solution = (population[best_idx][0].copy(), population[best_idx][1].copy(),
                                 population[best_idx][2].copy())
                last_improvement_time = _time.time()

            self.history['min'].append(current_min)
            self.history['mean'].append(float(np.mean(fitness_list)))

            if verbose:
                print(f'Gen {g}\t(best: {best_fitness:.4f})')

        self.used_bins    = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution     = self._to_brkga_format(best_solution[0], best_solution[1], best_solution[2])
        return 'feasible'


# ============================================================================
# Fast copy of a single active_bin dict (avoids ghost aliasing)
# ============================================================================
def _copy_bin(ab):
    return {
        'type': ab['type'],
        'idx': ab['idx'],
        'spaces': ab['spaces'].copy(),
        'current_weight': ab['current_weight'],
        'max_weight': ab['max_weight'],
        'current_vol': ab['current_vol'],
        'max_vol': ab['max_vol'],
        'current_value': ab.get('current_value', 0.0),
        'max_value_item': ab.get('max_value_item', float('inf')),
        'z_layers': {z: list(layer) for z, layer in ab.get('z_layers', {}).items()},
        'items': [rec.copy() for rec in ab['items']],
        'last_pruned': ab.get('last_pruned', 1),
    }


# ============================================================================
# GroupingGA: bin-centric GA that replaces PermutationGA
# ============================================================================
class GroupingGA:
    """
    Grouping Genetic Algorithm for 3D-BPP.

    Genotype  : list[active_bin]  — phenotype IS the chromosome.
    Crossover : inherit elite spatial structures from both parents; orphaned
                items are repaired in-place via DBLF (_try_repack_records).
    Mutation  : destroy 1-2 random bins and reinsert their items for exploration.
    Elitism   : top `elite_frac` of the population passes unchanged each generation.
    """

    _ELITE_UTIL_P1 = 0.85   # threshold for "elite" bin status in parent P1
    _ELITE_UTIL_P2 = 0.70   # threshold for "compatible" bins from parent P2
    _TOURNAMENT_K  = 3

    def __init__(self, solver, vehicles_dict, items,
                 num_generations=150, num_individuals=80,
                 elite_frac=0.10, mut_prob=0.15,
                 time_budget=None, n_workers=1):
        self.solver        = solver           # solver_338874_2 instance (all geometric methods)
        self.vehicles_dict = vehicles_dict
        self.items         = items
        self.num_generations  = int(num_generations)
        self.num_individuals  = int(num_individuals)
        self.elite_frac    = float(elite_frac)
        self.mut_prob      = float(mut_prob)
        self.time_budget   = time_budget
        self.n_workers     = max(1, int(n_workers))

        self.best_bins = None
        self.best_cost = float('inf')
        self.history   = {'mean': [], 'min': []}

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------
    def _eval_individual(self, bins):
        return self.solver._cost_from_bins(bins, self.vehicles_dict, self.items)

    def _cal_fitness(self, population):
        if self.n_workers > 1 and len(population) >= self.n_workers * 2:
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                return list(ex.map(self._eval_individual, population))
        return [self._eval_individual(ind) for ind in population]

    # ------------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------------
    def _tournament_select(self, population, fitness_list):
        k    = min(self._TOURNAMENT_K, len(population))
        idxs = random.sample(range(len(population)), k)
        return population[min(idxs, key=lambda i: fitness_list[i])]

    # ------------------------------------------------------------------
    # Grouping Crossover
    # ------------------------------------------------------------------
    def _grouping_crossover(self, p1_bins, p2_bins):
        """
        Phase A — Copy elite bins from P1 (util >= _ELITE_UTIL_P1) into the child.
        Phase B — From P2, copy high-util bins that share NO items with Phase A.
        Phase C — Collect all orphan records; reinsert via _try_repack_records.

        Parents are never modified; child_bins is a fresh list of copied bins.
        """
        s          = self.solver
        child_bins = []
        packed_ids: set = set()

        # Phase A: freeze elite spatial structures from P1
        for ab in p1_bins:
            if s._bin_utilization(ab) >= self._ELITE_UTIL_P1:
                child_bins.append(_copy_bin(ab))
                for rec in ab['items']:
                    packed_ids.add(rec['i_idx'])

        # Phase B: integrate non-conflicting high-util bins from P2
        for ab in p2_bins:
            if s._bin_utilization(ab) < self._ELITE_UTIL_P2:
                continue
            bin_ids = {rec['i_idx'] for rec in ab['items']}
            if bin_ids & packed_ids:        # any shared item → conflict, skip
                continue
            child_bins.append(_copy_bin(ab))
            packed_ids.update(bin_ids)

        # Phase C: collect orphan records (prefer P1 record metadata)
        seen: dict = {}
        for ab in p1_bins:
            for rec in ab['items']:
                seen.setdefault(rec['i_idx'], rec)
        for ab in p2_bins:
            for rec in ab['items']:
                seen.setdefault(rec['i_idx'], rec)

        orphans = [rec for i_idx, rec in seen.items() if i_idx not in packed_ids]

        if not orphans:
            return child_bins

        # Sort: priority desc, then volume desc — mirrors GRASP insertion order
        orphans.sort(
            key=lambda r: (r['item'].get('p_i', 0), r['dx'] * r['dy'] * r['dz']),
            reverse=True,
        )

        # Reinsertion: fill gaps in child bins first, open new bins as needed
        success, remaining = s._try_repack_records(
            orphans, child_bins, self.vehicles_dict, allow_new=True
        )
        # Defensive fallback (should not trigger with allow_new=True unless
        # the instance has item-value constraints that block all placements)
        if not success and remaining:
            s._try_repack_records(remaining, child_bins, self.vehicles_dict, allow_new=True)

        return child_bins

    # ------------------------------------------------------------------
    # Grouping Mutation
    # ------------------------------------------------------------------
    def _mutate_bins(self, bins):
        """
        With probability mut_prob: destroy 1-2 random bins, shuffle their items,
        and reinsert via _try_repack_records to favour spatial exploration.
        Modifies `bins` in-place and returns it.
        """
        if not bins or random.random() >= self.mut_prob:
            return bins

        n_destroy   = random.randint(1, min(2, len(bins)))
        destroy_set = set(random.sample(range(len(bins)), n_destroy))

        extracted = []
        for idx in destroy_set:
            extracted.extend(bins[idx]['items'])

        surviving = [b for i, b in enumerate(bins) if i not in destroy_set]

        if not extracted:
            return surviving

        random.shuffle(extracted)   # break geometric bias for exploration
        self.solver._try_repack_records(
            extracted, surviving, self.vehicles_dict, allow_new=True
        )
        return surviving

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------
    def fit(self, seed_population=None, patient=5, verbose=False):
        """
        Run the GroupingGA.

        Parameters
        ----------
        seed_population : list[list[active_bin]]
            Bootstrap chromosomes — each element is a list of active_bins
            produced directly by the GRASP phase.
        patient : int
            Stop after `patient` consecutive generations without improvement.
        verbose : bool

        Returns
        -------
        'feasible' or 'infeasible'
        """
        t_start = _time.time()
        s       = self.solver

        n_elite = max(1, int(self.num_individuals * self.elite_frac))

        # --- Build initial population from GRASP seeds ---
        population = []
        if seed_population:
            for bins in seed_population[:self.num_individuals]:
                population.append(s._fast_copy_bins(bins))

        # Pad remainder with mutated clones of seeds
        seed_list = seed_population or []
        while len(population) < self.num_individuals and seed_list:
            base  = random.choice(seed_list)
            clone = s._fast_copy_bins(base)
            clone = self._mutate_bins(clone)
            population.append(clone)

        if not population:
            return 'infeasible'

        # --- Initial fitness evaluation ---
        fitness_list = self._cal_fitness(population)
        best_idx     = int(np.argmin(fitness_list))
        self.best_cost = float(fitness_list[best_idx])
        self.best_bins = s._fast_copy_bins(population[best_idx])

        self.history['min'].append(self.best_cost)
        self.history['mean'].append(float(np.mean(fitness_list)))

        best_iter = 0

        for g in range(self.num_generations):
            # --- Stopping criteria ---
            if self.time_budget is not None and (_time.time() - t_start) > self.time_budget:
                if verbose:
                    print(f'[GroupingGA] Time budget reached at gen {g}.')
                break
            if g - best_iter > patient:
                if verbose:
                    print(f'[GroupingGA] Patience exhausted at gen {g}.')
                break

            # --- Elitism: top elite_frac pass unchanged ---
            sorted_idx = np.argsort(fitness_list)
            elite_pop  = [s._fast_copy_bins(population[i]) for i in sorted_idx[:n_elite]]
            elite_fit  = [fitness_list[i]                  for i in sorted_idx[:n_elite]]

            # --- Offspring: crossover + mutation ---
            offspring = []
            while len(offspring) < self.num_individuals - n_elite:
                p1    = self._tournament_select(population, fitness_list)
                p2    = self._tournament_select(population, fitness_list)
                child = self._grouping_crossover(p1, p2)
                child = self._mutate_bins(child)
                offspring.append(child)

            offspring_fitness = self._cal_fitness(offspring)

            population   = elite_pop + offspring
            fitness_list = elite_fit + list(offspring_fitness)

            current_min = float(np.min(fitness_list))
            if current_min < self.best_cost:
                best_iter      = g
                self.best_cost = current_min
                best_idx       = int(np.argmin(fitness_list))
                self.best_bins = s._fast_copy_bins(population[best_idx])

            self.history['min'].append(current_min)
            self.history['mean'].append(float(np.mean(fitness_list)))

            if verbose:
                print(f'[GroupingGA] Gen {g:03d} | best={self.best_cost:.2f} | '
                      f'mean={self.history["mean"][-1]:.2f}')

        return 'feasible' if self.best_bins is not None else 'infeasible'


# ============================================================================
# _TrackedPlacement: records per-item assignments
# ============================================================================
class _TrackedPlacement(PlacementProcedure):
    def __init__(self, inputs, solution):
        self.assignments = []
        super().__init__(inputs, solution)

    def placement(self):
        items_sorted   = [self.boxes[i]             for i in self.BPS]
        allowed_sorted = [self.allowed_rotations[i] for i in self.BPS]
        values_sorted  = [self.item_values[i]       for i in self.BPS]

        for i, box in enumerate(items_sorted):
            allowed    = allowed_sorted[i]
            item_value = values_sorted[i]
            box_vol = box[0] * box[1] * box[2]

            selected_bin = None
            selected_EMS = None
            for k in range(self.num_opend_bins):
                if self.Bins[k].current_value + item_value > self.Bins[k].max_value:
                    continue
                EMS = self.DFTRC_2(box, k, allowed_directions=allowed, box_vol=box_vol)
                if EMS is not None:
                    selected_bin = k
                    selected_EMS = EMS
                    break

            if selected_bin is None:
                if self.num_opend_bins >= self._max_bins:
                    self.infisible = True
                    return
                self._open_bin(self.num_opend_bins)
                selected_bin = self.num_opend_bins
                self.num_opend_bins += 1
                selected_EMS = self.Bins[selected_bin].EMSs[0]

            BO = self.selecte_box_orientaion(
                self.VBO[i], box, selected_EMS,
                allowed_directions=allowed, bin_idx=selected_bin
            )
            placed_box = self.orient(box, BO)
            min_vol, min_dim = self.elimination_rule(items_sorted[i + 1:])
            self.Bins[selected_bin].current_value += item_value
            self.Bins[selected_bin].update(placed_box, selected_EMS, min_vol, min_dim)

            x, y, z = int(selected_EMS[0][0]), int(selected_EMS[0][1]), int(selected_EMS[0][2])
            self.assignments.append({
                'original_item_idx': int(self.BPS[i]),
                'bin_idx': selected_bin,
                'BO': BO,
                'x': x, 'y': y, 'z': z,
            })


# ============================================================================
# solver_338874_2 — main hybrid solver
# ============================================================================
class solver_338874_2(_SolverBase):
    """Hybrid GRASP + GroupingGA + VND solver (bin-centric, efficiency-tuned).

    Pipeline:
        [GRASP]     _run_iteration -> active_bins  (phenotype = chromosome)
        [SEED]      feed GroupingGA.fit(seed_population=...)
        [GroupingGA] grouping crossover + bin-destroy mutation
                     (no encode/decode step — works directly in phenotype space)
        [VND]       _vnd_loop
        [POST]      _post_process_repack_partial + _post_process_downgrade
        [SAVE]      _rebuild_sol_from_bins  (unique sequential ids)
    """

    _BO_TO_ENV = {1: 0, 2: 2, 3: 1, 4: 5, 5: 3, 6: 4}
    _ENV_TO_BO = {0: 1, 1: 3, 2: 2, 3: 5, 4: 6, 5: 4}

    def __init__(self, inst):
        super().__init__(inst)
        self.name = "solver_338874_2"

    # ------------------------------------------------------------------
    # Adaptive BRKGA sizing (more conservative as N grows)
    # ------------------------------------------------------------------
    def _scale_brkga_params(self, n_items, base_individuals, base_generations):
        if n_items < 200:
            individuals = base_individuals
            generations = base_generations
            patient = 4
            stagn = None
        elif n_items < 500:
            individuals = max(60, int(base_individuals * 0.85))
            generations = max(100, int(base_generations * 0.85))
            patient = 4
            stagn = 0.6
        elif n_items < 1000:
            individuals = max(50, int(base_individuals * 0.65))
            generations = max(80, int(base_generations * 0.65))
            patient = 3
            stagn = 0.5
        else:
            # Large N: decoder cost dominates -> aggressive reduction
            individuals = max(36, int(base_individuals * 0.4))
            generations = max(50, int(base_generations * 0.4))
            patient = 3
            stagn = 0.4
        elites = max(6, int(individuals * 0.10))
        mutants = max(8, int(individuals * 0.15))
        if elites + mutants >= individuals:
            mutants = max(2, individuals - elites - 1)
        return individuals, generations, elites, mutants, patient, stagn

    # ------------------------------------------------------------------
    # ILS perturbation: destroy-and-repair
    # ------------------------------------------------------------------
    def _perturb_bins(self, bins, vehicles_dict, n_destroy=None):
        if not bins or len(bins) < 2:
            return self._fast_copy_bins(bins)

        working = self._fast_copy_bins(bins)

        if n_destroy is None:
            avg_util = (sum(b['current_vol'] / max(b['max_vol'], 1) for b in working)
                        / len(working))
            if avg_util >= 0.80:
                n_destroy = 1
            elif avg_util >= 0.60:
                n_destroy = random.randint(1, 2)
            else:
                n_destroy = random.randint(2, 3)

        n_destroy = min(n_destroy, len(working) - 1)
        destroy_set = set(random.sample(range(len(working)), n_destroy))

        extracted = []
        for idx in destroy_set:
            extracted.extend(working[idx]['items'])

        surviving = [b for i, b in enumerate(working) if i not in destroy_set]
        if not extracted:
            return surviving

        random.shuffle(extracted)
        self._try_repack_records(extracted, surviving, vehicles_dict, allow_new=True)
        return surviving

    # ------------------------------------------------------------------
    # Build BRKGA inputs
    # OPT: items_dict avoids repeated .iterrows; bin slots capped per type
    # ------------------------------------------------------------------
    def _build_brkga_inputs(self, items, vehicles, items_dict=None):
        n_items = len(items)
        if items_dict is None:
            items_dict = items.to_dict('index')
        item_ids = list(items.index)

        v = [(int(items_dict[i]['depth']),
              int(items_dict[i]['width']),
              int(items_dict[i]['height'])) for i in item_ids]

        allowed_rotations = []
        item_values = []
        for i in item_ids:
            row = items_dict[i]
            ar_str = str(row['allowedRotations'])
            ar = [self._ENV_TO_BO[int(c)] for c in ar_str if c.isdigit()] or [1, 2, 3, 4, 5, 6]
            allowed_rotations.append(ar)
            item_values.append(float(row.get('value', 0)))

        total_item_vol = sum(b[0] * b[1] * b[2] for b in v)

        # Build pool of unique vehicle types (vehicles passed pre-sorted cheapest-per-vol first).
        # One entry per type; bins are opened dynamically via BTG genes during placement.
        V_pool, type_keys, max_values, grav_strengths, costs_per_vol = [], [], [], [], []
        max_bins = 0
        for vt, vr in vehicles.iterrows():
            v_vol = int(vr['depth']) * int(vr['width']) * int(vr['height'])
            mv  = float(vr.get('maxValue', float('inf')))
            gs  = float(vr.get('gravityStrength', 0)) / 100.0
            cpv = float(vr.get('cost', 1)) / max(v_vol, 1)
            V_pool.append((int(vr['depth']), int(vr['width']), int(vr['height'])))
            type_keys.append(vt)
            max_values.append(mv)
            grav_strengths.append(gs)
            costs_per_vol.append(cpv)
            slots_needed = max(4, int(math.ceil(total_item_vol / max(v_vol, 1) * 1.5)))
            max_bins += min(n_items, slots_needed)
        max_bins = min(n_items, max_bins)

        # Precompute cost-weighted CDF: cheap types (low cpv) get a larger bucket in [0,1].
        n_types = len(V_pool)
        if n_types > 1:
            weights = np.array([1.0 / max(c, 1e-9) for c in costs_per_vol])
            weights = weights / weights.sum()
            type_cdf = list(np.cumsum(weights))
            type_cdf[-1] = 1.0
        else:
            type_cdf = [1.0]

        inputs = {
            'v': v, 'V': V_pool,
            'vehicle_type_keys':  type_keys,
            'max_bins':           max_bins,
            'type_costs_per_vol': costs_per_vol,
            'type_cdf':           type_cdf,
            'allowed_rotations':  allowed_rotations,
            'item_values':        item_values,
            'max_values':         max_values,
            'gravity_strengths':  grav_strengths,
            'max_ems_per_bin':    DEFAULT_MAX_EMS_PER_BIN,
        }
        return inputs

    # ------------------------------------------------------------------
    # Convert tracked placement -> active_bins (uses dict lookups)
    # ------------------------------------------------------------------
    def _placement_to_active_bins(self, placement, inputs, slot_vtypes,
                                  vehicles_dict, items, items_dict=None):
        item_ids = list(items.index)
        if items_dict is None:
            items_dict = items.to_dict('index')

        by_bin = {}
        for asgn in placement.assignments:
            by_bin.setdefault(asgn['bin_idx'], []).append(asgn)

        type_cnt = {}
        active_bins = []
        for b_idx in sorted(by_bin):
            vt      = slot_vtypes[b_idx]
            v_model = vehicles_dict[vt]
            idx     = type_cnt.get(vt, 0)
            type_cnt[vt] = idx + 1

            ab = {
                'type': vt, 'idx': idx,
                'spaces': self._make_spaces(
                    0, 0, 0,
                    int(v_model['depth']), int(v_model['width']), int(v_model['height'])
                ),
                'current_weight': 0.0, 'max_weight':     float(v_model['maxWeight']),
                'current_vol':    0.0, 'max_vol':        float(v_model['volume']),
                'current_value':  0.0, 'max_value_item': float(v_model.get('maxValue', float('inf'))),
                'z_layers': {}, 'items': [], 'last_pruned': 1,
            }

            for asgn in sorted(by_bin[b_idx], key=lambda a: a['z']):
                orig_idx = asgn['original_item_idx']
                i_idx    = item_ids[orig_idx]
                # Keep Series accessible for downstream code that uses .get/.item()
                item_row = items.loc[i_idx]
                dx, dy, dz = _bo_orient(inputs['v'][orig_idx], asgn['BO'])
                r = self._BO_TO_ENV[asgn['BO']]
                x, y, z = asgn['x'], asgn['y'], asgn['z']
                # OPT: weight/value via dict (hot path)
                w   = float(items_dict[i_idx]['weight'])
                val = float(items_dict[i_idx].get('value', 0))

                ab['current_weight'] += w
                ab['current_vol']    += dx * dy * dz
                ab['current_value']  += val
                ab['spaces'] = self._update_ems_spaces(
                    ab['spaces'], x, y, z, dx, dy, dz, ab
                )
                ab['z_layers'].setdefault(int(z + dz), []).append(
                    {'x': x, 'y': y, 'd': dx, 'w': dy}
                )
                ab['items'].append({
                    'i_idx': i_idx, 'item': item_row,
                    'x': x, 'y': y, 'z': z,
                    'dx': dx, 'dy': dy, 'dz': dz,
                    'r': r, 'weight': w,
                })
            active_bins.append(ab)
        return active_bins

    # ------------------------------------------------------------------
    # Encoder: GRASP phenotype -> (sequence, rotations) for PermutationGA
    # ------------------------------------------------------------------
    def _encode_grasp_to_chromosome(self, active_bins, n_items, items, inputs,
                                    idx_to_pos=None):
        if idx_to_pos is None:
            item_ids = list(items.index)
            idx_to_pos = {i_idx: pos for pos, i_idx in enumerate(item_ids)}

        rots = np.random.uniform(0.0, 1.0, size=n_items).astype(np.float64)
        seq = np.empty(n_items, dtype=np.int64)

        placed_records = []
        for b_idx, ab in enumerate(active_bins):
            for rec in ab['items']:
                placed_records.append((b_idx, int(rec['z']), int(rec['y']),
                                       int(rec['x']), rec))
        placed_records.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

        placed_pos_set = set()
        rank = 0
        for _, z, y, x, rec in placed_records:
            i_idx = rec['i_idx']
            pos = idx_to_pos.get(i_idx)
            if pos is None or pos in placed_pos_set:
                continue
            placed_pos_set.add(pos)
            seq[rank] = pos

            r_env = int(rec['r'])
            target_BO = self._ENV_TO_BO.get(r_env)
            allowed = inputs['allowed_rotations'][pos]
            if target_BO is not None and allowed and target_BO in allowed:
                bo_index = allowed.index(target_BO)
                L = len(allowed)
                rots[pos] = (bo_index + 0.5) / L

            rank += 1

        unplaced = [p for p in range(n_items) if p not in placed_pos_set]
        random.shuffle(unplaced)
        for p in unplaced:
            seq[rank] = p
            rank += 1

        # Encode BTG: map each GRASP bin's vehicle type to the midpoint of its CDF bucket
        type_keys = inputs.get('vehicle_type_keys', [])
        type_cdf  = inputs.get('type_cdf', [])
        max_bins  = inputs.get('max_bins', n_items)
        btg = np.random.uniform(0.0, 1.0, size=max_bins)
        if type_keys and type_cdf:
            cdf_low = [0.0] + list(type_cdf[:-1])
            for b_slot, ab in enumerate(active_bins):
                if b_slot >= max_bins:
                    break
                vt = ab['type']
                if vt in type_keys:
                    tidx = type_keys.index(vt)
                    btg[b_slot] = (cdf_low[tidx] + type_cdf[tidx]) / 2.0

        return (seq, rots, btg)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------
    def solve(self, num_generations=200, num_individuals=120, patient=4,
              time_limit=600, n_grasp_iters=15, n_workers=1,
              return_detailed=False, **kwargs):
        start = _time.time()

        items    = self.inst.df_items.copy()
        vehicles = self.inst.df_vehicles.copy()

        items['volume']    = items['width'] * items['depth'] * items['height']
        items['p_i']       = items.apply(self._decide_priority_level, axis=1)
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles['cost_per_vol'] = vehicles['cost'] / vehicles['volume']

        vehicles_for_constructive = vehicles.sort_values(
            by=['cost_per_vol', 'cost'], ascending=[True, True]
        )
        vehicles_for_vnd = vehicles.sort_values(by='cost', ascending=True)
        vehicles_dict    = vehicles_for_vnd.to_dict('index')

        items   = self._heterogeneous_mix(items)
        n_items = len(items)

        # ---- Adaptive sizing ----
        (eff_individuals, eff_generations, _eff_elites, _eff_mutants,
         eff_patient, _eff_stagn) = self._scale_brkga_params(
            n_items, num_individuals, num_generations
        )

        if n_items > 1000:
            n_grasp_iters = min(n_grasp_iters, 6)
            self.max_vnd_iterations = 30
        elif n_items > 500:
            n_grasp_iters = min(n_grasp_iters, 10)
            self.max_vnd_iterations = 30
        else:
            self.max_vnd_iterations = 30

        print(f"[{self.name}] N={n_items}: GRASP iters={n_grasp_iters}, "
              f"GroupGA pop={eff_individuals}, gens={eff_generations}, "
              f"patient={eff_patient}, n_workers={n_workers}")

        # =============================================================
        # PHASE 1 — GRASP seeding
        # =============================================================
        grasp_budget   = time_limit * 0.15
        groupga_budget = time_limit * 0.60

        seed_bins_pool  = []
        delta_pool      = [0.1, 0.3, 0.5, 0.8]
        best_grasp_cost = float('inf')
        best_grasp_bins = None

        print(f"[{self.name}] PHASE 1: GRASP (budget {grasp_budget:.1f}s)")

        for it in range(n_grasp_iters):
            if _time.time() - start > grasp_budget:
                print(f"[{self.name}] GRASP budget exhausted at iter {it}.")
                break

            current_delta = float(np.random.choice(delta_pool))
            try:
                _, _, active_bins_g = self._run_iteration(
                    items, vehicles_for_constructive, current_delta
                )
            except Exception as e:
                print(f"[{self.name}] GRASP iter {it} crashed: {e}")
                continue

            if not active_bins_g:
                continue

            for b in active_bins_g:
                v_model = vehicles_dict[b['type']]
                b['max_vol']        = float(v_model['volume'])
                b['current_vol']    = sum(r['dx'] * r['dy'] * r['dz'] for r in b['items'])
                b['max_value_item'] = float(v_model.get('maxValue', float('inf')))
                b['current_value']  = sum(float(r['item'].get('value', 0)) for r in b['items'])
                b['last_pruned']    = 1

            cost_g = self._cost_from_bins(active_bins_g, vehicles_dict, items)
            if cost_g < best_grasp_cost:
                best_grasp_cost = cost_g
                best_grasp_bins = self._fast_copy_bins(active_bins_g)

            # Phenotype IS the chromosome — no encoding step needed
            seed_bins_pool.append(self._fast_copy_bins(active_bins_g))
            print(f"[{self.name}]   iter {it+1:02d}: cost={cost_g:.2f}")

        print(f"[{self.name}] Seeds={len(seed_bins_pool)}, best GRASP cost={best_grasp_cost:.2f}")

        # =============================================================
        # PHASE 2 — GroupingGA
        # =============================================================
        eff_individuals = max(eff_individuals, len(seed_bins_pool) + 20)

        print(f"[{self.name}] PHASE 2: GroupingGA "
              f"(pop={eff_individuals}, gens={eff_generations}, "
              f"budget {groupga_budget:.1f}s)")

        groupga = GroupingGA(
            solver=self,
            vehicles_dict=vehicles_dict,
            items=items,
            num_generations=eff_generations,
            num_individuals=eff_individuals,
            n_workers=n_workers,
            time_budget=groupga_budget,
        )

        groupga.fit(
            seed_population=seed_bins_pool,
            patient=eff_patient,
            verbose=False,
        )

        # =============================================================
        # PHASE 3 — VND refinement
        # No decode step: GroupingGA works directly in phenotype space.
        # =============================================================
        if groupga.best_bins is None:
            print(f"[{self.name}] GroupingGA empty — fallback to best GRASP.")
            optimized_bins = best_grasp_bins or []
        else:
            active_bins = groupga.best_bins
            init_cost   = self._cost_from_bins(active_bins, vehicles_dict, items)
            print(f"[{self.name}] GroupingGA result: {len(active_bins)} bins, cost {init_cost:.2f}")

            optimized_bins = active_bins
            best_vnd_cost  = init_cost

            vnd_budget = time_limit - (_time.time() - start)
            ils_start  = _time.time()

            for _ils_it in range(self.max_vnd_iterations):
                if _time.time() - ils_start > vnd_budget:
                    print(f"[{self.name}]   ILS time budget exhausted at iter {_ils_it+1}.")
                    break

                if _ils_it == 0:
                    candidate_bins = self._fast_copy_bins(optimized_bins)
                else:
                    candidate_bins = self._perturb_bins(optimized_bins, vehicles_dict)

                sol_iter = self._rebuild_sol_from_bins(candidate_bins)
                candidate_bins, _ = self._vnd_loop(sol_iter, candidate_bins, vehicles_dict, items)
                candidate_cost = self._cost_from_bins(candidate_bins, vehicles_dict, items)

                if candidate_cost == float('inf'):
                    print(f"[{self.name}]   ILS iter {_ils_it+1:02d}: infeasible, skipping.")
                    continue

                if candidate_cost < best_vnd_cost:
                    best_vnd_cost  = candidate_cost
                    optimized_bins = candidate_bins
                elif random.random() < 0.05:
                    optimized_bins = candidate_bins

                print(f"[{self.name}]   ILS iter {_ils_it+1:02d}: "
                      f"cost={candidate_cost:.2f} (best={best_vnd_cost:.2f}) "
                      f"elapsed={_time.time()-ils_start:.1f}s")

            if best_grasp_bins is not None and best_grasp_cost < best_vnd_cost:
                print(f"[{self.name}] GRASP best ({best_grasp_cost:.2f}) beats "
                      f"GroupingGA+VND ({best_vnd_cost:.2f}) — re-running VND on GRASP.")
                sol_g2 = self._rebuild_sol_from_bins(best_grasp_bins)
                optimized_bins, _ = self._vnd_loop(
                    sol_g2, best_grasp_bins, vehicles_dict, items
                )

        # =============================================================
        # PHASE 4 — POST-PROCESSING
        # =============================================================
        if optimized_bins:
            improved, optimized_bins = self._post_process_repack_partial(
                optimized_bins, vehicles_dict, items
            )
            print(f"[{self.name}] repack_partial: improved={improved}")

            improved_d, optimized_bins = self._post_process_downgrade(
                optimized_bins, vehicles_dict
            )
            print(f"[{self.name}] downgrade: improved={improved_d}")

        final_cost = (self._cost_from_bins(optimized_bins, vehicles_dict, items)
                      if optimized_bins else float('inf'))
        print(f"[{self.name}] FINAL: {len(optimized_bins) if optimized_bins else 0} bins, "
              f"cost {final_cost:.2f} in {_time.time() - start:.1f}s")

        # =============================================================
        # PHASE 5 — SAVE (unique sequential ids)
        # =============================================================
        self.sol.clear()
        self.sol.update(self._rebuild_sol_from_bins(optimized_bins) if optimized_bins else {})
        self.active_bins = optimized_bins or []

        if return_detailed:
            return {
                'cost': final_cost,
                'solution_dict': self.sol,
                'active_bins': self.active_bins,
            }