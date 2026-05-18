"""
solver_338874_master.py
=======================
Definitive Deep-Search Hybrid Solver for 3D-BPP (10-minute budget).

Pipeline:
  [0:00 - 0:60] PHASE 1: GRASP Seeding
  [1:00 - 8:00] PHASE 2: Memetic ILS (Ruin -> Recreate -> VND) with Simulated Annealing & Checkpointing
  [8:00 - 9:45] PHASE 3: OR-Tools Matheuristic (CP-SAT Exact Multi-Bin Merging)
  [9:45 - END]  PHASE 4: Post-Processing & Save
"""

import math
import random
import time as _time
import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: ortools non trovato. Matheuristic disabilitata. Esegui 'pip install ortools'.")

from .additional_script import ConstructiveSolver

# ============================================================================
# Module-level constants / helpers
# ============================================================================
INFEASIBLE = 100000

def _copy_bin(ab):
    """Deep-enough copy of one active_bin dict (avoids ghost aliasing)."""
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

class solver_338874_3(ConstructiveSolver):
    _BO_TO_ENV = {1: 0, 2: 2, 3: 1, 4: 5, 5: 3, 6: 4}
    _ENV_TO_BO = {0: 1, 1: 3, 2: 2, 3: 5, 4: 6, 5: 4}

    def __init__(self, inst):
        super().__init__(inst)
        self.name = "solver_338874_3"
        self.max_vnd_iterations = 20 # Ottimizzato per uso in-the-loop

    # ==================================================================
    # 0. CORE GEOMETRICO
    # ==================================================================
    def _rebuild_bin_spaces(self, ab, vehicles_dict):
        v_model = vehicles_dict[ab['type']]
        ab['spaces'] = self._make_spaces(0, 0, 0, int(v_model['depth']), int(v_model['width']), int(v_model['height']))
        ab['z_layers'] = {}
        ab['last_pruned'] = 1
        remaining = sorted(ab['items'], key=lambda r: r['z'])
        for rec in remaining:
            ab['spaces'] = self._update_ems_spaces(ab['spaces'], rec['x'], rec['y'], rec['z'], rec['dx'], rec['dy'], rec['dz'], ab)
            top_z = int(rec['z'] + rec['dz'])
            ab['z_layers'].setdefault(top_z, []).append({'x': rec['x'], 'y': rec['y'], 'd': rec['dx'], 'w': rec['dy']})

    def _bin_gravity_valid(self, ab, v_model):
        min_ratio = float(v_model.get('gravityStrength', 75)) / 100.0
        if min_ratio <= 0: return True
        for rec in ab['items']:
            if int(rec['z']) == 0: continue
            if not self._check_container_gravity_strength(rec['x'], rec['y'], rec['z'], rec['dx'], rec['dy'], ab, v_model):
                return False
        return True

    def _checkpoint_solution(self, bins_to_save):
        """Salva istantaneamente la soluzione migliore parziale nelle strutture di classe."""
        self.sol.clear()
        self.sol.update(self._rebuild_sol_from_bins(bins_to_save) if bins_to_save else {})
        self.active_bins = self._fast_copy_bins(bins_to_save) if bins_to_save else []

    # ==================================================================
    # 1. ORCHESTRATORE PRINCIPALE
    # ==================================================================
    def solve(self, time_limit=600, n_grasp_iters=100, return_detailed=False, **kwargs):
        start = _time.time()
        items = self.inst.df_items.copy()
        vehicles = self.inst.df_vehicles.copy()

        items['volume'] = items['width'] * items['depth'] * items['height']
        items['p_i'] = items.apply(self._decide_priority_level, axis=1)
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles['cost_per_vol'] = vehicles['cost'] / vehicles['volume']

        vehicles_for_constructive = vehicles.sort_values(by=['cost_per_vol', 'cost'], ascending=[True, True])
        vehicles_dict = vehicles.sort_values(by='cost', ascending=True).to_dict('index')
        items = self._heterogeneous_mix(items)
        n_items = len(items)

        grasp_budget = time_limit * 0.10
        ils_budget   = time_limit * 0.70
        math_budget  = time_limit * 0.15

        print(f"[{self.name}] BUDGET: GRASP {grasp_budget:.0f}s | ILS {ils_budget:.0f}s | OR-Tools {math_budget:.0f}s")

        # --- PHASE 1: GRASP ---
        best_grasp_cost = float('inf')
        best_grasp_bins = None
        delta_pool = [0.1, 0.3, 0.5, 0.8]

        for it in range(n_grasp_iters):
            if _time.time() - start > grasp_budget: break
            try:
                current_delta = float(np.random.choice(delta_pool))
                _, _, active_bins_g = self._run_iteration(items, vehicles_for_constructive, current_delta)
                for b in active_bins_g:
                    vm = vehicles_dict[b['type']]
                    b['max_vol'] = float(vm['volume'])
                    b['current_vol'] = sum(r['dx']*r['dy']*r['dz'] for r in b['items'])
                    b['max_value_item'] = float(vm.get('maxValue', float('inf')))
            except: continue

            cost_g = self._cost_from_bins(active_bins_g, vehicles_dict, items)
            if cost_g < best_grasp_cost:
                best_grasp_cost = cost_g
                best_grasp_bins = self._fast_copy_bins(active_bins_g)
                self._checkpoint_solution(best_grasp_bins)

        print(f"[{self.name}] GRASP done. Best cost: {best_grasp_cost:.2f}")

        # --- PHASE 2: MEMETIC ILS ---
        ils_patience = max(25, 75 - (n_items // 15)) 
        optimized_bins = best_grasp_bins
        
        if best_grasp_bins:
            optimized_bins = self._run_memetic_ils(
                best_grasp_bins, vehicles_dict, items, 
                time_budget=ils_budget, t_start_global=start,
                max_no_improve=ils_patience
            )

        # --- PHASE 3: MATHEURISTIC (OR-Tools) ---
        if ORTOOLS_AVAILABLE and optimized_bins:
            optimized_bins = self._run_ortools_matheuristic(
                optimized_bins, vehicles_dict, items,
                time_budget=math_budget
            )

        # --- PHASE 4 & 5: POST-PROCESSING & SAVE ---
        if optimized_bins:
            _, optimized_bins = self._post_process_repack_partial(optimized_bins, vehicles_dict, items)
            _, optimized_bins = self._post_process_downgrade(optimized_bins, vehicles_dict)

        final_cost = self._cost_from_bins(optimized_bins, vehicles_dict, items) if optimized_bins else float('inf')
        print(f"[{self.name}] FINISHED. Final Cost: {final_cost:.2f} | Time: {_time.time()-start:.1f}s")

        self.sol.clear()
        self.sol.update(self._rebuild_sol_from_bins(optimized_bins) if optimized_bins else {})
        self.active_bins = optimized_bins or []

        if return_detailed:
            return {'cost': final_cost, 'solution_dict': self.sol, 'active_bins': self.active_bins}

    # ==================================================================
    # 2. THE MEMETIC ILS (VND IN-THE-LOOP + SIMULATED ANNEALING)
    # ==================================================================
    def _run_memetic_ils(self, start_bins, vehicles_dict, items, time_budget, t_start_global, max_no_improve=40):
        t_start_ils = _time.time()
        
        current_bins = self._fast_copy_bins(start_bins)
        best_bins = self._fast_copy_bins(start_bins)
        current_cost = self._cost_from_bins(current_bins, vehicles_dict, items)
        best_cost = current_cost
        
        T0 = sum(v['cost'] for v in vehicles_dict.values()) / len(vehicles_dict)
        
        it = 0
        no_improve = 0 
        
        while True:
            elapsed_ils = _time.time() - t_start_ils
            if elapsed_ils > time_budget: 
                print(f"[{self.name}] ILS Terminato: Time Budget esaurito.")
                break
            if no_improve >= max_no_improve:
                print(f"[{self.name}] ILS Terminato Anticipatamente (Early Stop): nessuno stallo superato per {max_no_improve} iterazioni.")
                break
            
            T = max(0.1, T0 * (1.0 - (elapsed_ils / time_budget)))
            
            it += 1
            working = self._fast_copy_bins(current_bins)
            orphans = []
            
            ruin_type = random.choice(['eliminate', 'eject', 'strip'])
            
            if ruin_type == 'eliminate' and len(working) > 1:
                n_destroy = random.randint(1, min(3, len(working)-1))
                order = sorted(range(len(working)), key=lambda i: self._bin_utilization(working[i]))
                destroy_set = set(order[:n_destroy])
                for idx in destroy_set: orphans.extend(working[idx]['items'])
                working = [b for i, b in enumerate(working) if i not in destroy_set]
                
            elif ruin_type == 'eject':
                all_pairs = [(b_i, rec) for b_i, b in enumerate(working) for rec in b['items']]
                if not all_pairs: continue
                n_eject = max(1, int(len(all_pairs) * random.uniform(0.10, 0.25)))
                ejected = random.sample(all_pairs, n_eject)
                by_bin = {}
                for b_i, rec in ejected:
                    by_bin.setdefault(b_i, set()).add(rec['i_idx'])
                    orphans.append(rec)
                
                evict_all = []
                for b_i, eject_ids in by_bin.items():
                    ab = working[b_i]
                    ab['items'] = [r for r in ab['items'] if r['i_idx'] not in eject_ids]
                    if ab['items']:
                        self._rebuild_bin_spaces(ab, vehicles_dict)
                        if not self._bin_gravity_valid(ab, vehicles_dict[ab['type']]): evict_all.append(b_i)
                    else:
                        evict_all.append(b_i)
                
                for b_i in evict_all:
                    orphans.extend(working[b_i]['items'])
                    working[b_i]['items'] = []
                working = [b for b in working if b['items']]
                
            else: # strip
                evict_all = []
                for b_i, ab in enumerate(working):
                    sorted_items = sorted(ab['items'], key=lambda r: r['z'], reverse=True)
                    n_strip = max(1, int(len(sorted_items) * 0.20))
                    strip_ids = {r['i_idx'] for r in sorted_items[:n_strip]}
                    orphans.extend([r for r in ab['items'] if r['i_idx'] in strip_ids])
                    ab['items'] = [r for r in ab['items'] if r['i_idx'] not in strip_ids]
                    if ab['items']:
                        self._rebuild_bin_spaces(ab, vehicles_dict)
                        if not self._bin_gravity_valid(ab, vehicles_dict[ab['type']]): evict_all.append(b_i)
                    else:
                        evict_all.append(b_i)
                
                for b_i in evict_all:
                    orphans.extend(working[b_i]['items'])
                    working[b_i]['items'] = []
                working = [b for b in working if b['items']]

            if not orphans: 
                no_improve += 1
                continue

            orphans.sort(key=lambda r: (r['item'].get('p_i', 0), r['dx']*r['dy']*r['dz']), reverse=True)
            ok, remaining = self._try_repack_records(orphans, working, vehicles_dict, allow_new=True)
            if not ok and remaining: self._try_repack_records(remaining, working, vehicles_dict, allow_new=True)

            sol_iter = self._rebuild_sol_from_bins(working)
            working, _ = self._vnd_loop(sol_iter, working, vehicles_dict, items, max_iters=2) 
            
            new_cost = self._cost_from_bins(working, vehicles_dict, items)
            delta = new_cost - current_cost

            accepted = False
            if delta < 0:
                accepted = True
            elif delta == 0:
                sq_new = sum((b['current_vol']/max(b['max_vol'],1))**2 for b in working)
                sq_cur = sum((b['current_vol']/max(b['max_vol'],1))**2 for b in current_bins)
                if len(working) < len(current_bins) or sq_new > sq_cur: accepted = True
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob: accepted = True

            if accepted:
                current_bins = working
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_bins = self._fast_copy_bins(working)
                    no_improve = 0  
                    self._checkpoint_solution(best_bins)
                    print(f"[{self.name}] ILS Iter {it:04d} | NEW BEST: {best_cost:.2f} | T={T:.1f}")
                else:
                    no_improve += 1 
            else:
                no_improve += 1

        return best_bins

    # ==================================================================
    # 3. MATHEURISTIC AVANZATA (CP-SAT MULTI-BIN MERGING)
    # ==================================================================
    def _run_ortools_matheuristic(self, active_bins, vehicles_dict, items, time_budget):
        t_start = _time.time()
        print(f"[{self.name}] Avvio Matheuristic CP-SAT Multi-Bin (Budget {time_budget:.0f}s)...")
        
        n_bins_to_destroy = min(3, len(active_bins))
        if n_bins_to_destroy < 2: return active_bins
        
        sorted_bins = sorted(enumerate(active_bins), key=lambda x: self._bin_utilization(x[1]))
        target_indices = [idx for idx, _ in sorted_bins[:n_bins_to_destroy]]
        bins_to_destroy = [active_bins[idx] for idx in target_indices]
        
        current_cost = sum(vehicles_dict[b['type']]['cost'] for b in bins_to_destroy)
        items_to_pack = [rec for b in bins_to_destroy for rec in b['items']]
        
        total_vol = sum(r['dx'] * r['dy'] * r['dz'] for r in items_to_pack)
        total_weight = sum(r['weight'] for r in items_to_pack)
        
        vehicle_types = sorted(
            vehicles_dict.items(), 
            key=lambda x: float(x[1].get('cost', 0)) / max(float(x[1].get('volume', 1)), 1.0)
        )
        
        best_combo, best_combo_cost = self._select_best_bin_combination(
            total_vol, total_weight, vehicle_types, budget=current_cost - 0.01
        )
        
        if not best_combo:
            print(f"[{self.name}] Nessuna combinazione di veicoli più economica può contenere il volume liquido ({total_vol}). Skip CP-SAT.")
            return active_bins
            
        print(f"[{self.name}] CP-SAT Tenta di repackare {len(items_to_pack)} pacchi (Costo attuale: {current_cost}) in una nuova combo (Costo: {best_combo_cost}).")

        model = cp_model.CpModel()
        
        n_items_target = len(items_to_pack)
        n_bins_target = len(best_combo)
        
        max_D = max(int(v_model['depth']) for _, v_model in best_combo)
        max_W = max(int(v_model['width']) for _, v_model in best_combo)
        max_H = max(int(v_model['height']) for _, v_model in best_combo)
        
        x_s = [model.NewIntVar(0, max_D, f'x_s_{i}') for i in range(n_items_target)]
        y_s = [model.NewIntVar(0, max_W, f'y_s_{i}') for i in range(n_items_target)]
        z_s = [model.NewIntVar(0, max_H, f'z_s_{i}') for i in range(n_items_target)]
        
        x_d = [model.NewIntVar(1, max_D, f'x_d_{i}') for i in range(n_items_target)]
        y_d = [model.NewIntVar(1, max_W, f'y_d_{i}') for i in range(n_items_target)]
        z_d = [model.NewIntVar(1, max_H, f'z_d_{i}') for i in range(n_items_target)]
        
        x_e = [model.NewIntVar(1, max_D, f'x_e_{i}') for i in range(n_items_target)]
        y_e = [model.NewIntVar(1, max_W, f'y_e_{i}') for i in range(n_items_target)]
        z_e = [model.NewIntVar(1, max_H, f'z_e_{i}') for i in range(n_items_target)]
        
        bin_id = [model.NewIntVar(0, n_bins_target - 1, f'bin_id_{i}') for i in range(n_items_target)]
        rots_var = {}
        
        for i, itm_rec in enumerate(items_to_pack):
            model.Add(x_e[i] == x_s[i] + x_d[i])
            model.Add(y_e[i] == y_s[i] + y_d[i])
            model.Add(z_e[i] == z_s[i] + z_d[i])
            
            bin_bools = []
            for b_idx, (v_type, v_model) in enumerate(best_combo):
                b_var = model.NewBoolVar(f'is_in_bin_{i}_{b_idx}')
                bin_bools.append(b_var)
                
                model.Add(bin_id[i] == b_idx).OnlyEnforceIf(b_var)
                model.Add(bin_id[i] != b_idx).OnlyEnforceIf(b_var.Not())
                
                D, W, H = int(v_model['depth']), int(v_model['width']), int(v_model['height'])
                model.Add(x_e[i] <= D).OnlyEnforceIf(b_var)
                model.Add(y_e[i] <= W).OnlyEnforceIf(b_var)
                model.Add(z_e[i] <= H).OnlyEnforceIf(b_var)
                
            model.AddExactlyOne(bin_bools)
            
            item_raw = items.loc[itm_rec['i_idx']]
            allowed_rots = [self._ENV_TO_BO[int(c)] for c in str(item_raw['allowedRotations']) if c.isdigit()]
            if not allowed_rots: allowed_rots = [1,2,3,4,5,6]
            
            rot_bools = []
            for r in allowed_rots:
                r_var = model.NewBoolVar(f'rot_{i}_{r}')
                rot_bools.append(r_var)
                rots_var[(i, r)] = r_var
                
                dx, dy, dz = self._get_rotated_dims(itm_rec['item'], self._BO_TO_ENV[r])
                model.Add(x_d[i] == dx).OnlyEnforceIf(r_var)
                model.Add(y_d[i] == dy).OnlyEnforceIf(r_var)
                model.Add(z_d[i] == dz).OnlyEnforceIf(r_var)
                
            model.AddExactlyOne(rot_bools)

        for i in range(n_items_target):
            for j in range(i + 1, n_items_target):
                same_bin = model.NewBoolVar(f'same_bin_{i}_{j}')
                model.Add(bin_id[i] == bin_id[j]).OnlyEnforceIf(same_bin)
                model.Add(bin_id[i] != bin_id[j]).OnlyEnforceIf(same_bin.Not())
                
                left = model.NewBoolVar(f'left_{i}_{j}')
                right = model.NewBoolVar(f'right_{i}_{j}')
                front = model.NewBoolVar(f'front_{i}_{j}')
                back = model.NewBoolVar(f'back_{i}_{j}')
                under = model.NewBoolVar(f'under_{i}_{j}')
                above = model.NewBoolVar(f'above_{i}_{j}')
                
                model.Add(x_e[i] <= x_s[j]).OnlyEnforceIf(left)
                model.Add(x_e[j] <= x_s[i]).OnlyEnforceIf(right)
                model.Add(y_e[i] <= y_s[j]).OnlyEnforceIf(front)
                model.Add(y_e[j] <= y_s[i]).OnlyEnforceIf(back)
                model.Add(z_e[i] <= z_s[j]).OnlyEnforceIf(under)
                model.Add(z_e[j] <= z_s[i]).OnlyEnforceIf(above)
                
                model.AddBoolOr([same_bin.Not(), left, right, front, back, under, above])

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_budget
        solver.parameters.num_search_workers = 8 
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"[{self.name}] CP-SAT SUCCESSO! I bin originali sostituiti con successo.")
            
            repacked_by_bin = {b_idx: [] for b_idx in range(len(best_combo))}
            for i, itm_rec in enumerate(items_to_pack):
                item_raw = items.loc[itm_rec['i_idx']]
                allowed_rots = [self._ENV_TO_BO[int(c)] for c in str(item_raw['allowedRotations']) if c.isdigit()]
                if not allowed_rots: allowed_rots = [1,2,3,4,5,6]
                
                b_target = solver.Value(bin_id[i])
                r_target = None
                for r in allowed_rots:
                    if solver.Value(rots_var[(i, r)]):
                        r_target = r
                        break
                        
                itm_copy = itm_rec.copy()
                itm_copy['r_target'] = self._BO_TO_ENV[r_target]
                itm_copy['z_target'] = solver.Value(z_s[i])
                itm_copy['y_target'] = solver.Value(y_s[i])
                itm_copy['x_target'] = solver.Value(x_s[i])
                repacked_by_bin[b_target].append(itm_copy)
            
            temp_bins = self._fast_copy_bins(active_bins)
            for idx in sorted(target_indices, reverse=True): temp_bins.pop(idx)
            max_idx = max([b['idx'] for b in temp_bins], default=-1)
            
            all_ok = True
            for b_idx, (v_type, v_model) in enumerate(best_combo):
                repacked_sequence = repacked_by_bin[b_idx]
                if not repacked_sequence: continue 
                
                repacked_sequence.sort(key=lambda rec: (rec['z_target'], rec['y_target'], rec['x_target']))
                D, W, H = int(v_model['depth']), int(v_model['width']), int(v_model['height'])
                new_bin = {
                    'type': v_type, 'idx': max_idx + 1 + b_idx,
                    'spaces': self._make_spaces(0, 0, 0, D, W, H),
                    'current_weight': 0.0, 'max_weight': float(v_model['maxWeight']),
                    'current_vol': 0.0, 'max_vol': float(v_model['volume']),
                    'current_value': 0.0, 'max_value_item': float(v_model.get('maxValue', float('inf'))),
                    'z_layers': {}, 'items': [], 'last_pruned': 1
                }
                
                ok, _ = self._try_repack_records(repacked_sequence, [new_bin], vehicles_dict)
                if ok:
                    temp_bins.append(new_bin)
                else:
                    all_ok = False
                    break
            
            if all_ok:
                print(f"[{self.name}] Sostituzione applicata. Risparmio confermato!")
                self._checkpoint_solution(temp_bins) 
                return temp_bins
            else:
                print(f"[{self.name}] CP-SAT ha fallito la tolleranza di gravità. Rollback.")
                return active_bins
        else:
            print(f"[{self.name}] CP-SAT: Infeasible o Timeout. Non è stato possibile ottimizzare questi veicoli.")
            return active_bins

    # ==================================================================
    # 4. VND CORE & MOVES
    # ==================================================================
    def _vnd_loop(self, sol, active_bins, vehicles_dict, items, max_iters=999):
        neighborhood_moves = [
            self._move_remove_least_occupied,
            self._move_selective_removal,
            self._move_split_bins,
            self._move_compact_bins,
            self._move_split_into_cheaper,
            self._move_advanced_merge,
            self._move_ejection_swap
        ]
        best_sol = self._fast_copy_sol(sol)
        best_bins = self._fast_copy_bins(active_bins)
        best_metrics = self._evaluate_solution(best_bins, items, vehicles_dict)
        neighborhood_index = 0
        iters = 0
        
        while neighborhood_index < len(neighborhood_moves) and iters < max_iters:
            move_fn = neighborhood_moves[neighborhood_index]
            current_sol = self._fast_copy_sol(best_sol)
            current_bins = self._fast_copy_bins(best_bins)
            
            if move_fn(current_sol, current_bins, vehicles_dict, items):
                new_metrics = self._evaluate_solution(current_bins, items, vehicles_dict)
                if self._is_better(new_metrics, best_metrics) or random.random() < 0.1:
                    best_sol = current_sol
                    best_bins = current_bins
                    best_metrics = new_metrics
                    neighborhood_index = 0
                else: 
                    neighborhood_index += 1
            else: 
                neighborhood_index += 1
            iters += 1
            
        return best_bins, best_sol

    def _move_remove_least_occupied(self, sol, active_bins, vehicles_dict, items):
        if len(active_bins) < 2: return False
        idx = min(range(len(active_bins)), key=lambda i: self._bin_utilization(active_bins[i]))
        if self._bin_utilization(active_bins[idx]) >= 0.60: return False
        temp_bins = self._fast_copy_bins(active_bins)
        records = sorted(temp_bins[idx]['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
        others = [b for i, b in enumerate(temp_bins) if i != idx]
        success, _ = self._try_repack_records(records, others, vehicles_dict)
        if success:
            temp_bins.pop(idx)
            active_bins[:] = temp_bins
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False

    def _move_selective_removal(self, sol, active_bins, vehicles_dict, items):
        if not active_bins: return False
        temp_bins = self._fast_copy_bins(active_bins)
        idx = min(range(len(temp_bins)), key=lambda i: self._bin_utilization(temp_bins[i]))
        target = temp_bins[idx]
        n_to_remove = min(40, max(1, int(len(target['items']) * 0.20)))
        to_rem = random.sample(target['items'], n_to_remove)
        to_remove_ids = {r['i_idx'] for r in to_rem}
        target['items'] = [item for item in target['items'] if item['i_idx'] not in to_remove_ids]
        target['current_weight'] -= sum(r['weight'] for r in to_rem)
        target['current_vol']    -= sum(r['dx'] * r['dy'] * r['dz'] for r in to_rem)
        self._rebuild_bin_spaces(target, vehicles_dict)
        if not self._bin_gravity_valid(target, vehicles_dict[target['type']]): return False
        to_rem.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
        others = [b for i, b in enumerate(temp_bins) if i != idx]
        success, _ = self._try_repack_records(to_rem, others, vehicles_dict)
        if success:
            active_bins[:] = temp_bins
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False

    def _move_split_bins(self, sol, active_bins, vehicles_dict, items):
        if not active_bins: return False
        temp_bins = self._fast_copy_bins(active_bins)
        nonempty_indices = [i for i, b in enumerate(temp_bins) if b['items']]
        if not nonempty_indices: return False
        idx = random.choice(nonempty_indices)
        ab = temp_bins[idx]
        axis = random.choice(['depth', 'width', 'height'])
        limit = vehicles_dict[ab['type']][axis] / 2.0
        dim_map = {'depth': 'dx', 'width': 'dy', 'height': 'dz'}
        to_move = [r for r in ab['items'] if r[dim_map[axis]] > limit]
        if not to_move: return False
        to_move_ids = {r['i_idx'] for r in to_move}
        ab['items'] = [item for item in ab['items'] if item['i_idx'] not in to_move_ids]
        ab['current_weight'] -= sum(r['weight'] for r in to_move)
        ab['current_vol']    -= sum(r['dx'] * r['dy'] * r['dz'] for r in to_move)
        self._rebuild_bin_spaces(ab, vehicles_dict)
        if not self._bin_gravity_valid(ab, vehicles_dict[ab['type']]): return False
        to_move.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
        others = [b for b in temp_bins if b['idx'] != ab['idx']]
        success, _ = self._try_repack_records(to_move, others, vehicles_dict)
        if success:
            active_bins[:] = temp_bins
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False

    def _move_compact_bins(self, sol, active_bins, vehicles_dict, items):
        if len(active_bins) < 2: return False
        improved = False
        sorted_indices = sorted(range(len(active_bins)), key=lambda i: self._bin_utilization(active_bins[i]))
        for idx in sorted_indices[:3]:
            if idx >= len(active_bins): continue
            temp_bins = self._fast_copy_bins(active_bins)
            source = temp_bins[idx]
            source_vol = source['current_vol']
            viable_targets = [b for j, b in enumerate(temp_bins) if j != idx and (b['max_vol'] - b['current_vol']) > 0]
            if not viable_targets: continue
            total_free = sum(b['max_vol'] - b['current_vol'] for b in viable_targets)
            if source_vol > total_free: continue
            records = sorted(source['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
            success, _ = self._try_repack_records(records, viable_targets, vehicles_dict)
            if success:
                temp_bins.pop(idx)
                active_bins[:] = temp_bins
                sol.update(self._rebuild_sol_from_bins(active_bins))
                improved = True
                break
        return improved

    def _move_split_into_cheaper(self, sol, active_bins, vehicles_dict, items):
        if not active_bins: return False
        sorted_indices = sorted(range(len(active_bins)), key=lambda i: (-vehicles_dict[active_bins[i]['type']].get('cost', 1000), self._bin_utilization(active_bins[i])))
        for idx in sorted_indices[:3]:
            if idx >= len(active_bins): continue
            temp_bins = self._fast_copy_bins(active_bins)
            ab = temp_bins[idx]
            orig_cost = vehicles_dict[ab['type']].get('cost', 1000)
            cheaper_catalog = {k: v for k, v in vehicles_dict.items() if v.get('cost', 1000) < orig_cost}
            if not cheaper_catalog: continue
            records = sorted(ab['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
            other_existing = [b for j, b in enumerate(temp_bins) if j != idx]
            temp_new_bins = []
            success, remaining = self._try_repack_records(records, other_existing, vehicles_dict)
            if not success and remaining:
                success, remaining = self._try_repack_records(remaining, temp_new_bins, cheaper_catalog, allow_new=True)
            if success and len(remaining) == 0:
                new_cost = sum(vehicles_dict[b['type']].get('cost', 1000) for b in temp_new_bins)
                if new_cost < orig_cost:
                    temp_bins.pop(idx)
                    temp_bins.extend(temp_new_bins)
                    active_bins[:] = temp_bins
                    sol.update(self._rebuild_sol_from_bins(active_bins))
                    return True
        return False

    def _move_advanced_merge(self, sol, active_bins, vehicles_dict, items):
        if len(active_bins) < 2: return False
        sorted_indices = sorted(range(len(active_bins)), key=lambda i: self._bin_utilization(active_bins[i]))
        max_v_vol = max(float(v.get('volume', v['depth'] * v['width'] * v['height'])) for v in vehicles_dict.values())
        max_v_weight = max(float(v['maxWeight']) for v in vehicles_dict.values())

        if len(active_bins) >= 3:
            idx1, idx2, idx3 = sorted_indices[0], sorted_indices[1], sorted_indices[2]
            b1, b2, b3 = active_bins[idx1], active_bins[idx2], active_bins[idx3]
            combined_vol    = b1['current_vol'] + b2['current_vol'] + b3['current_vol']
            combined_weight = b1['current_weight'] + b2['current_weight'] + b3['current_weight']
            if combined_vol <= (max_v_vol * 2) and combined_weight <= (max_v_weight * 2):
                orig_cost = vehicles_dict[b1['type']]['cost'] + vehicles_dict[b2['type']]['cost'] + vehicles_dict[b3['type']]['cost']
                combined_items = b1['items'] + b2['items'] + b3['items']
                combined_items.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
                temp_bins = self._fast_copy_bins(active_bins)
                new_bins = []
                success, _ = self._try_repack_records(combined_items, new_bins, vehicles_dict, allow_new=True)
                if success:
                    new_cost = sum(vehicles_dict[b['type']].get('cost', 1000) for b in new_bins)
                    if new_cost < orig_cost or (new_cost == orig_cost and len(new_bins) < 3):
                        for idx in sorted([idx1, idx2, idx3], reverse=True): temp_bins.pop(idx)
                        temp_bins.extend(new_bins)
                        active_bins[:] = temp_bins
                        sol.update(self._rebuild_sol_from_bins(active_bins))
                        return True

        for i in range(min(3, len(sorted_indices) - 1)):
            for j in range(i + 1, min(4, len(sorted_indices))):
                idx1, idx2 = sorted_indices[i], sorted_indices[j]
                b1, b2 = active_bins[idx1], active_bins[idx2]
                combined_vol    = b1['current_vol'] + b2['current_vol']
                combined_weight = b1['current_weight'] + b2['current_weight']
                if combined_vol > max_v_vol or combined_weight > max_v_weight: continue
                orig_cost = vehicles_dict[b1['type']]['cost'] + vehicles_dict[b2['type']]['cost']
                combined_items = b1['items'] + b2['items']
                combined_items.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']), reverse=True)
                temp_bins = self._fast_copy_bins(active_bins)
                new_bins = []
                success, _ = self._try_repack_records(combined_items, new_bins, vehicles_dict, allow_new=True)
                if success:
                    new_cost = sum(vehicles_dict[b['type']].get('cost', 1000) for b in new_bins)
                    if new_cost < orig_cost or (new_cost == orig_cost and len(new_bins) < 2):
                        for idx in sorted([idx1, idx2], reverse=True): temp_bins.pop(idx)
                        temp_bins.extend(new_bins)
                        active_bins[:] = temp_bins
                        sol.update(self._rebuild_sol_from_bins(active_bins))
                        return True
        return False

    def _move_ejection_swap(self, sol, active_bins, vehicles_dict, items):
        if len(active_bins) < 2: return False
        temp_bins = self._fast_copy_bins(active_bins)
        utils = [(self._bin_utilization(b), i) for i, b in enumerate(temp_bins)]
        utils.sort()
        worst_idx = utils[0][1]
        best_idx  = utils[-1][1]
        worst_bin = temp_bins[worst_idx]
        best_bin  = temp_bins[best_idx]
        if not worst_bin['items'] or not best_bin['items']: return False

        biggest = max(worst_bin['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']))
        ejected = sorted(best_bin['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx'] * r['dy'] * r['dz']))[:min(3, len(best_bin['items']))]

        worst_bin['items'] = [r for r in worst_bin['items'] if r['i_idx'] != biggest['i_idx']]
        worst_bin['current_weight'] -= biggest['weight']
        worst_bin['current_vol']    -= biggest['dx'] * biggest['dy'] * biggest['dz']
        self._rebuild_bin_spaces(worst_bin, vehicles_dict)
        if not self._bin_gravity_valid(worst_bin, vehicles_dict[worst_bin['type']]): return False

        ejected_iidx = {r['i_idx'] for r in ejected}
        for r in ejected:
            best_bin['current_weight'] -= r['weight']
            best_bin['current_vol']    -= r['dx'] * r['dy'] * r['dz']
        best_bin['items'] = [r for r in best_bin['items'] if r['i_idx'] not in ejected_iidx]
        self._rebuild_bin_spaces(best_bin, vehicles_dict)
        if not self._bin_gravity_valid(best_bin, vehicles_dict[best_bin['type']]): return False

        success_in, _ = self._try_repack_records([biggest], [best_bin], vehicles_dict)
        if not success_in: return False

        others = [b for i, b in enumerate(temp_bins) if i != worst_idx]
        success_out, _ = self._try_repack_records(ejected, others, vehicles_dict)
        if success_out:
            active_bins[:] = temp_bins
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False