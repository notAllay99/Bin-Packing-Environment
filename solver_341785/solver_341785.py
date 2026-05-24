import random
import time
import numpy as np
import pandas as pd
from .additional_script import ConstructiveSolver

class solver_341785(ConstructiveSolver):
    """
    Risolutore Metaeuristico Ibrido (GRASP + ILS/VND)
    Versione Definitiva: Zero-Cloning, Transazionale, ID Univoci garantiti.
    """
    def __init__(self, inst):
        super().__init__(inst)
        self.name = "solver_341785"
        self.max_vnd_iterations = 30

    # ==================================================================
    # 0. HELPER GEOMETRICO ASSOLUTO (DISTRUGGE I FANTASMI)
    # ==================================================================
    def _rebuild_bin_spaces(self, ab, vehicles_dict):
        """Quando rimuovi un item, rigenera gli spazi vuoti del bin da zero"""
        v_model = vehicles_dict[ab['type']]
        ab['spaces'] = self._make_spaces(0, 0, 0, int(v_model['depth']), int(v_model['width']), int(v_model['height']))
        ab['z_layers'] = {}
        ab['last_pruned'] = 1
        
        remaining = sorted(ab['items'], key=lambda r: r['z'])
        for rec in remaining:
            ab['spaces'] = self._update_ems_spaces(ab['spaces'], rec['x'], rec['y'], rec['z'], rec['dx'], rec['dy'], rec['dz'], ab)
            top_z = int(rec['z'] + rec['dz'])
            ab['z_layers'].setdefault(top_z, []).append({'x': rec['x'], 'y': rec['y'], 'd': rec['dx'], 'w': rec['dy']})

    # ==================================================================
    # 1. ORCHESTRATORE GRASP
    # ==================================================================
    def solve(self, return_detailed=False, time_limit=600, **kwargs):
        start_time = time.time()
        items = self.inst.df_items.copy()
        
        n_items = len(items)

        # ── Size-based routing ──────────────────────────────────────────
        if n_items <= 200:
            print(f"[{self.name}] Small dataset ({n_items} items) → ILS solver")
            return self._solve_small(items, time_limit, start_time, return_detailed)
        elif n_items <= 700:
            print(f"[{self.name}] Medium dataset ({n_items} items) → GRASP+VND")
        else:
            print(f"[{self.name}] Large dataset ({n_items} items) → GRASP+VND (speed mode)")
        # ────────────────────────────────────────────────────────────────

        if n_items > 1000:
            max_iterations = 40
            no_improve_limit = 3
            self.max_vnd_iterations = 30
            print(f"[{self.name}] Dataset enorme rilevato ({n_items} items). Parametri scalati per velocità.")
        else:
            max_iterations = 30
            no_improve_limit = 5
            self.max_vnd_iterations = 30

        delta_pool       = [0.1, 0.3, 0.5, 0.8]
        delta_probs      = [0.25, 0.25, 0.25, 0.25]
        delta_history    = {d: [] for d in delta_pool}

        vehicles = self.inst.df_vehicles.copy()
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles['cost_per_vol'] = vehicles['cost'] / vehicles['volume']
        vehicles_for_vnd = vehicles.sort_values(by='cost', ascending=True)
        vehicles_dict = vehicles_for_vnd.to_dict('index')

        items['volume'] = items['width'] * items['depth'] * items['height']
        items['p_i']    = items.apply(self._decide_priority_level, axis=1)

        profile = self._profile_dataset(items, vehicles)
        if profile['diamond_ids'] or profile['lead_ids'] or profile['locked_ids']:
            print(f"[{self.name}] Anomalies — diamonds: {len(profile['diamond_ids'])}, leads: {len(profile['lead_ids'])}, locked: {len(profile['locked_ids'])}")

        vehicles_for_constructive = self._sort_vehicles_adaptive(vehicles, profile)
        items = self._order_items_adaptive(items, profile)
        frequency_vector = {i_idx: 0 for i_idx in items.index}

        best_cost = float('inf')
        best_sol  = None
        best_bins = None
        no_improve_count = 0
        # Tracks consecutive non-improving iterations without ever resetting on
        # diversification triggers — used purely for plateau detection below.
        consecutive_no_improve = 0
        # After this many consecutive non-improvements, GRASP is considered stuck
        # and we hand off to ILS (only for datasets ≤800 items).
        PLATEAU_THRESHOLD = 50

        print(f"[{self.name}] Avvio generazione Multi-Start...")

        post_processing_budget = 60 if n_items > 1000 else 30
        grasp_time_limit = time_limit - post_processing_budget
        # Absolute wall: post-processing must never push us past the time limit.
        hard_deadline = start_time + time_limit - 2

        iteration = 0
        while time.time() - start_time < grasp_time_limit:
            current_delta = np.random.choice(delta_pool, p=delta_probs)

            # Every no_improve_limit non-improving iterations, re-sort items by
            # how often they were left unpacked — gives harder items priority.
            if no_improve_count >= no_improve_limit:
                sorted_items = self._diversification_sort(items, frequency_vector)
                no_improve_count = 0
            else:
                sorted_items = items

            _, sol, active_bins = self._run_iteration(sorted_items, vehicles_for_constructive, current_delta)

            for b in active_bins:
                v_model = vehicles_dict[b['type']]
                b['max_vol'] = float(v_model['volume'])
                b['current_vol'] = sum(r['dx'] * r['dy'] * r['dz'] for r in b['items'])
                b['max_value_item'] = float(v_model.get('maxValue', float('inf')))
                b['current_value'] = sum(float(r['item'].get('value', 0)) for r in b['items'])
                b['last_pruned'] = 1

            packed_ids = set(sol['id_item'])
            for i_idx in items.index:
                if i_idx not in packed_ids:
                    frequency_vector[i_idx] += 1

            optimized_bins, optimized_sol = self._vnd_loop(sol, active_bins, vehicles_dict, items)

            cost = self._cost_from_bins(optimized_bins, vehicles_dict, items)
            delta_history[current_delta].append(cost)
            iteration += 1

            if cost < best_cost:
                best_cost = cost
                best_sol  = self._fast_copy_sol(optimized_sol)
                best_bins = self._fast_copy_bins(optimized_bins)
                no_improve_count = 0
                consecutive_no_improve = 0
                print(f"Iter {iteration:02d}: NUOVO RECORD! Costo = {cost:.2f} (delta={current_delta})")
            else:
                no_improve_count += 1
                consecutive_no_improve += 1

            # Plateau detection: if GRASP hasn't improved in PLATEAU_THRESHOLD
            # consecutive iterations, generating fresh random solutions won't help.
            # Break early so we can decide whether to switch to ILS below.
            if consecutive_no_improve >= PLATEAU_THRESHOLD:
                print(f"[{self.name}] Plateau ({PLATEAU_THRESHOLD} iter senza miglioramento).")
                break

            if iteration % 3 == 0:
                avg_costs = [np.mean(delta_history[d]) if delta_history[d] else best_cost for d in delta_pool]
                inv = [1.0 / (c + 1e-5) for c in avg_costs]
                delta_probs = [v / sum(inv) for v in inv]

        # Adaptive ILS switch: use the actual speed of this run to decide whether
        # switching to ILS is worth it. We measure how long each GRASP iteration
        # took on average, then estimate how many ILS iterations we could fit in
        # the remaining budget. ILS iters cost roughly the same as GRASP iters
        # (both do perturbation/construction + VND). If we can fit at least 50,
        # the switch is worthwhile — otherwise too few iterations to explore
        # meaningfully and we go straight to post-processing.
        if best_bins is not None and iteration > 0:
            elapsed       = time.time() - start_time
            avg_iter_time = elapsed / iteration
            remaining     = grasp_time_limit - elapsed
            expected_ils_iters = remaining / avg_iter_time if avg_iter_time > 0 else 0

            if expected_ils_iters >= 50:
                print(f"[{self.name}] avg {avg_iter_time:.1f}s/iter → ~{expected_ils_iters:.0f} ILS iters possible. Switching to ILS...")
                best_cost, best_bins, best_sol = self._medium_ils_phase(
                    best_bins, best_cost, best_sol, vehicles_dict, items,
                    start_time, grasp_time_limit)
            else:
                print(f"[{self.name}] avg {avg_iter_time:.1f}s/iter → only ~{expected_ils_iters:.0f} ILS iters possible. Skipping ILS.")

        if time.time() < hard_deadline:
            print(f"[{self.name}] Avvio post-processing...")

        if best_bins and time.time() < hard_deadline - 5:
            improved, best_bins = self._post_process_repack_partial(best_bins, vehicles_dict, items, deadline=hard_deadline - 3)
            if improved: best_cost = self._cost_from_bins(best_bins, vehicles_dict, items)

        if best_bins and time.time() < hard_deadline - 2:
            print(f"[{self.name}] Avvio downgrade veicoli...")
            improved_cost, best_bins = self._post_process_downgrade(best_bins, vehicles_dict, deadline=hard_deadline - 1)
            if improved_cost: best_cost = self._cost_from_bins(best_bins, vehicles_dict, items)

        self.sol.clear()
        # LA CHIAMATA CHIAVE: rigenera gli ID univoci prima di salvare
        self.sol.update(self._rebuild_sol_from_bins(best_bins) if best_bins else {})
        self.active_bins = best_bins or []

        print(f"[{self.name}] Completato in {time.time()-start_time:.2f}s. Costo Finale: {best_cost:.2f}")
        if return_detailed:
            return {'cost': best_cost, 'solution_dict': best_sol, 'active_bins': best_bins}

    # ==================================================================
    # 2. POST-PROCESSING
    # ==================================================================
    def _select_best_bin_combination(self, total_vol, total_weight, vehicle_types, budget):
        BEAM_WIDTH   = 20   
        MAX_BIN_ADDS = 10   
        initial_state = (0.0, float(total_vol), float(total_weight), [])
        beam = [initial_state]
        best_combo = None
        best_cost  = float(budget) 

        min_cost_per_vol = min(
            float(v_model.get('cost', 0)) / max(float(v_model.get('volume', 1)), 1.0)
            for _, v_model in vehicle_types
        )

        for _ in range(MAX_BIN_ADDS):
            if not beam: break
            next_beam = []

            for state_cost, vol_res, weight_res, combo in beam:
                if vol_res <= 0 and weight_res <= 0:
                    if state_cost < best_cost:
                        best_cost  = state_cost
                        best_combo = combo
                    continue

                for v_type, v_model in vehicle_types:
                    v_vol    = float(v_model.get('volume',    0))
                    v_weight = float(v_model.get('maxWeight', 0))
                    v_cost   = float(v_model.get('cost',      0))

                    if v_vol <= 0 or v_weight <= 0: continue

                    new_cost       = state_cost + v_cost
                    new_vol_res    = vol_res    - v_vol
                    new_weight_res = weight_res - v_weight
                    new_combo      = combo + [(v_type, v_model)]

                    if new_cost >= best_cost: continue

                    if new_vol_res <= 0 and new_weight_res <= 0:
                        if new_cost < best_cost:
                            best_cost  = new_cost
                            best_combo = new_combo
                        continue

                    next_beam.append((new_cost, new_vol_res, new_weight_res, new_combo))

            if not next_beam: break

            def state_priority(s):
                cost, v_r, w_r, _ = s
                return cost + max(v_r, 0) * min_cost_per_vol

            next_beam.sort(key=state_priority)
            beam = next_beam[:BEAM_WIDTH]

        return best_combo, best_cost

    def _post_process_repack_partial(self, active_bins, vehicles_dict, items, deadline=None):
        THRESHOLD  = 0.80
        healthy_bins   = [b for b in active_bins if self._bin_utilization(b) >= THRESHOLD]
        candidate_bins = [b for b in active_bins if self._bin_utilization(b) <  THRESHOLD]

        if not candidate_bins: return False, active_bins

        candidate_cost = sum(vehicles_dict[b['type']].get('cost', 0) for b in candidate_bins)
        all_records    = [item for b in candidate_bins for item in b['items']]
        total_vol      = sum(float(r['dx']*r['dy']*r['dz']) for r in all_records)
        total_weight   = sum(float(r['weight']) for r in all_records)

        vehicle_types = sorted(vehicles_dict.items(), key=lambda x: float(x[1].get('cost', 0)) / max(float(x[1].get('volume', 1)), 1.0))
        best_combo, best_combo_cost = self._select_best_bin_combination(total_vol, total_weight, vehicle_types, candidate_cost)

        if not best_combo: return False, active_bins

        N_ATTEMPTS = 15
        max_existing_idx = max((b['idx'] for b in active_bins), default=-1)
        best_new_bins = None
        best_new_cost = candidate_cost

        for attempt in range(N_ATTEMPTS):
            if deadline is not None and time.time() > deadline:
                break
            if attempt == 0: shuffled = sorted(all_records, key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
            elif attempt == 1: shuffled = sorted(all_records, key=lambda r: r['dx']*r['dy']*r['dz'], reverse=True)
            elif attempt == 2: shuffled = sorted(all_records, key=lambda r: float(r['weight']), reverse=True)
            elif attempt == 3: shuffled = sorted(all_records, key=lambda r: r['dx']*r['dy'], reverse=True)
            else:
                shuffled = list(all_records)
                random.shuffle(shuffled)

            temp_new_bins = []
            for offset, (v_type, v_model) in enumerate(best_combo):
                temp_new_bins.append({
                    'type': v_type, 'idx': max_existing_idx + 1 + offset,
                    'spaces': self._make_spaces(0, 0, 0, int(v_model['depth']), int(v_model['width']), int(v_model['height'])),
                    'current_weight': 0.0, 'max_weight': float(v_model['maxWeight']),
                    'current_vol': 0.0, 'max_vol': float(v_model['volume']),
                    'current_value': 0.0, 'max_value_item': float(v_model.get('maxValue', float('inf'))),
                    'z_layers': {}, 'items': [], 'last_pruned': 1
                })

            success, remaining = self._try_repack_records(shuffled, temp_new_bins, vehicles_dict, allow_new=False)
            
            if not success:
                success, remaining = self._try_repack_records(remaining, temp_new_bins, vehicles_dict, allow_new=True)

            if not success: continue

            for b in temp_new_bins:
                b['current_vol'] = sum(r['dx']*r['dy']*r['dz'] for r in b['items'])
                b['current_value'] = sum(float(r['item'].get('value', 0)) for r in b['items'])

            temp_sol = self._rebuild_sol_from_bins(temp_new_bins)
            optimized_bins, _ = self._vnd_loop(temp_sol, temp_new_bins, vehicles_dict, items)
            new_cost = sum(vehicles_dict[b['type']].get('cost', 0) for b in optimized_bins)

            if new_cost < best_new_cost:
                best_new_cost = new_cost
                best_new_bins = self._fast_copy_bins(optimized_bins)

        if best_new_bins is not None:
            active_bins.clear()
            active_bins.extend(healthy_bins)
            active_bins.extend(best_new_bins)
            return True, active_bins

        return False, active_bins

    def _post_process_downgrade(self, active_bins, vehicles_dict, deadline=None):
        any_improvement = False
        final_bins = []
        cheapest_vehicles = sorted(vehicles_dict.items(), key=lambda x: x[1]['cost'])

        for ab in active_bins:
            if deadline is not None and time.time() > deadline:
                final_bins.append(ab)
                continue
            current_cost = vehicles_dict[ab['type']]['cost']
            items_to_repack = sorted(ab['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
            total_w = ab['current_weight']
            total_v = ab['current_vol']
            total_val = sum(float(r['item'].get('value', 0)) for r in ab['items'])
            found_better = False

            for v_type, v_model in cheapest_vehicles:
                if v_model['cost'] >= current_cost: break
                max_val = float(v_model.get('maxValue', float('inf')))
                if (total_w <= v_model['maxWeight'] and
                        total_v <= v_model['volume'] and
                        total_val <= max_val):
                    temp_bin = [{
                        'type': v_type, 'idx': ab['idx'],
                        'spaces': self._make_spaces(0, 0, 0, int(v_model['depth']), int(v_model['width']), int(v_model['height'])),
                        'current_weight': 0, 'max_weight': float(v_model['maxWeight']),
                        'current_vol': 0.0, 'max_vol': float(v_model['volume']),
                        'current_value': 0.0, 'max_value_item': max_val,
                        'z_layers': {}, 'items': [], 'last_pruned': 1
                    }]
                    
                    success, _ = self._try_repack_records(items_to_repack, temp_bin, vehicles_dict)
                    if success:
                        final_bins.append(temp_bin[0])
                        found_better = True
                        any_improvement = True
                        break
            
            if not found_better:
                final_bins.append(ab)
                
        if any_improvement:
            active_bins[:] = final_bins
        return any_improvement, active_bins

    # ==================================================================
    # 3. HELPER E METRICHE
    # ==================================================================

    def _profile_dataset(self, items, vehicles):
        veh = vehicles.copy()
        if 'volume' not in veh.columns:
            veh['volume'] = veh['width'] * veh['depth'] * veh['height']
        eps = 1e-9
        item_v_density = items['value'] / items['volume'].clip(lower=eps)
        item_w_density = items['weight'] / items['volume'].clip(lower=eps)
        finite_mv = vehicles['maxValue'].dropna()
        if len(finite_mv) > 0:
            best_vd = float((finite_mv.values / veh.loc[finite_mv.index, 'volume'].clip(lower=eps).values).max())
        else:
            best_vd = float('inf')
        best_wd = float((vehicles['maxWeight'] / veh['volume'].clip(lower=eps)).max())
        diamond_ids = set(items.index[item_v_density > best_vd * 2]) if best_vd < float('inf') else set()
        lead_ids    = set(items.index[item_w_density > best_wd * 2])
        locked_ids  = set(items.index[items['allowedRotations'].str.len() == 1])
        n = max(len(items), 1)
        return {
            'diamond_ids':       diamond_ids,
            'lead_ids':          lead_ids,
            'locked_ids':        locked_ids,
            'diamond_severity':  len(diamond_ids) / n,
            'lead_severity':     len(lead_ids)    / n,
            'lock_severity':     len(locked_ids)  / n,
        }

    def _sort_vehicles_adaptive(self, vehicles, profile):
        ds = profile['diamond_severity']
        ls = profile['lead_severity']
        if ds < 0.05 and ls < 0.05:
            return vehicles.sort_values(by=['cost_per_vol', 'cost'], ascending=[True, True])
        eps = 1e-9
        veh = vehicles.copy()
        if 'cost_per_vol' not in veh.columns:
            veh['cost_per_vol'] = veh['cost'] / veh['volume'].clip(lower=eps)
        veh['_maxval_eff'] = veh['maxValue'].fillna(veh['cost'] * 1e6)
        veh['_val_score']  = veh['cost'] / veh['_maxval_eff'].clip(lower=eps)
        veh['_wt_score']   = veh['cost'] / veh['maxWeight'].clip(lower=eps)
        normal_s = max(0.0, 1.0 - ds - ls)
        veh['_blended'] = ds * veh['_val_score'] + ls * veh['_wt_score'] + normal_s * veh['cost_per_vol']
        return veh.sort_values(by=['_blended', 'cost'], ascending=[True, True])

    def _order_items_adaptive(self, items, profile):
        diamond_ids = profile['diamond_ids']
        lead_ids    = profile['lead_ids']
        locked_ids  = profile['locked_ids']
        anomaly_ids = diamond_ids | lead_ids | locked_ids
        if not anomaly_ids:
            return self._heterogeneous_mix(items)
        def _rank(i):
            if i in diamond_ids and i in lead_ids: return 4
            if i in diamond_ids: return 3
            if i in lead_ids:    return 2
            if i in locked_ids:  return 1
            return 0
        df = items.copy()
        df['_anom'] = [_rank(i) for i in df.index]
        anom_mask = df.index.isin(anomaly_ids)
        anom_items   = df[anom_mask].sort_values(by=['_anom', 'p_i', 'volume'], ascending=[False, False, False])
        normal_items = self._heterogeneous_mix(df[~anom_mask])
        return pd.concat([anom_items, normal_items]).drop(columns=['_anom'], errors='ignore')

    def _fast_copy_sol(self, sol): return {k: list(v) for k, v in sol.items()}
    def _fast_copy_bins(self, active_bins):
        return [{
            'type': b['type'], 'idx': b['idx'], 'spaces': b['spaces'].copy(),
            'current_weight': b['current_weight'], 'max_weight': b['max_weight'],
            'current_vol': b['current_vol'], 'max_vol': b['max_vol'],
            'current_value': b.get('current_value', 0.0), 'max_value_item': b.get('max_value_item', float('inf')),
            'z_layers': {z: list(layer) for z, layer in b.get('z_layers', {}).items()},
            'items': [rec.copy() for rec in b['items']], 'last_pruned': b.get('last_pruned', 1)
        } for b in active_bins]

    def _diversification_sort(self, items, frequency_vector):
        def sort_key(row):
            return (-frequency_vector.get(row.name, 0), -row.get('p_i', 1), -row['volume']) 
        return items.copy().iloc[sorted(range(len(items)), key=lambda k: sort_key(items.iloc[k]))]

    def _heterogeneous_mix(self, items):
        if 'p_i' not in items.columns: items['p_i'] = items.apply(self._decide_priority_level, axis=1)
        items = items.sort_values(by=['p_i', 'volume'], ascending=[False, False])
        mixed_dfs = []
        for p_val, group in items.groupby('p_i', sort=False):
            if len(group) < 10:
                mixed_dfs.append(group)
                continue
            n = len(group)
            large, medium, small = group.iloc[:n//3], group.iloc[n//3:2*n//3], group.iloc[2*n//3:]
            mixed_indices = []
            l_idx, m_idx, s_idx = 0, 0, 0
            while l_idx < len(large) or m_idx < len(medium) or s_idx < len(small):
                for _ in range(2):
                    if l_idx < len(large): mixed_indices.append(large.index[l_idx]); l_idx += 1
                if m_idx < len(medium): mixed_indices.append(medium.index[m_idx]); m_idx += 1
                for _ in range(2):
                    if s_idx < len(small): mixed_indices.append(small.index[s_idx]); s_idx += 1
            mixed_dfs.append(group.loc[mixed_indices])
        return pd.concat(mixed_dfs) if mixed_dfs else items

    def _evaluate_solution(self, active_bins, items, vehicles_dict):
        if not active_bins: return {'total_bins': 0, 'packed_items': 0, 'packed_priority': 0, 'squared_util': 0.0, 'total_cost': 0.0}
        packed_items_count = sum(len(b['items']) for b in active_bins)
        packed_priority_score = sum(rec['item'].get('p_i', 1) for b in active_bins for rec in b['items'])
        squared_utilization = sum((b['current_vol'] / b['max_vol'])**2 for b in active_bins)
        total_cost = sum(vehicles_dict[b['type']].get('cost', 1000) for b in active_bins)
        
        return {
            'total_bins': len(active_bins), 
            'packed_items': packed_items_count, 
            'packed_priority': packed_priority_score, 
            'squared_util': squared_utilization,
            'total_cost': total_cost
        }

    def _is_better(self, new_m, old_m):
        if new_m['packed_priority'] > old_m['packed_priority']: return True
        if new_m['packed_priority'] < old_m['packed_priority']: return False
        if new_m['packed_items'] > old_m['packed_items']: return True
        if new_m['packed_items'] < old_m['packed_items']: return False
        if new_m['total_cost'] < old_m['total_cost']: return True
        if new_m['total_cost'] > old_m['total_cost']: return False
        if new_m['total_bins'] < old_m['total_bins']: return True
        if new_m['total_bins'] > old_m['total_bins']: return False
        return new_m['squared_util'] > old_m['squared_util']

    def _cost_from_bins(self, active_bins, vehicles_dict, items):
        if sum(len(b['items']) for b in active_bins) < len(items): return float('inf')
        return sum(vehicles_dict[b['type']].get('cost', 1000) for b in active_bins)

    def _bin_utilization(self, ab): return ab['current_vol'] / max(ab['max_vol'], 1)

    # ==================================================================
    # IL FIX SUPREMO (REBUILD SOL)
    # ==================================================================
    def _rebuild_sol_from_bins(self, active_bins):
        sol = {k: [] for k in ['type_vehicle', 'idx_vehicle', 'id_item', 'x_origin', 'y_origin', 'z_origin', 'orient']}
        # Iterando con enumerate, forziamo un ID perfettamente incrementale e univoco
        for real_idx, ab in enumerate(active_bins):
            ab['idx'] = real_idx  # Allinea l'ID interno per sicurezza
            for rec in ab['items']:
                sol['type_vehicle'].append(ab['type'])
                sol['idx_vehicle'].append(real_idx)  # <-- Questo salva la vita al checker
                sol['id_item'].append(rec['i_idx'])
                sol['x_origin'].append(rec['x'])
                sol['y_origin'].append(rec['y'])
                sol['z_origin'].append(rec['z'])
                sol['orient'].append(rec['r'])
        return sol

    # ==================================================================
    # 4. CORE GEOMETRICO E REPACKING
    # ==================================================================
    def _bin_gravity_valid(self, ab, v_model):
        """Returns False if any remaining item in the bin is no longer gravity-supported.
        Must be called after _rebuild_bin_spaces to get accurate z_layers."""
        min_ratio = float(v_model.get('gravityStrength', 75)) / 100.0
        if min_ratio <= 0:
            return True
        for rec in ab['items']:
            if int(rec['z']) == 0:
                continue
            if not self._check_container_gravity_strength(
                rec['x'], rec['y'], rec['z'], rec['dx'], rec['dy'], ab, v_model
            ):
                return False
        return True


    def _evaluate_moves_for_rotation(self, item, r, vehicles_input, active_bins, item_weight, item_value):
        candidates = []
        dx, dy, dz = self._get_rotated_dims(item, r)
        item_vol = dx * dy * dz
        is_dict = isinstance(vehicles_input, dict)

        max_open_bins_to_explore = 3 if not is_dict else float('inf')
        max_new_vehicles_to_explore = 2 if not is_dict else float('inf')

        sorted_bins = sorted(enumerate(active_bins), key=lambda x: x[1].get('max_vol', float('inf')) - x[1].get('current_vol', 0), reverse=True)
        valid_bins_found = 0

        for b_idx, active_bin in sorted_bins:
            if active_bin['current_weight'] + item_weight > active_bin['max_weight']: continue
            if active_bin.get('current_vol', 0) + item_vol > active_bin.get('max_vol', float('inf')): continue
            if active_bin.get('current_value', 0) + item_value > active_bin.get('max_value_item', float('inf')): continue

            spaces = active_bin['spaces']
            if len(spaces) == 0: continue

            fits_mask = (spaces[:, 3] >= dx) & (spaces[:, 4] >= dy) & (spaces[:, 5] >= dz)
            fit_indices = np.where(fits_mask)[0]
            if len(fit_indices) == 0: continue

            v_model = vehicles_input[active_bin['type']] if is_dict else vehicles_input.loc[active_bin['type']]
            best_s_idx = None
            best_score = float('inf')
            
            for s_idx in fit_indices:
                s_row = spaces[s_idx]
                score = int(s_row[0])**2 + int(s_row[1])**2 + int(s_row[2])**2
                if score < best_score:
                    if self._check_container_gravity_strength(s_row[0], s_row[1], s_row[2], dx, dy, active_bin, v_model):
                        best_score = score
                        best_s_idx = s_idx

            if best_s_idx is not None:
                s_row = spaces[best_s_idx]
                wasted_vol = (int(s_row[3]) * int(s_row[4]) * int(s_row[5])) - item_vol
                candidates.append({
                    'is_new': False, 'b_idx': b_idx, 'score': (0, best_score, wasted_vol),
                    'dx': dx, 'dy': dy, 'dz': dz, 'x': int(s_row[0]), 'y': int(s_row[1]), 'z': int(s_row[2]), 'r': r
                })
                valid_bins_found += 1
                if valid_bins_found >= max_open_bins_to_explore: break

        seen_types = set()
        iterator = vehicles_input.items() if is_dict else vehicles_input.iterrows()
        valid_new_vehicles_found = 0
        
        for v_type, v_model in iterator:
            if v_type in seen_types: continue
            if (float(v_model['maxWeight']) >= item_weight and
                    float(v_model.get('maxValue', float('inf'))) >= item_value and
                    int(v_model['depth']) >= dx and
                    int(v_model['width']) >= dy and int(v_model['height']) >= dz):
                
                cost_penalty = float(v_model.get('cost', 1000))
                vehicle_vol = float(v_model.get('volume', int(v_model['depth']) * int(v_model['width']) * int(v_model['height'])))
                waste_ratio = 1.0 - (item_vol / max(vehicle_vol, 1))
                candidates.append({
                    'is_new': True, 'v_type': v_type, 'v_model': v_model, 'r': r,
                    'score': (1, cost_penalty, waste_ratio), 'dx': dx, 'dy': dy, 'dz': dz,
                    'lx': int(v_model['depth']), 'ly': int(v_model['width']), 'lz': int(v_model['height'])
                })
                seen_types.add(v_type)
                valid_new_vehicles_found += 1
                if valid_new_vehicles_found >= max_new_vehicles_to_explore: break
                
        return candidates

    def _try_repack_records(self, records, target_bins, vehicles_dict, allow_new=False):
        unpacked = []
        max_idx = max([b['idx'] for b in target_bins], default=-1)

        for rec in records:
            item, item_weight = rec['item'], float(rec['weight'])
            allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]
            all_candidates = []
            
            for r in allowed_rots:
                cands = self._evaluate_moves_for_rotation(item, r, vehicles_dict, target_bins, item_weight, float(item.get('value', 0)))
                if not allow_new: cands = [c for c in cands if not c['is_new']]
                all_candidates.extend(cands)
            
            if all_candidates:
                all_candidates.sort(key=lambda x: x['score'])
                if allow_new and all_candidates[0]['is_new']:
                    top_new = [c for c in all_candidates if c['is_new']][:3]
                    best_move = random.choice(top_new)
                else:
                    best_move = all_candidates[0]
                
                dx, dy, dz = best_move['dx'], best_move['dy'], best_move['dz']
                item_vol = dx * dy * dz
                
                if best_move['is_new']:
                    max_idx += 1
                    ab = {
                        'type': best_move['v_type'], 'idx': max_idx,
                        'spaces': self._make_spaces(0, 0, 0, best_move['lx'], best_move['ly'], best_move['lz']),
                        'current_weight': 0, 'max_weight': float(best_move['v_model']['maxWeight']),
                        'current_vol': 0.0, 'max_vol': float(best_move['v_model']['volume']),
                        'current_value': 0.0, 'max_value_item': float(best_move['v_model'].get('maxValue', float('inf'))),
                        'z_layers': {}, 'items': [], 'last_pruned': 1
                    }
                    target_bins.append(ab)
                    b_idx, x, y, z = len(target_bins) - 1, 0, 0, 0
                else:
                    b_idx = best_move['b_idx']
                    ab = target_bins[b_idx]
                    x, y, z = best_move['x'], best_move['y'], best_move['z']
                
                ab['current_weight'] += item_weight
                ab['current_vol'] += item_vol
                ab['current_value'] = ab.get('current_value', 0) + float(item.get('value', 0))
                ab['spaces'] = self._update_ems_spaces(ab['spaces'], x, y, z, dx, dy, dz, ab)
                ab['z_layers'].setdefault(int(z + dz), []).append({'x': x, 'y': y, 'd': dx, 'w': dy})
                ab['items'].append({**rec, 'x': x, 'y': y, 'z': z, 'dx': dx, 'dy': dy, 'dz': dz, 'r': best_move['r']})
            else:
                unpacked.append(rec)
        return len(unpacked) == 0, unpacked

    # ==================================================================
    # 5. ILS / VND FASE DI MIGLIORAMENTO 
    # ==================================================================
    def _vnd_loop(self, sol, active_bins, vehicles_dict, items):
        neighborhood_moves = [
            self._move_remove_least_occupied,
            self._move_selective_removal,
            self._move_split_bins,
            self._move_compact_bins,
            self._move_split_into_cheaper,
            self._move_advanced_merge,
            self._move_ejection_swap
        ]

        best_sol     = self._fast_copy_sol(sol)
        best_bins    = self._fast_copy_bins(active_bins)
        best_metrics = self._evaluate_solution(best_bins, items, vehicles_dict)

        neighborhood_index = 0

        while neighborhood_index < len(neighborhood_moves):
            move_fn = neighborhood_moves[neighborhood_index]
            current_sol  = self._fast_copy_sol(best_sol)
            current_bins = self._fast_copy_bins(best_bins)

            move_successful = move_fn(current_sol, current_bins, vehicles_dict, items)

            if move_successful:
                new_metrics = self._evaluate_solution(current_bins, items, vehicles_dict)
                if self._is_better(new_metrics, best_metrics) or random.random() < 0.1:
                    best_sol     = current_sol
                    best_bins    = current_bins
                    best_metrics = new_metrics
                    neighborhood_index = 0  
                else:
                    neighborhood_index += 1
            else:
                neighborhood_index += 1

        return best_bins, best_sol

    # ==================================================================
    # 6. LIBRERIA DELLE MOSSE VND TRANSAZIONALI
    # ==================================================================
    def _move_remove_least_occupied(self, sol, active_bins, vehicles_dict, items):
        if len(active_bins) < 2: return False
        
        idx = min(range(len(active_bins)), key=lambda i: self._bin_utilization(active_bins[i]))
        if self._bin_utilization(active_bins[idx]) >= 0.60: return False
        
        temp_bins = self._fast_copy_bins(active_bins)
        records = sorted(temp_bins[idx]['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
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
        target['current_vol'] -= sum(r['dx'] * r['dy'] * r['dz'] for r in to_rem)
        
        self._rebuild_bin_spaces(target, vehicles_dict)

        if not self._bin_gravity_valid(target, vehicles_dict[target['type']]):
            return False

        to_rem.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)

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
        ab['current_vol'] -= sum(r['dx']*r['dy']*r['dz'] for r in to_move)
        
        self._rebuild_bin_spaces(ab, vehicles_dict)

        if not self._bin_gravity_valid(ab, vehicles_dict[ab['type']]):
            return False

        to_move.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
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
            
            records = sorted(source['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
            
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
        
        sorted_indices = sorted(range(len(active_bins)), 
            key=lambda i: (-vehicles_dict[active_bins[i]['type']].get('cost', 1000), self._bin_utilization(active_bins[i])))
                            
        for idx in sorted_indices[:3]:
            if idx >= len(active_bins): continue
            
            temp_bins = self._fast_copy_bins(active_bins)
            ab = temp_bins[idx]
            orig_cost = vehicles_dict[ab['type']].get('cost', 1000)
            
            cheaper_catalog = {k: v for k, v in vehicles_dict.items() if v.get('cost', 1000) < orig_cost}
            if not cheaper_catalog: continue
            
            records = sorted(ab['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
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
        max_v_vol = max(float(v.get('volume', v['depth']*v['width']*v['height'])) for v in vehicles_dict.values())
        max_v_weight = max(float(v['maxWeight']) for v in vehicles_dict.values())

        if len(active_bins) >= 3:
            idx1, idx2, idx3 = sorted_indices[0], sorted_indices[1], sorted_indices[2]
            b1, b2, b3 = active_bins[idx1], active_bins[idx2], active_bins[idx3]
            
            combined_vol = b1['current_vol'] + b2['current_vol'] + b3['current_vol']
            combined_weight = b1['current_weight'] + b2['current_weight'] + b3['current_weight']
            
            if combined_vol <= (max_v_vol * 2) and combined_weight <= (max_v_weight * 2):
                orig_cost = vehicles_dict[b1['type']]['cost'] + vehicles_dict[b2['type']]['cost'] + vehicles_dict[b3['type']]['cost']
                combined_items = b1['items'] + b2['items'] + b3['items']
                combined_items.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
                
                temp_bins = self._fast_copy_bins(active_bins)
                new_bins = []
                success, _ = self._try_repack_records(combined_items, new_bins, vehicles_dict, allow_new=True)
                
                if success:
                    new_cost = sum(vehicles_dict[b['type']].get('cost', 1000) for b in new_bins)
                    if new_cost < orig_cost or (new_cost == orig_cost and len(new_bins) < 3):
                        for idx in sorted([idx1, idx2, idx3], reverse=True):
                            temp_bins.pop(idx)
                        temp_bins.extend(new_bins)
                        active_bins[:] = temp_bins
                        sol.update(self._rebuild_sol_from_bins(active_bins))
                        return True

        for i in range(min(3, len(sorted_indices) - 1)):
            for j in range(i + 1, min(4, len(sorted_indices))):
                idx1, idx2 = sorted_indices[i], sorted_indices[j]
                b1, b2 = active_bins[idx1], active_bins[idx2]
                
                combined_vol = b1['current_vol'] + b2['current_vol']
                combined_weight = b1['current_weight'] + b2['current_weight']
                
                if combined_vol > max_v_vol or combined_weight > max_v_weight: continue
                    
                orig_cost = vehicles_dict[b1['type']]['cost'] + vehicles_dict[b2['type']]['cost']
                combined_items = b1['items'] + b2['items']
                combined_items.sort(key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']), reverse=True)
                
                temp_bins = self._fast_copy_bins(active_bins)
                new_bins = []
                success, _ = self._try_repack_records(combined_items, new_bins, vehicles_dict, allow_new=True)
                
                if success:
                    new_cost = sum(vehicles_dict[b['type']].get('cost', 1000) for b in new_bins)
                    if new_cost < orig_cost or (new_cost == orig_cost and len(new_bins) < 2):
                        for idx in sorted([idx1, idx2], reverse=True):
                            temp_bins.pop(idx)
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

        biggest = max(worst_bin['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']))
        ejected = sorted(best_bin['items'], key=lambda r: (r['item'].get('p_i', 1), r['dx']*r['dy']*r['dz']))[:min(3, len(best_bin['items']))]

        biggest_iidx = biggest['i_idx']
        worst_bin['items'] = [r for r in worst_bin['items'] if r['i_idx'] != biggest_iidx]
        worst_bin['current_weight'] -= biggest['weight']
        worst_bin['current_vol']    -= biggest['dx']*biggest['dy']*biggest['dz']
        self._rebuild_bin_spaces(worst_bin, vehicles_dict)

        if not self._bin_gravity_valid(worst_bin, vehicles_dict[worst_bin['type']]):
            return False

        ejected_iidx = {r['i_idx'] for r in ejected}
        for r in ejected:
            best_bin['current_weight'] -= r['weight']
            best_bin['current_vol']    -= r['dx']*r['dy']*r['dz']
        best_bin['items'] = [r for r in best_bin['items'] if r['i_idx'] not in ejected_iidx]
        self._rebuild_bin_spaces(best_bin, vehicles_dict)

        if not self._bin_gravity_valid(best_bin, vehicles_dict[best_bin['type']]):
            return False

        success_in, _ = self._try_repack_records([biggest], [best_bin], vehicles_dict)
        if not success_in: return False

        others = [b for i, b in enumerate(temp_bins) if i != worst_idx]
        success_out, _ = self._try_repack_records(ejected, others, vehicles_dict)
        
        if success_out:
            active_bins[:] = temp_bins
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True

        return False

    # ==================================================================
    # 7. MEDIUM-DATASET ILS PHASE (post-GRASP plateau)
    # ==================================================================

    def _medium_ils_phase(self, best_bins, best_cost, best_sol, vehicles_dict, items, start_time, time_limit):
        current_bins = self._fast_copy_bins(best_bins)
        current_cost = best_cost
        no_improve   = 0
        ils_iter     = 0
        # After this many consecutive non-improvements, snap back to the global
        # best to avoid drifting too far from the known good region.
        HARD_RESET   = 30

        print(f"[{self.name}] ILS-Medium avviato ({len(best_bins)} bins, budget: {time_limit - (time.time() - start_time):.0f}s)...")

        while time.time() - start_time < time_limit:
            ils_iter += 1

            # Perturbation strength is proportional to how many bins exist, so
            # we always shake roughly the same fraction of the solution regardless
            # of dataset size. Strength escalates the longer we stay stuck, to
            # push harder out of the local optimum.
            n_bins = max(len(current_bins), 1)
            if no_improve < 15:
                strength = max(2, n_bins // 10)   # ~10% of bins ejected
            elif no_improve < 25:
                strength = max(3, n_bins // 7)    # ~14%
            else:
                strength = max(4, n_bins // 5)    # ~20%

            # Eject 'strength' bins worth of items and repack them into the rest.
            perturbed = self._ils_perturb(current_bins, vehicles_dict, strength)
            if perturbed is None:
                break

            # Polish the perturbed solution with VND before evaluating it.
            p_sol     = self._rebuild_sol_from_bins(perturbed)
            new_bins, new_sol = self._vnd_loop(p_sol, perturbed, vehicles_dict, items)
            new_cost  = self._cost_from_bins(new_bins, vehicles_dict, items)

            if new_cost < best_cost:
                # Global improvement: update both the best and the current search point.
                best_cost = new_cost
                best_sol  = self._fast_copy_sol(new_sol)
                best_bins = self._fast_copy_bins(new_bins)
                current_bins = self._fast_copy_bins(new_bins)
                current_cost = new_cost
                no_improve   = 0
                elapsed = time.time() - start_time
                print(f"[ILS-M] iter {ils_iter}: new best {best_cost:.2f} (strength={strength}, elapsed={elapsed:.1f}s)")
            elif new_cost <= current_cost:
                # Lateral move: no global improvement but accept to avoid getting
                # trapped — allows exploring neighbouring regions.
                current_bins = self._fast_copy_bins(new_bins)
                current_cost = new_cost
                no_improve  += 1
            else:
                no_improve += 1

            if no_improve >= HARD_RESET:
                # Deeply stuck: reset to global best and try again with fresh energy.
                current_bins = self._fast_copy_bins(best_bins)
                current_cost = best_cost
                no_improve   = 0

        print(f"[{self.name}] ILS-Medium done ({ils_iter} iters). Best cost: {best_cost:.2f}")
        return best_cost, best_bins, best_sol

    # ==================================================================
    # 8. SMALL-DATASET ILS SOLVER
    # ==================================================================

    def _ils_perturb(self, bins, vehicles_dict, strength=1):
        """Eject all items from 'strength' bins and repack into remaining bins.
        Bins to eject are chosen by mixing least-utilized and random picks."""
        if len(bins) < 2:
            return None

        bins_copy = self._fast_copy_bins(bins)
        sorted_by_util = sorted(range(len(bins_copy)),
                                key=lambda i: self._bin_utilization(bins_copy[i]))

        n_to_eject = min(strength, len(bins_copy) - 1)

        if random.random() < 0.4:
            eject_indices = set(random.sample(range(len(bins_copy)), n_to_eject))
        else:
            eject_indices = set(sorted_by_util[:n_to_eject])

        ejected_items, remaining_bins = [], []
        for i, b in enumerate(bins_copy):
            if i in eject_indices:
                ejected_items.extend(b['items'])
            else:
                remaining_bins.append(b)

        if not ejected_items:
            return None

        random.shuffle(ejected_items)
        self._try_repack_records(ejected_items, remaining_bins, vehicles_dict, allow_new=True)
        return remaining_bins

    def _solve_small(self, items, time_limit, start_time, return_detailed=False):
        """ILS-driven solver for small datasets (≤ 200 items).
        Phase 1: short GRASP+VND burst to find a strong initial solution.
        Phase 2: ILS loop (perturb → VND) using the remaining time budget."""

        # ── Setup ───────────────────────────────────────────────────────
        vehicles = self.inst.df_vehicles.copy()
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles['cost_per_vol'] = vehicles['cost'] / vehicles['volume']
        vehicles_for_vnd = vehicles.sort_values(by='cost', ascending=True)
        vehicles_dict = vehicles_for_vnd.to_dict('index')

        items = items.copy()
        items['volume'] = items['width'] * items['depth'] * items['height']
        items['p_i'] = items.apply(self._decide_priority_level, axis=1)

        profile = self._profile_dataset(items, vehicles)
        if profile['diamond_ids'] or profile['lead_ids'] or profile['locked_ids']:
            print(f"[{self.name}] Anomalies — diamonds: {len(profile['diamond_ids'])}, leads: {len(profile['lead_ids'])}, locked: {len(profile['locked_ids'])}")

        vehicles_for_constructive = self._sort_vehicles_adaptive(vehicles, profile)
        items = self._order_items_adaptive(items, profile)

        delta_pool   = [0.1, 0.3, 0.5, 0.8]
        delta_probs  = [0.25, 0.25, 0.25, 0.25]
        delta_history = {d: [] for d in delta_pool}

        best_cost = float('inf')
        best_bins = None
        best_sol  = None

        def _enrich_bins(active_bins):
            for b in active_bins:
                v = vehicles_dict[b['type']]
                b['max_vol']        = float(v['volume'])
                b['current_vol']    = sum(r['dx']*r['dy']*r['dz'] for r in b['items'])
                b['max_value_item'] = float(v.get('maxValue', float('inf')))
                b['current_value']  = sum(float(r['item'].get('value', 0)) for r in b['items'])
                b['last_pruned']    = 1

        # ── Phase 1: GRASP+VND burst (≤ 20% of time budget) ─────────────
        phase1_limit = start_time + time_limit * 0.20
        grasp_iter   = 0
        print(f"[Small] Phase 1: GRASP+VND burst...")

        while time.time() < phase1_limit:
            current_delta = np.random.choice(delta_pool, p=delta_probs)
            _, sol, active_bins = self._run_iteration(
                items, vehicles_for_constructive, current_delta)

            _enrich_bins(active_bins)
            opt_bins, opt_sol = self._vnd_loop(sol, active_bins, vehicles_dict, items)
            cost = self._cost_from_bins(opt_bins, vehicles_dict, items)

            delta_history[current_delta].append(cost)
            grasp_iter += 1

            if cost < best_cost:
                best_cost = cost
                best_sol  = self._fast_copy_sol(opt_sol)
                best_bins = self._fast_copy_bins(opt_bins)
                print(f"[Small] GRASP iter {grasp_iter}: new best {cost:.2f} (delta={current_delta})")

            if grasp_iter % 3 == 0:
                avg = [np.mean(delta_history[d]) if delta_history[d] else best_cost
                       for d in delta_pool]
                inv = [1.0 / (c + 1e-5) for c in avg]
                delta_probs = [v / sum(inv) for v in inv]

        if best_bins is None:
            print("[Small] Phase 1 found no solution — falling back to default solver.")
            return

        print(f"[Small] Phase 1 done ({grasp_iter} iters). Best cost: {best_cost:.2f}")

        # ── Phase 2: ILS loop ────────────────────────────────────────────
        BUFFER_SECS      = 10
        HARD_RESET_LIMIT = 40

        current_bins  = self._fast_copy_bins(best_bins)
        current_cost  = best_cost
        no_improve    = 0
        ils_iter      = 0
        print(f"[Small] Phase 2: ILS loop (budget left: "
              f"{time_limit - (time.time() - start_time):.0f}s)...")

        while time.time() - start_time < time_limit - BUFFER_SECS:
            ils_iter += 1

            # Escalate perturbation strength when stuck
            if no_improve < 10:
                strength = 1
            elif no_improve < 20:
                strength = 2
            else:
                strength = 3

            perturbed = self._ils_perturb(current_bins, vehicles_dict, strength)
            if perturbed is None:
                break

            p_sol      = self._rebuild_sol_from_bins(perturbed)
            new_bins, new_sol = self._vnd_loop(p_sol, perturbed, vehicles_dict, items)
            new_cost   = self._cost_from_bins(new_bins, vehicles_dict, items)

            if new_cost < best_cost:
                best_cost  = new_cost
                best_sol   = self._fast_copy_sol(new_sol)
                best_bins  = self._fast_copy_bins(new_bins)
                current_bins = self._fast_copy_bins(new_bins)
                current_cost = new_cost
                no_improve   = 0
                elapsed = time.time() - start_time
                print(f"[Small] ILS iter {ils_iter}: new best {best_cost:.2f} "
                      f"(strength={strength}, elapsed={elapsed:.1f}s)")
            elif new_cost <= current_cost:
                current_bins = self._fast_copy_bins(new_bins)
                current_cost = new_cost
                no_improve  += 1
            else:
                no_improve += 1

            # Hard reset to global best when deeply stuck
            if no_improve >= HARD_RESET_LIMIT:
                current_bins = self._fast_copy_bins(best_bins)
                current_cost = best_cost
                no_improve   = 0

        print(f"[Small] Phase 2 done ({ils_iter} ILS iters). Best cost: {best_cost:.2f}")

        # ── Post-processing (same as default solver) ─────────────────────
        hard_deadline = start_time + time_limit - 2

        if best_bins and time.time() < hard_deadline - 5:
            improved, best_bins = self._post_process_repack_partial(
                best_bins, vehicles_dict, items, deadline=hard_deadline - 3)
            if improved:
                best_cost = self._cost_from_bins(best_bins, vehicles_dict, items)

        if best_bins and time.time() < hard_deadline - 2:
            improved_cost, best_bins = self._post_process_downgrade(
                best_bins, vehicles_dict, deadline=hard_deadline - 1)
            if improved_cost:
                best_cost = self._cost_from_bins(best_bins, vehicles_dict, items)

        self.sol.clear()
        self.sol.update(self._rebuild_sol_from_bins(best_bins) if best_bins else {})
        self.active_bins = best_bins or []

        elapsed = time.time() - start_time
        print(f"[{self.name}] Small solver done in {elapsed:.2f}s. "
              f"Final cost: {best_cost:.2f}")

        if return_detailed:
            return {'cost': best_cost, 'solution_dict': best_sol, 'active_bins': best_bins}