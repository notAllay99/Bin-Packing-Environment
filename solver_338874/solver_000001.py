import copy
import random
import numpy as np
from .additional_script import ConstructiveSolver

class ImprovementSolver(ConstructiveSolver):
    """
    Ottimizzatore VND Standalone.
    Prende la soluzione del ConstructiveSolver e la compatta 
    usando la ricerca locale e la rigenerazione degli spazi (EMS).
    """

    def __init__(self, inst):
        super().__init__(inst)
        self.name = "ImprovedSolver"

    # ==================================================================
    # UTILITY DI COPIA VELOCE E RIGENERAZIONE GEOMETRICA
    # ==================================================================

    def _fast_copy_sol(self, sol):
        return {k: v[:] for k, v in sol.items()}

    def _fast_copy_bins(self, active_bins):
        new_bins = []
        for b in active_bins:
            new_bins.append({
                'type': b['type'], 'idx': b['idx'],
                'current_weight': b['current_weight'], 'max_weight': b['max_weight'],
                'spaces': b['spaces'].copy(),
                'z_layers': {z: [val.copy() for val in lst] for z, lst in b['z_layers'].items()},
                'items': [rec.copy() for rec in b['items']]
            })
        return new_bins

    def _rebuild_bin_spaces(self, ab):
        """
        FUNZIONE CRITICA: Quando rimuovi un oggetto, lo spazio 3D non si aggiorna da solo.
        Questa funzione resetta il veicolo e ci rimette dentro gli oggetti rimasti,
        ricreando perfettamente gli spazi vuoti disponibili.
        """
        v_model = self.vehicles_dict[ab['type']]
        ab['spaces'] = self._make_spaces(0, 0, 0, int(v_model['depth']), int(v_model['width']), int(v_model['height']))
        ab['z_layers'] = {}
        
        # Rimettiamo gli oggetti dal basso verso l'alto per non rompere la gravità
        remaining = sorted(ab['items'], key=lambda r: r['z'])
        for rec in remaining:
            ab['spaces'] = self._update_ems_spaces(ab['spaces'], rec['x'], rec['y'], rec['z'], rec['dx'], rec['dy'], rec['dz'])
            top_z = int(rec['z'] + rec['dz'])
            ab['z_layers'].setdefault(top_z, []).append({'x': rec['x'], 'y': rec['y'], 'd': rec['dx'], 'w': rec['dy']})

    # ==================================================================
    # ENTRY POINT PRINCIPALE
    # ==================================================================

    def solve_from_constructive(self, constructive_solver):
        print("\n--- Inizio Improvement Phase (VND) ---")
        
        items_df    = self.inst.df_items.copy()
        vehicles_df = self.inst.df_vehicles.copy()

        self.items_dict    = items_df.to_dict(orient='index')
        self.vehicles_dict = vehicles_df.to_dict(orient='index')

        total_items = len(self.items_dict)

        max_iterations   = 20
        no_improve_limit = 5

        frequency_vector = {i_idx: 0 for i_idx in items_df.index}

        base_sol  = self._fast_copy_sol(constructive_solver.sol)
        base_bins = self._fast_copy_bins(constructive_solver.active_bins)

        best_metrics = self._evaluate_solution(base_bins, total_items)
        print(f"Veicoli usati in partenza: {best_metrics['total_bins']} | Items fuori: {best_metrics['unfit_items']}")

        best_sol  = self._fast_copy_sol(base_sol)
        best_bins = self._fast_copy_bins(base_bins)
        no_improve_count = 0

        for iteration in range(max_iterations):
            if no_improve_count >= no_improve_limit:
                candidate_bins = self._perturb_solution(best_bins, frequency_vector)
                no_improve_count = 0
                print(f"Iter {iteration}: Perturbazione applicata per sblocco.")
            else:
                candidate_bins = self._fast_copy_bins(best_bins)

            candidate_sol = self._rebuild_sol_from_bins(candidate_bins)

            # Aggiorna frequenza item non piazzati
            packed_ids = set(candidate_sol['id_item'])
            for i_idx in items_df.index:
                if i_idx not in packed_ids:
                    frequency_vector[i_idx] += 1

            # Lancia il VND
            metrics, sol, bins = self._improvement_phase(candidate_sol, candidate_bins, total_items)

            # Se c'è un miglioramento oggettivo (meno veicoli, o meno oggetti fuori, o più compatto)
            if self._is_better(metrics, best_metrics):
                best_metrics = metrics
                best_sol     = self._fast_copy_sol(sol)
                best_bins    = self._fast_copy_bins(bins)
                no_improve_count = 0
                print(f"Iter {iteration}: Miglioramento! Veicoli={metrics['total_bins']} | Fuori={metrics['unfit_items']} | Compattazione={metrics['sum_sq_util']:.2f}")
            else:
                no_improve_count += 1

            if no_improve_count >= no_improve_limit * 2:
                print("Ottimizzazione esaurita.")
                break

            if metrics['total_bins'] == 1 and metrics['unfit_items'] == 0:
                print("1 solo veicolo usato. Ottimo assoluto raggiunto.")
                break

        self.sol.clear()
        self.sol.update(best_sol)
        self.active_bins = best_bins

    # ==================================================================
    # METRICHE RISCRITTE (CRITICHE PER IL VND)
    # ==================================================================

    def _bin_utilization(self, ab):
        v_row = self.vehicles_dict[ab['type']]
        cap   = float(v_row['depth']) * float(v_row['width']) * float(v_row['height'])
        used  = sum(r['dx'] * r['dy'] * r['dz'] for r in ab['items'])
        return used / max(cap, 1)

    def _evaluate_solution(self, active_bins, total_items_count):
        packed_ids  = {rec['i_idx'] for b in active_bins for rec in b['items']}
        unfit_items = total_items_count - len(packed_ids)
        
        # LA METRICA QUADRATICA: (0.9^2 + 0.1^2) = 0.82 è MEGLIO di (0.5^2 + 0.5^2) = 0.50
        # Questo forza l'algoritmo a riempire totalmente un camion svuotandone un altro!
        sum_sq_util = sum(self._bin_utilization(b)**2 for b in active_bins)

        return {
            'total_bins': len(active_bins),
            'unfit_items': unfit_items,
            'sum_sq_util': sum_sq_util
        }

    def _is_better(self, new_m, old_m):
        if new_m['unfit_items'] < old_m['unfit_items']: return True
        if new_m['unfit_items'] > old_m['unfit_items']: return False
        
        if new_m['total_bins'] < old_m['total_bins']: return True
        if new_m['total_bins'] > old_m['total_bins']: return False
        
        # A parità di camion usati e scatole lasciate fuori, vince chi compatta di più
        return new_m['sum_sq_util'] > old_m['sum_sq_util']

    # ==================================================================
    # CORE: MOTORE VND E REPACKING
    # ==================================================================

    def _improvement_phase(self, sol, active_bins, total_items_count):
        neighborhood_moves = [
            self._move_remove_least_occupied,
            self._move_selective_removal,
            self._move_split_bins,
            self._move_compact_bins,
            self._move_pairwise_merge,
        ]

        best_sol     = self._fast_copy_sol(sol)
        best_bins    = self._fast_copy_bins(active_bins)
        best_metrics = self._evaluate_solution(best_bins, total_items_count)

        neighborhood_idx = 0
        max_vnd_iter     = 50

        for _ in range(max_vnd_iter):
            if neighborhood_idx >= len(neighborhood_moves): break
            move_fn = neighborhood_moves[neighborhood_idx]

            candidate_sol  = self._fast_copy_sol(best_sol)
            candidate_bins = self._fast_copy_bins(best_bins)

            success = move_fn(candidate_sol, candidate_bins)

            if success:
                new_metrics = self._evaluate_solution(candidate_bins, total_items_count)
                if self._is_better(new_metrics, best_metrics):
                    best_sol, best_bins, best_metrics = candidate_sol, candidate_bins, new_metrics
                    neighborhood_idx = 0  # Restart del VND
                else: neighborhood_idx += 1
            else: neighborhood_idx += 1

        return best_metrics, best_sol, best_bins

    def _try_repack_records(self, records, active_bins):
        unpacked = []
        
        # Recuperiamo il dataframe dei veicoli per passarlo al padre
        vehicles_df = self.inst.df_vehicles 
        
        for rec in records:
            item, item_weight = rec['item'], rec['weight']
            allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]

            all_candidates = []
            for r in allowed_rots:
                # FIX: Aggiunto 'vehicles_df' tra 'r' e 'active_bins'
                cands = self._evaluate_moves_for_rotation(item, r, vehicles_df, active_bins, item_weight, 0)
                all_candidates.extend([c for c in cands if not c['is_new']])

            if all_candidates:
                all_candidates.sort(key=lambda x: x['score'])
                move = all_candidates[0]
                
                ab = active_bins[move['b_idx']]
                x, y, z = move['x'], move['y'], move['z']
                
                ab['current_weight'] += item_weight
                ab['spaces'] = self._update_ems_spaces(ab['spaces'], x, y, z, move['dx'], move['dy'], move['dz'])
                top_z = int(z + move['dz'])
                ab['z_layers'].setdefault(top_z, []).append({'x': x, 'y': y, 'd': move['dx'], 'w': move['dy']})
                
                rec_copy = rec.copy()
                rec_copy.update({'x': x, 'y': y, 'z': z, 'dx': move['dx'], 'dy': move['dy'], 'dz': move['dz'], 'r': move['r']})
                ab['items'].append(rec_copy)
            else:
                unpacked.append(rec)
                
        return len(unpacked) == 0, unpacked
    def _rebuild_sol_from_bins(self, active_bins):
        sol = {'type_vehicle': [], 'idx_vehicle': [], 'id_item': [], 'x_origin': [], 'y_origin': [], 'z_origin': [], 'orient': []}
        for ab in active_bins:
            for rec in ab['items']:
                sol['type_vehicle'].append(ab['type']); sol['idx_vehicle'].append(ab['idx']); sol['id_item'].append(rec['i_idx'])
                sol['x_origin'].append(rec['x']); sol['y_origin'].append(rec['y']); sol['z_origin'].append(rec['z'])
                sol['orient'].append(rec['r'])
        return sol

    # ==================================================================
    # MOSSE DI VICINATO (NEIGHBORHOODS)
    # ==================================================================

    def _move_remove_least_occupied(self, sol, active_bins):
        if len(active_bins) < 2: return False
        utils = [(self._bin_utilization(ab), i) for i, ab in enumerate(active_bins)]
        utils.sort()
        worst_util, worst_idx = utils[0]

        worst_bin = active_bins[worst_idx]
        records = [r.copy() for r in worst_bin['items']]
        other_bins = [ab for i, ab in enumerate(active_bins) if i != worst_idx]

        success, _ = self._try_repack_records(records, other_bins)
        if success:
            active_bins.pop(worst_idx)
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False

    def _move_selective_removal(self, sol, active_bins):
        if not active_bins: return False
        utils = [(self._bin_utilization(ab), i) for i, ab in enumerate(active_bins)]
        utils.sort()
        worst_bin = active_bins[utils[0][1]]

        n_remove = max(1, int(len(worst_bin['items']) * 0.20))
        to_remove = random.sample(worst_bin['items'], n_remove)
        
        for rec in to_remove:
            worst_bin['items'].remove(rec)
            worst_bin['current_weight'] -= rec['weight']
        
        # DISTRUZIONE SPAZI FANTASMA
        self._rebuild_bin_spaces(worst_bin)

        success, unpacked = self._try_repack_records([r.copy() for r in to_remove], active_bins)
        if success:
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False

    def _move_split_bins(self, sol, active_bins):
        nonempty = [ab for ab in active_bins if ab['items']]
        if not nonempty: return False

        ab = random.choice(nonempty)
        axis = random.choice(['depth', 'width', 'height'])
        dim_map = {'depth': 'dx', 'width': 'dy', 'height': 'dz'}
        half = float(self.vehicles_dict[ab['type']][axis]) / 2.0

        to_move = [r for r in ab['items'] if r[dim_map[axis]] > half]
        if not to_move: return False

        for rec in to_move:
            ab['items'].remove(rec)
            ab['current_weight'] -= rec['weight']
        
        # DISTRUZIONE SPAZI FANTASMA
        self._rebuild_bin_spaces(ab)

        success, unpacked = self._try_repack_records([r.copy() for r in to_move], active_bins)
        if success:
            sol.update(self._rebuild_sol_from_bins(active_bins))
            return True
        return False

    def _move_compact_bins(self, sol, active_bins):
        if len(active_bins) < 2: return False
        sorted_bins = sorted(active_bins, key=self._bin_utilization, reverse=True)
        low_util = [ab for ab in sorted_bins if self._bin_utilization(ab) < 0.50]
        
        improved = False
        for low_ab in low_util:
            records = list(low_ab['items'])
            for rec in records:
                low_ab['items'].remove(rec)
                low_ab['current_weight'] -= rec['weight']
                self._rebuild_bin_spaces(low_ab) # DISTRUZIONE SPAZI FANTASMA
                
                # Prova a infilarlo ovunque (anche nello stesso bin se si è liberato posto megliore)
                ok, _ = self._try_repack_records([rec.copy()], active_bins)
                if ok: improved = True
                else:
                    # Rollback se fallisce clamorosamente
                    low_ab['items'].append(rec)
                    low_ab['current_weight'] += rec['weight']
                    self._rebuild_bin_spaces(low_ab)

            if not low_ab['items']: 
                active_bins.remove(low_ab)

        if improved: sol.update(self._rebuild_sol_from_bins(active_bins))
        return improved

    def _move_pairwise_merge(self, sol, active_bins):
        if len(active_bins) < 2: return False
        for i in range(len(active_bins)):
            for j in range(i + 1, len(active_bins)):
                ab1, ab2 = active_bins[i], active_bins[j]
                combined = ab1['items'] + ab2['items']
                
                v_row = self.vehicles_dict[ab1['type']]
                if (sum(r['weight'] for r in combined) <= float(v_row['maxWeight']) and 
                    sum(r['dx'] * r['dy'] * r['dz'] for r in combined) <= float(v_row['depth']) * float(v_row['width']) * float(v_row['height'])):
                    
                    temp_bins = self._fast_copy_bins([ab1])
                    success, _ = self._try_repack_records([r.copy() for r in ab2['items']], temp_bins)
                    
                    if success:
                        active_bins[i] = temp_bins[0]
                        active_bins.pop(j)
                        sol.update(self._rebuild_sol_from_bins(active_bins))
                        return True
        return False

    # ==================================================================
    # PERTURBAZIONE
    # ==================================================================

    def _perturb_solution(self, bins, frequency_vector):
        candidate_bins = self._fast_copy_bins(bins)
        if not candidate_bins: return candidate_bins

        worst_bin = sorted(candidate_bins, key=self._bin_utilization)[0]
        n_remove = max(1, int(len(worst_bin['items']) * 0.3))
        
        worst_bin['items'].sort(key=lambda r: -frequency_vector.get(r['i_idx'], 0))
        to_remove = worst_bin['items'][:n_remove]

        for rec in to_remove:
            worst_bin['items'].remove(rec)
            worst_bin['current_weight'] -= rec['weight']
            
        self._rebuild_bin_spaces(worst_bin)
        self._try_repack_records([r.copy() for r in to_remove], candidate_bins)

        if not worst_bin['items']:
            candidate_bins.remove(worst_bin)
        return candidate_bins