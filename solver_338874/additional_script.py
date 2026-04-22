import numpy as np
import pandas as pd
import random
from .abstract_solver import AbstractSolver

ROTATIONS = np.array([
    [0, 1, 2],  # r=0: (w, d, h) -> Orientamento originale
    [1, 0, 2],  # r=1: (d, w, h)
    [2, 1, 0],  # r=2: (h, d, w)
    [1, 2, 0],  # r=3: (d, h, w)
    [0, 2, 1],  # r=4: (w, h, d)
    [2, 0, 1],  # r=5: (h, w, d)
], dtype=np.int32)

class ConstructiveSolver(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst)
        self.name = "ConstructivePhase_ReactiveGRASP"

    def _get_rotated_dims(self, item, r):
        dims = np.array([item['width'], item['depth'], item['height']], dtype=np.int32)
        w, d, h = dims[ROTATIONS[r]]  
        return int(d), int(w), int(h) 

    def _decide_priority_level(self, item):
        p_i = 0
        allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]
        if len(allowed_rots) < 6: p_i += 10
        max_dim = max(item['depth'], item['width'], item['height'])
        min_dim = min(item['depth'], item['width'], item['height'])
        if min_dim > 0 and max_dim / min_dim > 3: p_i += 5
        if 'dependency_group' in item and pd.notna(item['dependency_group']):
            p_i += int(item['dependency_group']) * 100 
        return p_i

    # ------------------------------------------------------------------
    # GESTIONE GRAVITA' ED EMPTY MAXIMAL SPACES (EMS)
    # ------------------------------------------------------------------
    def _check_container_gravity_strength(self, item_x, item_y, item_z, dim_d, dim_w, active_bin, v_model):
        """ 
        Verifica che l'oggetto abbia abbastanza area d'appoggio per non cadere.
        """
        if item_z == 0: 
            return True
            
        min_ratio = float(v_model.get('gravityStrength', 75)) / 100.0

        target_z = int(item_z)
        items_underneath = []
        
        # Tolleranza di ricerca +-1 per colmare micro-spazi
        for z_key, items in active_bin.get('z_layers', {}).items():
            if abs(z_key - target_z) <= 1:  
                items_underneath.extend(items)
                
        if not items_underneath: 
            return False

        base_area = float(dim_d * dim_w)
        supported_area = 0.0
        
        for placed in items_underneath:
            if (item_x >= placed['x'] + placed['d'] or item_x + dim_d <= placed['x'] or
                item_y >= placed['y'] + placed['w'] or item_y + dim_w <= placed['y']):
                continue

            overlap_d = max(0, min(item_x + dim_d, placed['x'] + placed['d']) - max(item_x, placed['x']))
            overlap_w = max(0, min(item_y + dim_w, placed['y'] + placed['w']) - max(item_y, placed['y']))
            
            supported_area += (overlap_d * overlap_w)
            
        return (supported_area / base_area) >= min_ratio

    @staticmethod
    def _make_spaces(x, y, z, lx, ly, lz):
        return np.array([[x, y, z, lx, ly, lz]], dtype=np.int32)

    @staticmethod
    def _remove_subsumed_spaces(spaces):
        if len(spaces) <= 1: return spaces
        u_spaces = np.unique(spaces, axis=0)
        if len(u_spaces) <= 1: return u_spaces

        sx, sy, sz = u_spaces[:, 0:1], u_spaces[:, 1:2], u_spaces[:, 2:3]
        ex = sx + u_spaces[:, 3:4]
        ey = sy + u_spaces[:, 4:5]
        ez = sz + u_spaces[:, 5:6]

        inside = (sx >= sx.T) & (sy >= sy.T) & (sz >= sz.T) & \
                 (ex <= ex.T) & (ey <= ey.T) & (ez <= ez.T)
        np.fill_diagonal(inside, False)
        is_subsumed = np.any(inside, axis=1)
        return u_spaces[~is_subsumed]

    def _update_ems_spaces(self, spaces, ix, iy, iz, dx, dy, dz):
        iex, iey, iez = ix + dx, iy + dy, iz + dz
        sx, sy, sz = spaces[:, 0], spaces[:, 1], spaces[:, 2]
        sex, sey, sez = sx + spaces[:, 3], sy + spaces[:, 4], sz + spaces[:, 5]

        inter = (ix < sex) & (iex > sx) & (iy < sey) & (iey > sy) & (iz < sez) & (iez > sz)
        if not np.any(inter): return spaces 

        surviving = spaces[~inter]
        intersected = spaces[inter]
        new_s = []
        
        for s in intersected:
            cx, cy, cz, clx, cly, clz = s
            cex, cey, cez = cx + clx, cy + cly, cz + clz

            if ix > cx:  new_s.append([cx, cy, cz, ix - cx, cly, clz])
            if iex < cex: new_s.append([iex, cy, cz, cex - iex, cly, clz])
            if iy > cy:  new_s.append([cx, cy, cz, clx, iy - cy, clz])
            if iey < cey: new_s.append([cx, iey, cz, clx, cey - iey, clz])
            if iz > cz:  new_s.append([cx, cy, cz, clx, cly, iz - cz])
            if iez < cez: new_s.append([cx, cy, iez, clx, cly, cez - iez])

        if not new_s: return surviving
        return self._remove_subsumed_spaces(np.vstack([surviving, np.array(new_s, dtype=np.int32)]))

    # ------------------------------------------------------------------
    # LOGICA DI VALUTAZIONE E GRASP
    # ------------------------------------------------------------------
    def _evaluate_moves_for_rotation(self, item, r, vehicles, active_bins, item_weight, item_value):
        candidates = []
        
        # Le dimensioni dx (profondità) e dy (larghezza) vengono estratte in base 
        # alla rotazione specifica 'r'. Queste formeranno l'area di base!
        dx, dy, dz = self._get_rotated_dims(item, r)

        for b_idx, active_bin in enumerate(active_bins):
            if active_bin['current_weight'] + item_weight > active_bin['max_weight']: continue
            
            # RECUPERIAMO IL MODELLO DEL VEICOLO DAL SUO 'TYPE'
            v_model = vehicles.loc[active_bin['type']]

            spaces = active_bin['spaces']
            if len(spaces) == 0: continue
            fits = (spaces[:, 3] >= dx) & (spaces[:, 4] >= dy) & (spaces[:, 5] >= dz)
            
            for s_idx in np.where(fits)[0]:
                s_row = spaces[s_idx]
                
                # PASSIAMO 'v_model', 'dx' e 'dy' (le dimensioni ruotate)
                if self._check_container_gravity_strength(s_row[0], s_row[1], s_row[2], dx, dy, active_bin, v_model):
                    
                    x_val, y_val, z_val = int(s_row[0]), int(s_row[1]), int(s_row[2])
                    lx_val, ly_val, lz_val = int(s_row[3]), int(s_row[4]), int(s_row[5])
                    
                    dist_origin = x_val**2 + y_val**2 + z_val**2
                    wasted_vol = (lx_val * ly_val * lz_val) - (dx * dy * dz)
                    
                    score = (0, dist_origin, wasted_vol)
                    candidates.append({
                        'is_new': False, 'b_idx': b_idx, 's_idx': s_idx, 
                        'r': r, 'score': score, 'dx': dx, 'dy': dy, 'dz': dz, 
                        'x': x_val, 'y': y_val, 'z': z_val
                    })

        # Controllo per i veicoli NUOVI (vuoti)
        for v_type, v_model in vehicles.iterrows():
            if int(v_model['maxWeight']) >= item_weight:
                if int(v_model['depth']) >= dx and int(v_model['width']) >= dy and int(v_model['height']) >= dz:
                    cost_penalty = v_model.get('cost', 1000)
                    score = (1, cost_penalty, 0) 
                    candidates.append({
                        'is_new': True, 'v_type': v_type, 'v_model': v_model, 
                        'r': r, 'score': score, 'dx': dx, 'dy': dy, 'dz': dz,
                        'lx': int(v_model['depth']), 'ly': int(v_model['width']), 'lz': int(v_model['height'])
                    })
                    break 

        return candidates

    def solve(self):
        max_iterations = 10 
        delta_pool = [0.1, 0.3, 0.5, 0.8] 
        delta_probs = [0.25, 0.25, 0.25, 0.25]
        delta_history = {d: [] for d in delta_pool}
        
        best_overall_cost = float('inf')
        best_overall_sol = None
        best_overall_bins = None
        items = self.inst.df_items.copy()
        items['volume'] = items['width'] * items['depth'] * items['height']
        items['p_i'] = items.apply(self._decide_priority_level, axis=1)
        items = items.sort_values(by=['volume', 'p_i'], ascending=[False, False])
        
        vehicles = self.inst.df_vehicles.copy()
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles = vehicles.sort_values(by='volume', ascending=False) 

        for iteration in range(max_iterations):
            current_delta = np.random.choice(delta_pool, p=delta_probs)
            cost, solution, bins = self._constructive_iteration(items, vehicles, current_delta)
            
            delta_history[current_delta].append(cost)
            
            # Registra la migliore
            if cost < best_overall_cost or best_overall_sol is None:
                best_overall_cost = cost
                best_overall_sol = solution
                best_overall_bins = bins 
                print(f"Iter {iteration}: Nuovo best cost {cost} (delta={current_delta})")

            if (iteration + 1) % 3 == 0:
                avg_costs = [np.mean(delta_history[d]) if delta_history[d] else best_overall_cost for d in delta_pool]
                inv_costs = [1.0 / (c + 1e-5) for c in avg_costs]
                delta_probs = [inv / sum(inv_costs) for inv in inv_costs]

        # RISOLTO RIFERIMENTO MEMORIA: Non riassegniamo il dizionario, ma lo aggiorniamo in-place
        self.sol.clear()
        self.sol.update(best_overall_sol)
        self.active_bins = best_overall_bins

    def _constructive_iteration(self, items, vehicles, delta):
        active_bins = []
        vehicle_counter = 0
        current_sol = {'type_vehicle': [], 'idx_vehicle': [], 'id_item': [], 
                       'x_origin': [], 'y_origin': [], 'z_origin': [], 'orient': []}

        for i_idx, item in items.iterrows():
            item_weight = float(item.get('weight', 0))
            item_value  = float(item.get('value',  0))
            allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]
            
            candidates = []
            
            if 0 in allowed_rots:
                candidates = self._evaluate_moves_for_rotation(item, 0, vehicles, active_bins, item_weight, item_value)
            elif len(allowed_rots) > 0:
                candidates = self._evaluate_moves_for_rotation(item, allowed_rots[0], vehicles, active_bins, item_weight, item_value)

            if not candidates:
                for r in allowed_rots:
                    if r == 0: continue 
                    candidates.extend(self._evaluate_moves_for_rotation(item, r, vehicles, active_bins, item_weight, item_value))

            if not candidates:
                continue # L'item non entra. La penalità sarà calcolata alla fine.

            candidates.sort(key=lambda x: x['score'])
            rcl_size = max(1, int(len(candidates) * delta))
            move = random.choice(candidates[:rcl_size])

            if move['is_new']:
                x, y, z = 0, 0, 0
                active_bin = {
                    'type': move['v_type'], 'idx': vehicle_counter, 
                    'spaces': self._make_spaces(0, 0, 0, move['lx'], move['ly'], move['lz']),
                    'current_weight': item_weight, 'max_weight': float(move['v_model']['maxWeight']),
                    'z_layers': {},
                    'items': []
                }
                active_bins.append(active_bin)
                b_idx_to_save = vehicle_counter
                v_type_to_save = move['v_type']
                vehicle_counter += 1
            else:
                x, y, z = move['x'], move['y'], move['z']
                active_bin = active_bins[move['b_idx']]
                active_bin['current_weight'] += item_weight
                b_idx_to_save = active_bin['idx']
                v_type_to_save = active_bin['type']

            active_bin['spaces'] = self._update_ems_spaces(active_bin['spaces'], x, y, z, move['dx'], move['dy'], move['dz'])
            
            top_z = int(z + move['dz'])
            if top_z not in active_bin['z_layers']: active_bin['z_layers'][top_z] = []
            active_bin['z_layers'][top_z].append({'x': x, 'y': y, 'd': move['dx'], 'w': move['dy']})

            active_bin['items'].append({
                'i_idx': i_idx, 'item': item,
                'x': x, 'y': y, 'z': z,
                'dx': move['dx'], 'dy': move['dy'], 'dz': move['dz'],
                'r': move['r'], 'weight': item_weight
            })

            current_sol['type_vehicle'].append(v_type_to_save)
            current_sol['idx_vehicle'].append(b_idx_to_save)
            current_sol['id_item'].append(i_idx)
            current_sol['x_origin'].append(x)
            current_sol['y_origin'].append(y)
            current_sol['z_origin'].append(z)
            current_sol['orient'].append(move['r'])

        # RISOLTO BUG PENALITA': Se la soluzione manca di oggetti, il costo è infinito
        if len(current_sol['id_item']) < len(items):
            total_cost = float('inf') 
        else:
            total_cost = sum(vehicles.loc[b['type'], 'cost'] for b in active_bins) if 'cost' in vehicles.columns else len(active_bins)
        
        return total_cost, current_sol, active_bins
