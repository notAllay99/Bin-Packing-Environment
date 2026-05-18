import numpy as np
import pandas as pd
import random
from .abstract_solver import AbstractSolver
import math

ROTATIONS = np.array([
    [0, 1, 2], [1, 0, 2], [2, 1, 0], 
    [1, 2, 0], [0, 2, 1], [2, 0, 1]
], dtype=np.int32)
def calculate_theoretical_lower_bounds(df_items, df_vehicles):
    """
    Calcola il Lower Bound teorico per il costo del 3D Bin Packing Problem.
    Restituisce un dizionario con i limiti calcolati.
    """
    items = df_items.copy()
    vehicles = df_vehicles.copy()

    # 1. Calcolo del volume e del peso totale degli items
    items['volume'] = items['width'] * items['depth'] * items['height']
    total_volume = items['volume'].sum()
    total_weight = items['weight'].sum()

    # 2. Calcolo della capacità volumetrica e di peso dei veicoli
    vehicles['volume_capacity'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
    vehicles['weight_capacity'] = vehicles['maxWeight']
    
    # Se la colonna costo non esiste (es. minimizzare solo il numero di veicoli), usiamo 1
    if 'cost' not in vehicles.columns:
        vehicles['cost'] = 1.0

    # 3. Calcolo dell'efficienza (Costo per unità di capacità)
    # Più il valore è basso, più il veicolo è conveniente
    vehicles['cost_per_volume'] = vehicles['cost'] / vehicles['volume_capacity']
    vehicles['cost_per_weight'] = vehicles['cost'] / vehicles['weight_capacity']

    # Troviamo il veicolo teoricamente migliore per il volume e per il peso
    best_vol_efficiency = vehicles['cost_per_volume'].min()
    best_weight_efficiency = vehicles['cost_per_weight'].min()

    # 4. Calcolo del Lower Bound Continuo (L1)
    # Moltiplichiamo il fabbisogno totale per il costo dell'unità più economica
    lb_volume = math.ceil(total_volume * best_vol_efficiency)
    lb_weight = math.ceil(total_weight * best_weight_efficiency)

    # Il Lower Bound finale è il vincolo più stringente (il massimo tra i due)
    absolute_lower_bound = max(lb_volume, lb_weight)

    return {
        "Total_Items_Volume": total_volume,
        "Total_Items_Weight": total_weight,
        "LB_Cost_Based_On_Volume": lb_volume,
        "LB_Cost_Based_On_Weight": lb_weight,
        "Absolute_Lower_Bound_Cost": absolute_lower_bound
    }

# ==========================================
# ESEMPIO DI UTILIZZO:
# ==========================================
# from instances.instance import Instance
#
# inst = Instance('datasets/Dataset0')
# bounds = calculate_theoretical_lower_bounds(inst.df_items, inst.df_vehicles)
#
# print(f"Il costo MINIMO teorico possibile è: {bounds['Absolute_Lower_Bound_Cost']}")
# print(f"(Dettagli: LB Volume={bounds['LB_Cost_Based_On_Volume']}, LB Peso={bounds['LB_Cost_Based_On_Weight']})")
class ConstructiveSolver(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst)
        self.name = "ConstructiveSolver"
        self.active_bins = []

    def _get_rotated_dims(self, item, r):
        dims = np.array([item['width'], item['depth'], item['height']], dtype=np.int32)
        w, d, h = dims[ROTATIONS[r]]  
        return int(d), int(w), int(h)

    def _decide_priority_level(self, item):
        p_i = 0
        allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]
        if len(allowed_rots) < 6: p_i += 10
        if 'dependency_group' in item and pd.notna(item['dependency_group']):
            p_i += int(item['dependency_group']) * 100 
        return p_i

    def _check_container_gravity_strength(self, item_x, item_y, item_z, dim_d, dim_w, active_bin, v_model):
        """Versione Vettorizzata (NumPy) per il controllo di stabilità gravitazionale."""
        if item_z == 0:
            return True
        
        min_ratio = float(v_model.get('gravityStrength', 75)) / 100.0
        target_z = int(item_z)
        
        nearby = []
        for z_key, layer_items in active_bin.get('z_layers', {}).items():
            if abs(z_key - target_z) <= 1:
                nearby.extend(layer_items)
        
        if not nearby:
            return False
        
        # Vettorizzazione estrema: calcolo intersezioni 2D in blocco
        placed = np.array([[p['x'], p['y'], p['x'] + p['d'], p['y'] + p['w']] for p in nearby])
        ix2, iy2 = item_x + dim_d, item_y + dim_w
        
        overlap_d = np.maximum(0, np.minimum(ix2, placed[:, 2]) - np.maximum(item_x, placed[:, 0]))
        overlap_w = np.maximum(0, np.minimum(iy2, placed[:, 3]) - np.maximum(item_y, placed[:, 1]))
        
        supported_area = np.sum(overlap_d * overlap_w)
        return (supported_area / float(dim_d * dim_w)) >= (min_ratio - 0.001)

    @staticmethod
    def _make_spaces(x, y, z, lx, ly, lz):
        return np.array([[x, y, z, lx, ly, lz]], dtype=np.int32)

    def _prune_spaces(self, spaces):
        """Pruning degli Spazi Vuoti Massimali (EMS) ottimizzato con pre-ordinamento decrescente."""
        if len(spaces) < 2:
            return spaces
        
        u_spaces = np.unique(spaces, axis=0)
        # Calcolo del volume per ordinare dal più grande al più piccolo
        vols = u_spaces[:, 3] * u_spaces[:, 4] * u_spaces[:, 5]
        order = np.argsort(-vols)
        u_spaces = u_spaces[order]
        
        sx, sy, sz = u_spaces[:, 0:1], u_spaces[:, 1:2], u_spaces[:, 2:3]
        ex = sx + u_spaces[:, 3:4]
        ey = sy + u_spaces[:, 4:5]
        ez = sz + u_spaces[:, 5:6]
        
        inside = (sx >= sx.T) & (sy >= sy.T) & (sz >= sz.T) & \
                 (ex <= ex.T) & (ey <= ey.T) & (ez <= ez.T)
        np.fill_diagonal(inside, False)
        
        return u_spaces[~np.any(inside, axis=1)]

    def _update_ems_spaces(self, spaces, ix, iy, iz, dx, dy, dz, active_bin):
        """Aggiorna gli EMS dopo il posizionamento di un item.
        La sussunzione è lazy: viene eseguita solo quando il numero di spazi
        supera la soglia dinamica max(40, last_pruned * 1.5), evitando
        l'O(n²) ad ogni inserimento senza introdurre spazi non validi
        (gli spazi ridondanti puntano sempre a regioni libere)."""
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
            if ix > cx:   new_s.append([cx, cy, cz, ix - cx, cly, clz])
            if iex < cex: new_s.append([iex, cy, cz, cex - iex, cly, clz])
            if iy > cy:   new_s.append([cx, cy, cz, clx, iy - cy, clz])
            if iey < cey: new_s.append([cx, iey, cz, clx, cey - iey, clz])
            if iz > cz:   new_s.append([cx, cy, cz, clx, cly, iz - cz])
            if iez < cez: new_s.append([cx, cy, iez, clx, cly, cez - iez])

        if not new_s:
            return surviving

        merged = np.vstack([surviving, np.array(new_s, dtype=np.int32)])

        # Soglia lazy: sussunzione solo se gli spazi sono cresciuti
        # del 50% rispetto all'ultima potatura, con floor a 40.
        threshold = max(40, int(active_bin.get('last_pruned', 1) * 1.5))
        if len(merged) >= threshold:
            merged = self._prune_spaces(merged)
            active_bin['last_pruned'] = len(merged)

        return merged

    def _evaluate_moves_for_rotation(self, item, r, vehicles, active_bins, item_weight, item_value):
        candidates = []
        dx, dy, dz = self._get_rotated_dims(item, r)
        
        for b_idx, active_bin in enumerate(active_bins):
            if active_bin['current_weight'] + item_weight > active_bin['max_weight']: continue
            if active_bin.get('current_value', 0) + item_value > active_bin.get('max_value_item', float('inf')): continue
            v_model = vehicles.loc[active_bin['type']]
            spaces = active_bin['spaces']
            fits = (spaces[:, 3] >= dx) & (spaces[:, 4] >= dy) & (spaces[:, 5] >= dz)
            for s_idx in np.where(fits)[0]:
                s_row = spaces[s_idx]
                if self._check_container_gravity_strength(s_row[0], s_row[1], s_row[2], dx, dy, active_bin, v_model):
                    x_v, y_v, z_v = int(s_row[0]), int(s_row[1]), int(s_row[2])
                    dist = x_v**2 + y_v**2 + z_v**2
                    waste = (int(s_row[3])*int(s_row[4])*int(s_row[5])) - (dx*dy*dz)
                    candidates.append({
                        'is_new': False, 'b_idx': b_idx, 'r': r, 
                        'score': (0, dist, waste), 'dx': dx, 'dy': dy, 'dz': dz, 
                        'x': x_v, 'y': y_v, 'z': z_v
                    })
                    
        for v_type, v_model in vehicles.iterrows():
            if (int(v_model['maxWeight']) >= item_weight and
                    float(v_model.get('maxValue', float('inf'))) >= item_value and
                    int(v_model['depth']) >= dx and int(v_model['width']) >= dy and int(v_model['height']) >= dz):
                cost = v_model.get('cost', 1000)
                waste = 1.0 - ((dx*dy*dz)/(int(v_model['depth'])*int(v_model['width'])*int(v_model['height'])))
                candidates.append({
                    'is_new': True, 'v_type': v_type, 'v_model': v_model, 'r': r, 
                    'score': (1, cost, waste), 'dx': dx, 'dy': dy, 'dz': dz, 
                    'lx': int(v_model['depth']), 'ly': int(v_model['width']), 'lz': int(v_model['height'])
                })
                break 
        return candidates

    def solve(self, return_detailed=False):
        max_iterations = 10 
        delta_pool = [0.1, 0.3, 0.5, 0.8] 
        delta_probs = [0.25, 0.25, 0.25, 0.25]
        delta_history = {d: [] for d in delta_pool}
        
        best_overall_cost = float('inf')
        best_overall_sol = None
        best_active_bins = None

        items = self.inst.df_items.copy()
        items['volume'] = items['width'] * items['depth'] * items['height']
        items['p_i'] = items.apply(self._decide_priority_level, axis=1)
        items = items.sort_values(by=['volume', 'p_i'], ascending=[False, False])
        
        vehicles = self.inst.df_vehicles.copy()
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles = vehicles.sort_values(by='volume', ascending=False) 

        for iteration in range(max_iterations):
            current_delta = np.random.choice(delta_pool, p=delta_probs)
            cost, solution, a_bins = self._run_iteration(items, vehicles, current_delta)
            
            delta_history[current_delta].append(cost)
            
            if cost < best_overall_cost or best_overall_sol is None:
                best_overall_cost = cost
                best_overall_sol = solution
                best_active_bins = a_bins
                # Stampiamo il log solo se non siamo in modalità 'silenziosa/dettagliata'
                if not return_detailed:
                    print(f"Constructive Iter {iteration}: Nuovo best cost {cost:.2f} (delta={current_delta})")

            if (iteration + 1) % 3 == 0:
                avg_costs = [np.mean(delta_history[d]) if delta_history[d] else best_overall_cost for d in delta_pool]
                inv_costs = [1.0 / (c + 1e-5) for c in avg_costs]
                delta_probs = [inv / sum(inv_costs) for inv in inv_costs]

        # Aggiornamento standard dello stato della classe
        self.sol.clear()
        if best_overall_sol: 
            self.sol.update(best_overall_sol)
        self.active_bins = best_active_bins or []

        # GESTIONE DELLA FLAG DI RITORNO
        if return_detailed:
            return {
                'cost': best_overall_cost,
                'solution_dict': best_overall_sol,
                'active_bins': best_active_bins
            }
        else:
            # Comportamento standard: ritorna nulla o solo la soluzione formattata
            return best_overall_sol

    def _run_iteration(self, items, vehicles, delta):
        active_bins, vehicle_counter = [], 0
        sol = {'type_vehicle': [], 'idx_vehicle': [], 'id_item': [], 'x_origin': [], 'y_origin': [], 'z_origin': [], 'orient': []}
        
        for i_idx, item in items.iterrows():
            allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]
            candidates = []
            
            for r in allowed_rots:
                candidates.extend(self._evaluate_moves_for_rotation(item, r, vehicles, active_bins, float(item['weight']), float(item.get('value', 0))))
            if not candidates: continue
            
            candidates.sort(key=lambda x: x['score'])
            move = random.choice(candidates[:max(1, int(len(candidates)*delta))])
            
            if move['is_new']:
                x = y = z = 0
                ab = {
                    'type': move['v_type'],
                    'idx': vehicle_counter,
                    'spaces': self._make_spaces(0, 0, 0, move['lx'], move['ly'], move['lz']),
                    'current_weight': float(item['weight']),
                    'max_weight': float(move['v_model']['maxWeight']),
                    'current_value': float(item.get('value', 0)),
                    'max_value_item': float(move['v_model'].get('maxValue', float('inf'))),
                    'z_layers': {},
                    'items': [],
                    'last_pruned': 1
                }
                active_bins.append(ab)
                v_idx, v_type, vehicle_counter = ab['idx'], ab['type'], vehicle_counter + 1
            else:
                ab = active_bins[move['b_idx']]
                x, y, z = move['x'], move['y'], move['z']
                ab['current_weight'] += float(item['weight'])
                ab['current_value'] = ab.get('current_value', 0) + float(item.get('value', 0))
                v_idx, v_type = ab['idx'], ab['type']
                
            # Correzione: passaggio di 'ab' come ultimo argomento
            ab['spaces'] = self._update_ems_spaces(ab['spaces'], x, y, z, move['dx'], move['dy'], move['dz'], ab)
            
            top_z = int(z + move['dz'])
            ab['z_layers'].setdefault(top_z, []).append({'x': x, 'y': y, 'd': move['dx'], 'w': move['dy']})
            ab['items'].append({
                'i_idx': i_idx, 'item': item, 'x': x, 'y': y, 'z': z, 
                'dx': move['dx'], 'dy': move['dy'], 'dz': move['dz'], 'r': move['r'], 'weight': float(item['weight'])
            })
            
            sol['type_vehicle'].append(v_type); sol['idx_vehicle'].append(v_idx); sol['id_item'].append(i_idx)
            sol['x_origin'].append(x); sol['y_origin'].append(y); sol['z_origin'].append(z); sol['orient'].append(move['r'])
            
        base_cost = sum(vehicles.loc[b['type'], 'cost'] for b in active_bins) if 'cost' in vehicles.columns else len(active_bins)
        unpacked_items = len(items) - len(sol['id_item'])
        cost = base_cost + (unpacked_items * 1000000)

        return cost, sol, active_bins