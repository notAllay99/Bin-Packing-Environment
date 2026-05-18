import math
import copy
import random
import time as _time
import numpy as np
import pandas as pd
from .abstract_solver import AbstractSolver
from solver_338874.solver_338874 import solver_338874 as _SolverBase

INFEASIBLE = 100000


def _bo_orient(box, BO):
    """Apply BRKGA orientation BO (1–6) to a (depth, width, height) tuple."""
    d, w, h = box
    if   BO == 1: return (d, w, h)
    elif BO == 2: return (d, h, w)
    elif BO == 3: return (w, d, h)
    elif BO == 4: return (w, h, d)
    elif BO == 5: return (h, d, w)
    elif BO == 6: return (h, w, d)

"""
def generateInstances(N = 20, m = 10, V = (100,100,100)):
    def ur(lb, ub):
        # u.r. is an abbreviation of "uniformly random". [Martello (1995)]
        value = random.uniform(lb, ub)
        return int(value) if value >= 1 else 1
    
    L, W, H = V
    p = []; q = []; r = []
    for i in range(N):
        p.append(ur(1/6*L, 1/4*L))
        q.append(ur(1/6*W, 1/4*W))
        r.append(ur(1/6*H, 1/4*H))
    
    L = [L]*m
    W = [W]*m
    H = [H]*m
    return range(N), range(m), p, q, r, L, W, H

def generateInputs(N, m, V):
    N, M, p,q,r, L,W,H =generateInstances(N, m, V)
    inputs = {'v':list(zip(p, q, r)), 'V':list(zip(L, W, H))}
    return inputs
"""
class Bin():
    def __init__(self, V, max_value=float('inf'), gravity_strength=0.0, verbose=False):
        self.dimensions = V
        self.EMSs = [[np.array((0,0,0)), np.array(V)]]
        self.load_items = []
        self.current_value = 0.0
        self.max_value = max_value
        self.gravity_strength = gravity_strength

        if verbose:
            print('Init EMSs:',self.EMSs)
    
    def __getitem__(self, index):
        return self.EMSs[index]
    
    def __len__(self):
        return len(self.EMSs)

    
    def update(self, box, selected_EMS, min_vol = 1, min_dim = 1, verbose=False):

        # 1. place box in a EMS
        boxToPlace = np.array(box)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace]
        self.load_items.append(ems)
        
        if verbose:
            print('------------\n*Place Box*:\nEMS:', list(map(tuple, ems)))
        
        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):
                
                # eliminate overlapped EMS
                self.eliminate(EMS)
                
                if verbose:
                    print('\n*Elimination*:\nRemove overlapped EMS:',list(map(tuple, EMS)),'\nEMSs left:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
                
                # six new EMSs in 3 dimensionsc
                x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
                x3, y3, z3 = ems[0]; x4, y4, z4 = ems[1]
                new_EMSs = [
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
                ]
                

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True
                    
                    if verbose:
                        print('\n*New*\nEMS:', list(map(tuple, new_EMS)))

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                print('-> Totally inscribed by:', list(map(tuple, other_EMS)))
                            
                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.min(new_box) < min_dim:
                        isValid = False
                        if verbose:
                            print('-> Dimension too small.')
                        
                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.prod(new_box) < min_vol:
                        isValid = False
                        if verbose:
                            print('-> Volumne too small.')

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            print('-> Success\nAdd new EMS:', list(map(tuple, new_EMS)))

        if verbose:
            print('\nEnd:')
            print('EMSs:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
    
    def overlapped(self, ems, EMS):
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False
    
    def inscribed(self, ems, EMS):
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False
    
    def eliminate(self, ems):
        # numpy array can't compare directly
        ems = list(map(tuple, ems))    
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return
    
    def get_EMSs(self):
        return  list(map( lambda x : list(map(tuple,x)), self.EMSs))

    def gravity_ok(self, ems_min, box_dims):
        if self.gravity_strength <= 0.0:
            return True
        x, y, z = int(ems_min[0]), int(ems_min[1]), int(ems_min[2])
        dx, dy = int(box_dims[0]), int(box_dims[1])
        if z == 0:
            return True
        supported = 0
        for item in self.load_items:
            if int(item[1][2]) != z:
                continue
            ix0, iy0 = int(item[0][0]), int(item[0][1])
            ix1, iy1 = int(item[1][0]), int(item[1][1])
            ox = max(0, min(x + dx, ix1) - max(x, ix0))
            oy = max(0, min(y + dy, iy1) - max(y, iy0))
            supported += ox * oy
        return supported >= self.gravity_strength * dx * dy - 0.001

    def load(self):
        return np.sum([ np.prod(item[1] - item[0]) for item in self.load_items]) / np.prod(self.dimensions)
    
class PlacementProcedure():
    def __init__(self, inputs, solution, verbose=False):
        n_bins = len(inputs['V'])
        n_items = len(inputs['v'])
        max_values       = inputs.get('max_values',       [float('inf')] * n_bins)
        gravity_strengths = inputs.get('gravity_strengths', [0.0]          * n_bins)
        self.Bins = [
            Bin(V, max_value=max_values[i], gravity_strength=gravity_strengths[i])
            for i, V in enumerate(inputs['V'])
        ]
        self.boxes = inputs['v']
        self.allowed_rotations = inputs.get('allowed_rotations', [[1,2,3,4,5,6]] * n_items)
        self.item_values       = inputs.get('item_values',       [0.0]           * n_items)
        self.BPS = np.argsort(solution[:n_items])
        self.VBO = solution[n_items:]
        self.num_opend_bins = 1
        
        self.verbose = verbose
        if self.verbose:
            print('------------------------------------------------------------------')
            print('|   Placement Procedure')
            print('|    -> Boxes:', self.boxes)
            print('|    -> Box Packing Sequence:', self.BPS)
            print('|    -> Vector of Box Orientations:', self.VBO)
            print('-------------------------------------------------------------------')
        
        self.infisible = False
        self.placement()
        
    
    def placement(self):
        items_sorted   = [self.boxes[i]            for i in self.BPS]
        allowed_sorted = [self.allowed_rotations[i] for i in self.BPS]
        values_sorted  = [self.item_values[i]       for i in self.BPS]

        for i, box in enumerate(items_sorted):
            if self.verbose:
                print('Select Box:', box)

            allowed    = allowed_sorted[i]
            item_value = values_sorted[i]

            selected_bin = None
            selected_EMS = None
            for k in range(self.num_opend_bins):
                if self.Bins[k].current_value + item_value > self.Bins[k].max_value:
                    continue
                EMS = self.DFTRC_2(box, k, allowed_directions=allowed)
                if EMS is not None:
                    selected_bin = k
                    selected_EMS = EMS
                    break

            if selected_bin is None:
                self.num_opend_bins += 1
                selected_bin = self.num_opend_bins - 1
                if self.num_opend_bins > len(self.Bins):
                    self.infisible = True
                    if self.verbose:
                        print('No more bin to open. [Infeasible]')
                    return
                selected_EMS = self.Bins[selected_bin].EMSs[0]
                if self.verbose:
                    print('No available bin... open bin', selected_bin)

            if self.verbose:
                print('Select EMS:', list(map(tuple, selected_EMS)))

            BO = self.selecte_box_orientaion(
                self.VBO[i], box, selected_EMS,
                allowed_directions=allowed, bin_idx=selected_bin
            )
            min_vol, min_dim = self.elimination_rule(items_sorted[i+1:])
            self.Bins[selected_bin].current_value += item_value
            self.Bins[selected_bin].update(self.orient(box, BO), selected_EMS, min_vol, min_dim)

            if self.verbose:
                print('Add box to Bin', selected_bin)
                print(' -> EMSs:', self.Bins[selected_bin].get_EMSs())
                print('------------------------------------------------------------')
        if self.verbose:
            print('|')
            print('|     Number of used bins:', self.num_opend_bins)
            print('|')
            print('------------------------------------------------------------')
    
    # Distance to the Front-Top-Right Corner
    def DFTRC_2(self, box, k, allowed_directions=None):
        if allowed_directions is None:
            allowed_directions = [1, 2, 3, 4, 5, 6]
        maxDist = -1
        selectedEMS = None

        for EMS in self.Bins[k].EMSs:
            D, W, H = self.Bins[k].dimensions
            for direction in allowed_directions:
                d, w, h = self.orient(box, direction)
                if self.fitin((d, w, h), EMS):
                    if not self.Bins[k].gravity_ok(EMS[0], (d, w, h)):
                        continue
                    x, y, z = EMS[0]
                    distance = pow(D-x-d, 2) + pow(W-y-w, 2) + pow(H-z-h, 2)
                    if distance > maxDist:
                        maxDist = distance
                        selectedEMS = EMS
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

        # Gravity may be too strict; fall back to any fitting allowed direction
        if not BOs:
            for direction in allowed_directions:
                if self.fitin(self.orient(box, direction), EMS):
                    BOs.append(direction)

        selectedBO = BOs[math.ceil(VBO * len(BOs)) - 1]
        if self.verbose:
            print('Select VBO:', selectedBO, '  (BOs', BOs, ', vector', VBO, ')')
        return selectedBO
    
    def fitin(self, box, EMS):
        # all dimension fit
        for d in range(3):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True
    
    def elimination_rule(self, remaining_boxes):
        if len(remaining_boxes) == 0:
            return 0, 0
        
        min_vol = 999999999
        min_dim = 9999
        for box in remaining_boxes:
            # minimum dimension
            dim = np.min(box)
            if dim < min_dim:
                min_dim = dim
                
            # minimum volume
            vol = np.prod(box)
            if vol < min_vol:
                min_vol = vol
        return min_vol, min_dim
    
    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        
        leastLoad = 1
        for k in range(self.num_opend_bins):
            load = self.Bins[k].load()
            if load < leastLoad:
                leastLoad = load
        return self.num_opend_bins + leastLoad%1
    


class BRKGA():
    def __init__(self, inputs, num_generations = 200, num_individuals=120, num_elites = 12, num_mutants = 18, eliteCProb = 0.7, multiProcess = False):
        # Setting
        self.multiProcess = multiProcess
        # Input
        self.inputs =  copy.deepcopy(inputs)
        self.N = len(inputs['v'])
        
        # Configuration
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        
        # Result
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {
            'mean': [],
            'min': []
        }
        
    def decoder(self, solution):
        placement = PlacementProcedure(self.inputs, solution)
        return placement.evaluate()
    
    def cal_fitness(self, population):
        fitness_list = list()

        for solution in population:
            decoder = PlacementProcedure(self.inputs, solution)
            fitness_list.append(decoder.evaluate())
        return fitness_list

    def partition(self, population, fitness_list):
        fitness_list = np.array(fitness_list)
        sorted_indexs = np.argsort(fitness_list)
        return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]], fitness_list[sorted_indexs[:self.num_elites]]
    
    def crossover(self, elite, non_elite):
        # chance to choose the gene from elite and non_elite for each gene
        return [elite[gene] if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb else non_elite[gene] for gene in range(self.num_gene)]
    
    def mating(self, elites, non_elites):
        # biased selection of mating parents: 1 elite & 1 non_elite
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return [self.crossover(random.choice(elites), random.choice(non_elites)) for i in range(num_offspring)]
    
    def mutants(self):
        return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))
        
    def fit(self, patient = 4, verbose = False):
        # Initial population & fitness
        population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
        fitness_list = self.cal_fitness(population)
        
        if verbose:
            print('\nInitial Population:')
            print('  ->  shape:',population.shape)
            print('  ->  Best Fitness:',max(fitness_list))
            
        # best    
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))
        
        
        # Repeat generations
        best_iter = 0
        for g in range(self.num_generations):

            # early stopping
            if g - best_iter > patient:
                self.used_bins = math.floor(best_fitness)
                self.best_fitness = best_fitness
                self.solution = best_solution
                if verbose:
                    print('Early stop at iter', g, '(timeout)')
                return 'feasible'
            
            # Select elite group
            elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
            
            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)
            
            # Generate mutants
            mutants = self.mutants()

            # New Population & fitness
            offspring = np.concatenate((mutants,offsprings), axis=0)
            offspring_fitness_list = self.cal_fitness(offspring)
            
            population = np.concatenate((elites, offspring), axis=0)
            fitness_list = list(elite_fitness_list) + list(offspring_fitness_list)
            
            # Update Best Fitness
            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]
            
            self.history['min'].append(np.min(fitness_list))
            self.history['mean'].append(np.mean(fitness_list))
            
            if verbose:
                print("Generation :", g, ' \t(Best Fitness:', best_fitness,')')
            
        self.used_bins = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution
        return 'feasible'


class _TrackedPlacement(PlacementProcedure):
    """PlacementProcedure subclass that records per-item assignments."""

    def __init__(self, inputs, solution):
        self.assignments = []  # populated by overridden placement()
        super().__init__(inputs, solution)

    def placement(self):
        items_sorted   = [self.boxes[i]             for i in self.BPS]
        allowed_sorted = [self.allowed_rotations[i] for i in self.BPS]
        values_sorted  = [self.item_values[i]       for i in self.BPS]

        for i, box in enumerate(items_sorted):
            allowed    = allowed_sorted[i]
            item_value = values_sorted[i]

            selected_bin = None
            selected_EMS = None
            for k in range(self.num_opend_bins):
                if self.Bins[k].current_value + item_value > self.Bins[k].max_value:
                    continue
                EMS = self.DFTRC_2(box, k, allowed_directions=allowed)
                if EMS is not None:
                    selected_bin = k
                    selected_EMS = EMS
                    break

            if selected_bin is None:
                self.num_opend_bins += 1
                selected_bin = self.num_opend_bins - 1
                if self.num_opend_bins > len(self.Bins):
                    self.infisible = True
                    return
                selected_EMS = self.Bins[selected_bin].EMSs[0]

            BO = self.selecte_box_orientaion(
                self.VBO[i], box, selected_EMS,
                allowed_directions=allowed, bin_idx=selected_bin
            )
            placed_box = self.orient(box, BO)
            min_vol, min_dim = self.elimination_rule(items_sorted[i + 1:])
            self.Bins[selected_bin].current_value += item_value
            self.Bins[selected_bin].update(placed_box, selected_EMS, min_vol, min_dim)

            x, y, z = selected_EMS[0]
            self.assignments.append({
                'original_item_idx': int(self.BPS[i]),
                'bin_idx': selected_bin,
                'BO': BO,
                'x': int(x), 'y': int(y), 'z': int(z),
            })


class solver_338874_2(_SolverBase):
    """Hybrid BRKGA + VND solver.

    Initialisation: BRKGA decodes a chromosome into a placement via
    _TrackedPlacement, respecting allowedRotations, value caps and
    gravityStrength.

    Improvement: the decoded placement is converted to the active_bins dict
    format and handed to solver_338874's full VND loop + post-processors
    (_vnd_loop, _post_process_repack_partial, _post_process_downgrade),
    all inherited unchanged.

    Orientation mapping (items stored as (depth, width, height)):
        BRKGA BO  →  env orient
          1  →  0
          2  →  2
          3  →  1
          4  →  5
          5  →  3
          6  →  4
    """

    _BO_TO_ENV = {1: 0, 2: 2, 3: 1, 4: 5, 5: 3, 6: 4}
    _ENV_TO_BO = {0: 1, 1: 3, 2: 2, 3: 5, 4: 6, 5: 4}

    def __init__(self, inst):
        super().__init__(inst)
        self.name = "solver_338874_2"

    # ------------------------------------------------------------------
    # 1. Build BRKGA inputs from Instance DataFrames
    # ------------------------------------------------------------------
    def _build_brkga_inputs(self, items, vehicles):
        n_items = len(items)
        v = [(int(r['depth']), int(r['width']), int(r['height'])) for _, r in items.iterrows()]

        allowed_rotations = [
            [self._ENV_TO_BO[int(c)] for c in str(r['allowedRotations']) if c.isdigit()]
            or [1, 2, 3, 4, 5, 6]
            for _, r in items.iterrows()
        ]
        item_values = [float(r.get('value', 0)) for _, r in items.iterrows()]

        V, slot_vtypes, max_values, grav_strengths = [], [], [], []
        for vt, vr in vehicles.iterrows():
            mv = float(vr.get('maxValue', float('inf')))
            gs = float(vr.get('gravityStrength', 0)) / 100.0
            for _ in range(n_items):
                V.append((int(vr['depth']), int(vr['width']), int(vr['height'])))
                slot_vtypes.append(vt)
                max_values.append(mv)
                grav_strengths.append(gs)

        inputs = {
            'v': v, 'V': V,
            'allowed_rotations': allowed_rotations,
            'item_values':        item_values,
            'max_values':         max_values,
            'gravity_strengths':  grav_strengths,
        }
        return inputs, slot_vtypes

    # ------------------------------------------------------------------
    # 2. Convert _TrackedPlacement → active_bins (solver_338874 format)
    # ------------------------------------------------------------------
    def _placement_to_active_bins(self, placement, inputs, slot_vtypes, vehicles_dict, items):
        item_ids = list(items.index)

        by_bin = {}
        for asgn in placement.assignments:
            by_bin.setdefault(asgn['bin_idx'], []).append(asgn)

        type_cnt = {}
        active_bins = []
        for b_idx in sorted(by_bin):
            vt       = slot_vtypes[b_idx]
            v_model  = vehicles_dict[vt]
            idx      = type_cnt.get(vt, 0)
            type_cnt[vt] = idx + 1

            ab = {
                'type': vt, 'idx': idx,
                'spaces': self._make_spaces(
                    0, 0, 0,
                    int(v_model['depth']), int(v_model['width']), int(v_model['height'])
                ),
                'current_weight': 0.0, 'max_weight':     float(v_model['maxWeight']),
                'current_vol':    0.0, 'max_vol':         float(v_model['volume']),
                'current_value':  0.0, 'max_value_item':  float(v_model.get('maxValue', float('inf'))),
                'z_layers': {}, 'items': [], 'last_pruned': 1,
            }

            # Process items in z-order so _update_ems_spaces is correct
            for asgn in sorted(by_bin[b_idx], key=lambda a: a['z']):
                orig_idx  = asgn['original_item_idx']
                i_idx     = item_ids[orig_idx]
                item_row  = items.loc[i_idx]
                dx, dy, dz = _bo_orient(inputs['v'][orig_idx], asgn['BO'])
                r         = self._BO_TO_ENV[asgn['BO']]
                x, y, z   = asgn['x'], asgn['y'], asgn['z']
                w         = float(item_row['weight'])

                ab['current_weight'] += w
                ab['current_vol']    += dx * dy * dz
                ab['current_value']  += float(item_row.get('value', 0))
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
    # 3. Main solve: BRKGA init → inherited VND + post-processing
    # ------------------------------------------------------------------
    def solve(self, num_generations=200, num_individuals=120, patient=4,
              time_limit=600, **kwargs):
        start = _time.time()

        items    = self.inst.df_items.copy()
        vehicles = self.inst.df_vehicles.copy()

        # Derived columns required by VND helper methods
        items['volume']    = items['width'] * items['depth'] * items['height']
        items['p_i']       = items.apply(self._decide_priority_level, axis=1)
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles_dict      = vehicles.to_dict('index')

        inputs, slot_vtypes = self._build_brkga_inputs(items, vehicles)

        brkga = BRKGA(inputs, num_generations=num_generations, num_individuals=num_individuals)
        brkga.fit(patient=patient)

        if brkga.solution is None:
            return

        placement = _TrackedPlacement(inputs, brkga.solution)
        n_items   = len(items)
        if placement.infisible or len(placement.assignments) < n_items:
            print(f"[{self.name}] BRKGA infeasible "
                  f"({len(placement.assignments)}/{n_items} items placed)")
            return

        active_bins = self._placement_to_active_bins(
            placement, inputs, slot_vtypes, vehicles_dict, items
        )
        init_cost = self._cost_from_bins(active_bins, vehicles_dict, items)
        print(f"[{self.name}] BRKGA init: {len(active_bins)} bins, cost {init_cost:.2f}")

        # VND improvement (inherited from solver_338874)
        sol = self._rebuild_sol_from_bins(active_bins)
        optimized_bins, _ = self._vnd_loop(sol, active_bins, vehicles_dict, items)

        # Post-processing (inherited from solver_338874)
        _, optimized_bins = self._post_process_repack_partial(
            optimized_bins, vehicles_dict, items
        )
        _, optimized_bins = self._post_process_downgrade(optimized_bins, vehicles_dict)

        final_cost = self._cost_from_bins(optimized_bins, vehicles_dict, items)
        print(f"[{self.name}] Final: {len(optimized_bins)} bins, "
              f"cost {final_cost:.2f} in {_time.time() - start:.1f}s")

        self.sol.clear()
        self.sol.update(self._rebuild_sol_from_bins(optimized_bins))
        self.active_bins = optimized_bins