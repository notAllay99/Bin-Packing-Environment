import numpy as np
import pandas as pd
from .abstract_solver import AbstractSolver

# Precomputed rotation index table: each row maps [w, d, h] to a permutation
# Row r gives the indices into [w, d, h] for rotation r
ROTATIONS = np.array([
    [0, 1, 2],  # (w, d, h)
    [1, 0, 2],  # (d, w, h)
    [2, 1, 0],  # (h, d, w)
    [1, 2, 0],  # (d, h, w)
    [0, 2, 1],  # (w, h, d)
    [2, 0, 1],  # (h, w, d)
], dtype=np.int32)


class ConstructiveSolver(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst)
        self.name = "ConstructivePhase"

    def _get_rotated_dims(self, item, r):
        """
        Returns (dx, dy, dz) for a given item and rotation index r.
        Uses the precomputed ROTATIONS table instead of rebuilding a list every call.
        Checker axis mapping: X = Depth, Y = Width, Z = Height.
        """
        dims = np.array([item['width'], item['depth'], item['height']], dtype=np.int32)
        w, d, h = dims[ROTATIONS[r]]   # permuted according to rotation
        return int(d), int(w), int(h)  # dx=depth, dy=width, dz=height

    # ------------------------------------------------------------------
    # Space helpers — spaces are stored as numpy matrices, shape (N, 6)
    # Columns: [x, y, z, len_x, len_y, len_z]
    # ------------------------------------------------------------------

    @staticmethod
    def _make_spaces(x, y, z, lx, ly, lz):
        """Create a (1, 6) numpy matrix representing a single space."""
        return np.array([[x, y, z, lx, ly, lz]], dtype=np.int32)

    @staticmethod
    def _split_space(space_row, dx, dy, dz):
        """
        Given a space row (1D array of 6 ints) and the item dimensions that
        were just packed into it, return a (K, 6) matrix of the resulting
        sub-spaces (only those with all positive dimensions are kept).
        """
        x, y, z, lx, ly, lz = space_row
        candidates = np.array([
            [x + dx, y,      z,      lx - dx, ly,      lz     ],
            [x,      y + dy, z,      dx,      ly - dy, lz     ],
            [x,      y,      z + dz, dx,      dy,      lz - dz],
        ], dtype=np.int32)
        # Keep only spaces where all three dimensions are positive
        valid = (candidates[:, 3] > 0) & (candidates[:, 4] > 0) & (candidates[:, 5] > 0)
        return candidates[valid]

    @staticmethod
    def _find_space(spaces, dx, dy, dz):
        """
        Find the index of the best fitting space for an item of size (dx, dy, dz).
        'Best' mirrors the original logic: first space in (z, x, y) order that fits.

        Returns -1 if no space fits.
        """
        if len(spaces) == 0:
            return -1

        # Vectorized fit check across all spaces in one C-level operation
        fits = (spaces[:, 3] >= dx) & (spaces[:, 4] >= dy) & (spaces[:, 5] >= dz)
        if not fits.any():
            return -1

        # Sort by (z, x, y) — lexsort reads keys right-to-left
        order = np.lexsort((spaces[:, 1], spaces[:, 0], spaces[:, 2]))

        # Walk the sorted order and return the first index that fits
        for idx in order:
            if fits[idx]:
                return int(idx)

        return -1  # should never reach here given fits.any() above

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self):
        vehicles = self.inst.df_vehicles.copy()
        vehicles['volume'] = vehicles['width'] * vehicles['depth'] * vehicles['height']
        vehicles = vehicles.sort_values(by='volume', ascending=False)

        items = self.inst.df_items.copy()
        items['volume'] = items['width'] * items['depth'] * items['height']
        items = items.sort_values(by='volume', ascending=False)

        active_bins = []
        vehicle_counter = 0

        for i_idx, item in items.iterrows():
            packed = False
            allowed_rots = [int(c) for c in str(item['allowedRotations']) if c.isdigit()]

            item_weight = float(item.get('weight', 0))
            item_value  = float(item.get('value',  0))

            # ---- ATTEMPT 1: try existing open bins (cheapest first) ----
            active_bins_sorted = sorted(
                active_bins,
                key=lambda b: vehicles.loc[b['type'], 'cost']
            )
            for active_bin in active_bins_sorted:
                if packed:
                    break

                if active_bin['current_weight'] + item_weight > active_bin['max_weight']:
                    continue
                if active_bin['current_value']  + item_value  > active_bin['max_value']:
                    continue

                spaces = active_bin['spaces']  # numpy (N, 6) matrix

                for r in allowed_rots:
                    dx, dy, dz = self._get_rotated_dims(item, r)

                    s_idx = self._find_space(spaces, dx, dy, dz)
                    if s_idx == -1:
                        continue

                    space_row = spaces[s_idx]
                    self._pack_item(i_idx, active_bin['type'], active_bin['idx'], space_row, r)

                    active_bin['current_weight'] += item_weight
                    active_bin['current_value']  += item_value

                    # Remove used space and add the up-to-3 sub-spaces
                    new_spaces = self._split_space(space_row, dx, dy, dz)
                    spaces = np.delete(spaces, s_idx, axis=0)
                    if len(new_spaces) > 0:
                        spaces = np.vstack([spaces, new_spaces])
                    active_bin['spaces'] = spaces

                    packed = True
                    break

            # ---- ATTEMPT 2: open a new vehicle ----
            if not packed:
                for v_type, v_model in vehicles.iterrows():
                    if packed:
                        break

                    if int(v_model['maxWeight']) < item_weight or int(v_model['maxValue']) < item_value:
                        continue

                    v_len_x = int(v_model['depth'])
                    v_len_y = int(v_model['width'])
                    v_len_z = int(v_model['height'])

                    for r in allowed_rots:
                        dx, dy, dz = self._get_rotated_dims(item, r)

                        if v_len_x >= dx and v_len_y >= dy and v_len_z >= dz:
                            # Initial space is the entire vehicle interior
                            initial_space = self._make_spaces(0, 0, 0, v_len_x, v_len_y, v_len_z)
                            space_row = initial_space[0]

                            new_bin = {
                                'type':           v_type,
                                'idx':            vehicle_counter,
                                'spaces':         initial_space,
                                'current_weight': item_weight,
                                'current_value':  item_value,
                                'max_weight':     float(v_model['maxWeight']),
                                'max_value':      float(v_model['maxValue']),
                            }

                            self._pack_item(i_idx, new_bin['type'], new_bin['idx'], space_row, r)

                            # Split the initial space and store result
                            new_spaces = self._split_space(space_row, dx, dy, dz)
                            new_bin['spaces'] = new_spaces if len(new_spaces) > 0 else np.empty((0, 6), dtype=np.int32)

                            active_bins.append(new_bin)
                            vehicle_counter += 1
                            packed = True
                            break

    # ------------------------------------------------------------------
    # Pack item — space_row is now a 1D numpy array, not a Space object
    # ------------------------------------------------------------------

    def _pack_item(self, i_idx, v_type, num_idx, space_row, orient):
        self.sol['type_vehicle'].append(v_type)
        self.sol['idx_vehicle'].append(int(num_idx))
        self.sol['id_item'].append(i_idx)
        self.sol['x_origin'].append(int(space_row[0]))  # x
        self.sol['y_origin'].append(int(space_row[1]))  # y
        self.sol['z_origin'].append(int(space_row[2]))  # z
        self.sol['orient'].append(int(orient))

class ImprovementSolver(AbstractSolver):
    def __init__(self, inst):
        pass