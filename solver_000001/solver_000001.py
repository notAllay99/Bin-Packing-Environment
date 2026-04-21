import pandas as pd
import os
from .abstract_solver import AbstractSolver

class Space:
    def __init__(self, x, y, z, D, W, H):
        self.x, self.y, self.z = x, y, z
        self.D, self.W, self.H = D, W, H  # Profondità, Larghezza, Altezza

    def can_fit(self, item_d, item_w, item_h):
        return item_d <= self.D and item_w <= self.W and item_h <= self.H

class ConstructiveSolver(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst)
        self.name = "ConstructivePhase"

    def _get_rotated_dims(self, item, r):
        """Restituisce le dimensioni (d, w, h) in base all'indice di rotazione 0-5"""
        d, w, h = item['depth'], item['width'], item['height']
        rotations = {
            0: (d, w, h), 1: (w, d, h), 2: (d, h, w),
            3: (w, h, d), 4: (h, d, w), 5: (h, w, d)
        }
        return rotations.get(r, (d, w, h))

    def solve(self):
        # 1. Pre-processing: Ordinamento
        # Ora consideriamo vehicles come un CATALOGO di modelli disponibili
        vehicles = self.inst.df_vehicles.copy()
        vehicles['volume'] = vehicles['depth'] * vehicles['width'] * vehicles['height']
        vehicles = vehicles.sort_values(by='volume', ascending=False)

        items = self.inst.df_items.copy()
        items['volume'] = items['depth'] * items['width'] * items['height']
        items = items.sort_values(by=['volume'], ascending=False)

        # Inizializziamo la nostra FLOTTA ATTIVA (inizialmente vuota)
        active_bins = []
        vehicle_counter = 0  # Contatore numerico globale (0, 1, 2...)

        for i_idx, item in items.iterrows():
            packed = False
            allowed_rots = [int(r) for r in str(item['allowedRotations']).split(',')]

            # TENTATIVO 1: Cerchiamo spazio nei veicoli già aperti (nella flotta attiva)
            for active_bin in active_bins:
                if packed: break
                spaces = active_bin['spaces']
                
                for s_idx, space in enumerate(spaces):
                    if packed: break
                    
                    for r in allowed_rots:
                        id_d, id_w, id_h = self._get_rotated_dims(item, r)

                        if space.can_fit(id_d, id_w, id_h):
                            # Trovato! Registriamo la posizione
                            self._pack_item(i_idx, active_bin['type'], active_bin['idx'], space, id_d, id_w, id_h, r)
                            
                            # AGGIORNAMENTO SPAZI
                            new_spaces = [
                                Space(space.x + id_d, space.y, space.z, space.D - id_d, space.W, space.H),
                                Space(space.x, space.y + id_w, space.z, id_d, space.W - id_w, space.H),
                                Space(space.x, space.y, space.z + id_h, id_d, id_w, space.H - id_h)
                            ]
                            
                            spaces.pop(s_idx)
                            for ns in new_spaces:
                                if ns.D > 0 and ns.W > 0 and ns.H > 0:
                                    spaces.append(ns)
                            
                            packed = True
                            break

            # TENTATIVO 2: Se non entra in NESSUN veicolo aperto, COMPRIAMO un nuovo veicolo
            if not packed:
                for v_type, v_model in vehicles.iterrows():
                    if packed: break
                    
                    # Proviamo a vedere se l'oggetto entra in un veicolo completamente vuoto di questo modello
                    for r in allowed_rots:
                        id_d, id_w, id_h = self._get_rotated_dims(item, r)
                        
                        if v_model['depth'] >= id_d and v_model['width'] >= id_w and v_model['height'] >= id_h:
                            # Ottimo, l'oggetto entra! Generiamo un nuovo veicolo di questo modello
                            new_bin = {
                                'type': v_type,              # Es: "V0" o "V1"
                                'idx': vehicle_counter,      # Es: 0, 1, 2...
                                'spaces': [Space(0, 0, 0, v_model['depth'], v_model['width'], v_model['height'])]
                            }
                            
                            # Registriamo l'oggetto nel nuovo veicolo
                            space = new_bin['spaces'][0]
                            self._pack_item(i_idx, new_bin['type'], new_bin['idx'], space, id_d, id_w, id_h, r)
                            
                            # Aggiorniamo gli spazi per questo nuovo veicolo
                            new_spaces = [
                                Space(space.x + id_d, space.y, space.z, space.D - id_d, space.W, space.H),
                                Space(space.x, space.y + id_w, space.z, id_d, space.W - id_w, space.H),
                                Space(space.x, space.y, space.z + id_h, id_d, id_w, space.H - id_h)
                            ]
                            
                            new_bin['spaces'].pop(0)
                            for ns in new_spaces:
                                if ns.D > 0 and ns.W > 0 and ns.H > 0:
                                    new_bin['spaces'].append(ns)
                            
                            # Aggiungiamo il veicolo appena creato alla nostra flotta attiva e aggiorniamo il contatore
                            active_bins.append(new_bin)
                            vehicle_counter += 1
                            packed = True
                            break

            if not packed:
                print(f"⚠️ Item {i_idx} è troppo grande! Non entra fisicamente in nessun modello di veicolo vuoto!")

    def _pack_item(self, i_idx, v_type, num_idx, space, d, w, h, orient):
        """Salva i dati nel formato richiesto da AbstractSolver"""
        self.sol['type_vehicle'].append(v_type)
        self.sol['idx_vehicle'].append(num_idx)
        self.sol['id_item'].append(i_idx)
        self.sol['x_origin'].append(space.x)
        self.sol['y_origin'].append(space.y)
        self.sol['z_origin'].append(space.z)
        self.sol['orient'].append(orient)   