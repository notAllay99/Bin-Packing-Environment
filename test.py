from instances.instance import Instance
# Importiamo ConstructiveSolver dalla cartella e dal file corretti
from solver_000001.additional_script import ConstructiveSolver

from concurrent.futures import ProcessPoolExecutor
import time
# Questo usa 4 processi separati, ognuno su un core diverso
with ProcessPoolExecutor(max_workers=4) as executor:
    # 1. Carica l'istanza
    t = time.time()
    
    inst = Instance("DatasetC")

    # 2. Inizializza il risolutore che hai scritto in additional_script.py
    solver = ConstructiveSolver(inst)

    # 3. Risolvi e salva
    solver.solve()
    end = time.time()
    print(t - end)
    solver.write_solution_to_file()
   