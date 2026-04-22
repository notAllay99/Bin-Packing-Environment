from instances.instance import Instance
# Importiamo ConstructiveSolver dalla cartella e dal file corretti
from solver_338874.solver_000001 import ImprovementSolver
from solver_338874.additional_script import ConstructiveSolver
from concurrent.futures import ProcessPoolExecutor
import time
# Questo usa 4 processi separati, ognuno su un core diverso
#with ProcessPoolExecutor(max_workers=4) as executor:
    # 1. Carica l'istanza
t = time.time()

inst = Instance("DatasetA")

# 2. Inizializza il risolutore che hai scritto in additional_script.py
# 1. Costruisci la base
costruttore = ConstructiveSolver(inst)
costruttore.solve()

# 2. Ottimizza
imp = ImprovementSolver(inst)
imp.solve_from_constructive(costruttore)
# 3. La tua soluzione finale perfezionata ora si trova in:
# ottimizzatore.sol
end = time.time()
imp.write_solution_to_file()
print(end -t)