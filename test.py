import time

from instances.instance import Instance
from solver_338874.solver_338874 import solver_338874
from solver_338874.additional_script import ConstructiveSolver, calculate_theoretical_lower_bounds


if __name__ == '__main__':
    t_start = time.time()

    # 1. Carica l'istanza
    inst = Instance("DatasetA")
    '''
    bounds = calculate_theoretical_lower_bounds(inst.df_items, inst.df_vehicles)

    print(f"Il costo MINIMO teorico possibile è: {bounds['Absolute_Lower_Bound_Cost']}")
    '''
    print("Istanza caricata. Avvio solver costruttivo...")
    

    # 2. Inizializza ed esegui il Costruttivo
    #costruttore = ConstructiveSolver(inst)
    #starting_solution = costruttore.solve(return_detailed=False)
    
    #print("Costruttivo terminato. Avvio fase di miglioramento...")
    #starting_solution.write_solution_to_file()
    # 3. Inizializza l'ImprovementSolver (passando inst, come per il costruttore)
    imp = solver_338874(inst)
    imp.solve()
    imp.write_solution_to_file()
    
    t_end = time.time()
    print(f"Processo completato in {t_end - t_start:.2f} secondi.")
    