import os

# Import the Instance class as seen in your abstract_solver.py and additional_script.py
from instances.instance import Instance

# Import the GenpackSolver from your solver's module
from solver_338874_2.solver_338874_2 import solver_338874_2

from solver_338874_3.solver_338874_3 import solver_338874_3
def main():
    dataset_path = 'Dataset0'
    print(f"Loading instance from {dataset_path}...")
    inst = Instance(dataset_path)

    print("Initializing BRKGA Solver...")
    solver = solver_338874_3(inst)

    print("Solving...")
    #solver.solve(num_generations=200, num_individuals=120, patient=4)
    solver.solve()

    print(f"\nOptimization finished!")
    print(f"Bins used: {len(set(solver.sol['idx_vehicle']))}")

    os.makedirs('results', exist_ok=True)
    solver.write_solution_to_file()
    print(f"Solution saved to results/")

if __name__ == "__main__":
    main()