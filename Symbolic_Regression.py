import numpy as np
import pandas as pd
from pysr import PySRRegressor

df = pd.read_csv("./Opt_RS_descriptor.csv", index_col=0)

X = df.drop(df.columns[0:1], axis=1)
y = df["viscosity"]

for crossover_prob in np.arange(0.066, 0.067, 0.002): # 0.066
    for pop_size in range(100, 104, 20): # 33
        model = PySRRegressor(
            procs=20,
            #cluster_manager ="slurm",
            multithreading=True,

            populations=60, # 3*num_cores 144
            population_size=pop_size,
            ncycles_per_iteration=500,
            niterations=100, # 6000
            weight_optimize=0.001,
            batching=True,
            parsimony=0.0032, # 0.0001
            adaptive_parsimony_scaling=1000,
            warm_start=True,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "log"],
            constraints={
                "^": (9, 1),
                "/": (-1, 9),
                "square": 9,
                "cube": 9,
                "exp": 9,
            },
            nested_constraints={
                "exp": {"^": 0, "log": 0, "exp": 0},
                "log": {"^": 0, "log": 0, "exp": 0},
            },
            complexity_of_operators={"/": 2, "exp": 3},
            complexity_of_constants=2,
            warmup_maxsize_by=0.2,
            maxsize=45,
            crossover_probability=crossover_prob,
            tournament_selection_n=10,
            fraction_replaced=0.000364,
            fraction_replaced_hof=0.035,
            topn=12,
            early_stop_condition=(
                "stop_if(loss, complexity) = (loss < 0.5) && (complexity < 5)"
            ),
            extra_sympy_mappings={"inv": lambda x: 1 / x},
        )
        for i in range(500000):
            model.equation_file = f"result_{i+1}.csv"
            model.fit(X, y)
            print("**===========hyperparameter===========**")
            print(f"crossover_probability: {crossover_prob}, population_size: {pop_size}")
            print(model)
            print("**===========the best equation===========**")
            print(model.sympy())
