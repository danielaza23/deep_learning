import json
import numpy as np
from pathlib import Path
from time import time

from functions import torch_fun_1, tf_fun_1, torch_fun_2, tf_fun_2
from solution import optimize_torch_fun1, optimize_torch_fun2, optimize_tf_fun1, optimize_tf_fun2

def score_fun(fun, optimization_fun, optimal_score, minimal_score_for_points, timeout):

    try:

        # run optimizer
        t_start = time()
        params = optimization_fun(fun)
        runtime = time() - t_start
        if runtime > timeout:
            print(f"Optimizer used {runtime}s, but only {timeout} are allowed")

        # determine performance
        funval = fun(params)
        if funval > minimal_score_for_points:
            return 0.0
        else:
            return np.round(float((100 * (minimal_score_for_points - funval) / (minimal_score_for_points - optimal_score))), 1)
        
    except Exception:
        raise

# run optimizer on first task
leaderboard_scores = []

scores = []
for name, obj_fun, optimizer_fun, least_possible_fun_val, fun_val_to_receive_points in [
    ("PyTorch Function 1", torch_fun_1, optimize_torch_fun1, 0, 100),
    ("PyTorch Function 2", torch_fun_2, optimize_torch_fun2, 0, 100),
    ("TensorFlow Function 1", tf_fun_1, optimize_tf_fun1, 0, 100),
    ("TensorFlow Function 2", tf_fun_2, optimize_tf_fun2, 0, 10)
]:
    score = score_fun(obj_fun, optimizer_fun, least_possible_fun_val, fun_val_to_receive_points, timeout=30)
    leaderboard_scores.append({
        "name": name,
        "value": score
    })
    scores.append(score)


# write results
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
with open(f'{folder}/results.json', 'w') as f:
    json.dump({
    "score": np.mean(scores),
    "stdout_visibility": "visible",
    "leaderboard": leaderboard_scores
    }, f)
