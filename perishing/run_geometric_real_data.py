import warnings;
warnings.filterwarnings('ignore');

from time import sleep
from tqdm.auto import tqdm

import sys
import importlib
import numpy as np
import nbformat
# import plotly.express
# import plotly.express as px
import pandas as pd
import cvxpy as cp
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import seaborn as sns
import helper
import algorithms


np.random.seed(10)

num_iterations = 100

FILTER_OE = False


# algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_12', 'og_hope_guardrail_12']
# algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_35', 'og_hope_guardrail_35']

algo_list = ['static_b_over_n']

# algo_list = ['static_x_lower']

file_name = "geometric_perishing_2_real_data_"
print(f'Running for: {file_name}')

GINGER = True


if GINGER:
    p_val = 0.002239726027
    mean_size = 3.250342859
    var_size = 1.8487007
else:
    p_val = 0.034528751
    mean_size = 16.24037669
    var_size = 10.51764764


def perish_dist(b, n):
    val = np.minimum(n,np.random.geometric(p = p_val))
    return val

def demand_dist(n, mean_size, var_size=.1):
    size = np.maximum(0, np.random.normal(loc=mean_size, scale=np.sqrt(var_size), size=n))
    return size

# num_groups = np.logspace(2, 11, base=1.5, num=10).astype(int)
num_groups = [365]
# num_groups = np.logspace(2, 12, base=1.5, num=20).astype(int)
data = []

n = 365
max_budget = int(n * mean_size)
# max_budget = 12685

order = np.arange(0,max_budget,1)

# print(f'Num Locations: {n}, Max Budget: {max_budget}')
offset_prob = helper.check_offset_expiry(perish_dist, lambda n: demand_dist(n, mean_size, var_size), n, max_budget)
print(f' Probability process is offset expiring: {100*offset_prob}')

# CALCULATES
n_upper = helper.n_upper(lambda n: demand_dist(n, mean_size, var_size), n)

num_valid = 0    

x_lower_perish = helper.x_lower_line_search(perish_dist, lambda n: demand_dist(n, mean_size, var_size), n, max_budget, n_upper, order)
x_lower_no_perish = (max_budget / n_upper[0])

dperish = (max_budget / n_upper[0]) - x_lower_perish
print(f'Necessary X_lower due to perishing: {x_lower_perish}, B/N: {x_lower_no_perish}, and dperish: {dperish}')

while num_valid < num_iterations:
    print(f'Iteration: {num_valid}')
    demands = demand_dist(n, mean_size, var_size)
    resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)])
    # print(resource_perish)
    check_optimality = [(max_budget / np.sum(demands))*np.sum(demands[:(t+1)])
                            - np.count_nonzero([resource_perish <= t]) for t in range(n)]
    if FILTER_OE:
        if np.min(check_optimality) < 0: # checks if B/N is feasible in hindsight
            continue
        else:
            num_valid += 1
    else:
        num_valid += 1


    xopt = max_budget / np.sum(demands)
    print(f"Xopt: {xopt}")

    # print(f"Minimum L_T to get non trivial bound: {xopt - x_lower_perish} versus: {mean_size * (n**(-1/3))}")
    # Should also check feasibility of x_lower on this sample path that was sampled?
    
    for algo in algo_list:
        # print(algo)
        if algo == 'static_x_lower':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.fixed_threshold(demands, resource_perish, max_budget, xopt, x_lower_perish, n, order)
        elif algo == 'static_b_over_n':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.fixed_threshold(demands, resource_perish, max_budget, xopt, x_lower_no_perish, n, order)
        elif algo == 'hope_guardrail_12':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_perish(demands, resource_perish, max_budget, xopt, x_lower_perish, n, mean_size*(n**(-1/2)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
        elif algo == 'hope_guardrail_13':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_perish(demands, resource_perish, max_budget, xopt, x_lower_perish, n, mean_size*(n**(-1/3)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
        elif algo == 'hope_guardrail_14':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_perish(demands, resource_perish, max_budget, xopt, x_lower_perish, n, mean_size*(n**(-1/4)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
        elif algo == 'hope_guardrail_35':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_perish(demands, resource_perish, max_budget, xopt, x_lower_perish, n, mean_size*(n**(-0.35)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
        elif algo == 'og_hope_guardrail_35':
                perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_original(demands, resource_perish, max_budget, xopt, x_lower_no_perish, n, mean_size*(n**(-0.35)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)                   
        elif algo == 'og_hope_guardrail_12':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_original(demands, resource_perish, max_budget, xopt, x_lower_no_perish, n, mean_size*(n**(-1/2)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
        elif algo == 'og_hope_guardrail_13':
            perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_original(demands, resource_perish, max_budget, xopt, x_lower_no_perish, n, mean_size*(n**(-1/3)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)

        data_dict = {'NumGroups': n, 'Algorithm': algo, 'Norm': 'Hindsight_Envy', 'Value': hindsight_envy}
        data.append(data_dict)
        data_dict = {'NumGroups': n, 'Algorithm': algo, 'Norm': 'Counterfactual_Envy', 'Value': counterfactual_envy}
        data.append(data_dict)
        data_dict = {'NumGroups': n, 'Algorithm': algo, 'Norm': 'Waste', 'Value': waste}
        data.append(data_dict)
        data_dict = {'NumGroups': n, 'Algorithm': algo, 'Norm': 'Perished_Un_Allocated', 'Value': perish_un_allocate}
        data.append(data_dict)
        data_dict = {'NumGroups': n, 'Algorithm': algo, 'Norm': 'Stockout', 'Value': stockout}
        data.append(data_dict)

df = pd.DataFrame.from_records(data)

df.to_csv('./data/'+file_name+'.csv', index=False)