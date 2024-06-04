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
from itertools import permutations


num_iterations = 100

mean_size = 2
var_size = 0.1

FILTER_OE = False
DEBUG = False
EPS = 1

# algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_12', 'og_hope_guardrail_12']
algo_list = ['hope_guardrail_35']
# problem_list = ['A', 'B', 'C']
# problem_list = ['A']
# problem_list = ['A','B']
problem_list = ['C']
# order_list = ['mean', 'cv', 'random', 'reverse']
# order_list = ['mean', 'cv', 'lcb']
order_list = ['lcb', 'mean', 'cv']
# order_list = ['mean', 'cv', 'lcb', 'opt']
# order_list = ['lcb']
# order_list = ['opt']
# order_list = ['lcb']
# num_groups = [50]

# num_groups = np.logspace(4, 11, base=1.5, num=10).astype(int)
num_groups = [50]


def demand_dist(n, mean_size, var_size=.1):
    size = np.maximum(0, np.random.normal(loc=mean_size, scale=np.sqrt(var_size), size=n))
    return size



for setup in problem_list: # useless for loop at the moment, in case we want to run for different alpha variables

    file_name = "uniform_perishing_table_"+str(setup).replace('.','-')
    print(f'Running for: {file_name}')

    data = []

    for n in tqdm(num_groups):
        n = int(n)


        max_budget = mean_size*n
        

        if setup == 'A':
            EPS = 1
        elif setup == 'B':
            EPS = 4
        elif setup == 'C':
            EPS = 1


        if setup == 'A':
            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    mean = n / 2
                    d_range = EPS * b * (1 / mean_size)
                else:
                    mean = n
                    d_range = 0
                # print(mean, stdev)
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(mean - d_range, mean+d_range))))
                return val
            
            mean_list = np.asarray([n/2 if b < (mean_size*n / 2) else n for b in range(max_budget)])
            range_list = np.asarray([EPS * b * (1 / mean_size) if b < (mean_size*n / 2) else 0 for b in range(max_budget)])
            stdev_list = np.sqrt((1/3) * (range_list ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f'Range list: {range_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")


        elif setup == 'B':
            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    low_range = b+1
                    up_range = b+1
                else:
                    low_range = b+1
                    up_range = n
                # print(mean, stdev)
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range))))
                return val
            low_range = np.asarray([b+1 if b < (mean_size*n / 2) else b+1 for b in range(max_budget)])
            up_range = np.asarray([b+1 if b < (mean_size*n / 2) else n for b in range(max_budget)])
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f'Range list: {range_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")


        elif setup == 'C':
            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    low_range = b+1
                    up_range = b+1
                else:
                    low_range = (b+1 - (mean_size * n / 2))
                    up_range = (b+1 + (mean_size * n / 2))
                    mean = b+1
                # print(mean, stdev)
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range))))
                return val
            
            low_range = np.asarray([b+1 if b < (mean_size*n / 2) else (b+1 - (mean_size * n / 2)) for b in range(max_budget)])
            up_range = np.asarray([b+1 if b < (mean_size*n / 2) else (b+1 + (mean_size * n / 2)) for b in range(max_budget)])
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")

        elif setup == 'D':

            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    low_range = b+1
                    up_range = b+1
                else:
                    low_range = (b+1 - (mean_size * n / 2))
                    up_range = n
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range))))
                return val

            low_range = np.asarray([b+1 if b < (mean_size*n / 2) else (b+1 - (mean_size * n / 2)) for b in range(max_budget)])
            up_range = np.asarray([b+1 if b < (mean_size*n / 2) else n for b in range(max_budget)])
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")


        offset_prob = helper.check_offset_expiry(perish_dist, lambda n: demand_dist(n, mean_size, var_size), n, max_budget)
        print(f' Probability process is offset expiring: {100*offset_prob}')

        # CALCULATES
        n_upper = helper.n_upper(lambda n: demand_dist(n, mean_size, var_size), n)
        
        num_valid = 0    
        # print(max_budget)
        x_lower_no_perish = (max_budget / n_upper[0])
        print(f'X_lower_no_perishing: {x_lower_no_perish}')



        if 'opt' in order_list:
            max_x_lower_perish = 0
            for order in list(permutations(list(range(0, max_budget)))):
                # print(order)
                x_lower_perish = helper.x_lower_line_search(perish_dist, lambda n: demand_dist(n, mean_size, var_size), n, max_budget, n_upper, order)
                if x_lower_perish >= max_x_lower_perish:
                    opt_order = order
                    max_x_lower_perish = x_lower_perish
            print(f'Optimal Order: {opt_order} and x_lower: {max_x_lower_perish}')

        for _ in tqdm(range(num_iterations)):
            demands = demand_dist(n, mean_size, var_size)
            resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)])

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


            for algo in algo_list:
                for alloc_order in order_list:


                    if alloc_order == 'mean':
                        order = np.lexsort((np.random.random(mean_list.size), mean_list))
                    elif alloc_order == 'cv':
                        order = np.lexsort((np.random.random(mean_list.size),(-1)*CV))
                    elif alloc_order == 'lcb':
                        # lcb = np.maximum(mean_list - 1.96 * stdev_list, low_range)
                        lcb = low_range + .05*(up_range - low_range)
                        print(f' LCB Values: {lcb}')
                        order = np.lexsort((np.random.random(mean_list.size),lcb))
                        # print(lcb)

                    elif alloc_order == 'ucb':
                        # ucb = np.minimum(mean_list + 1.96 * stdev_list, up_range)
                        ucb = low_range + .05*(up_range - low_range)

                        order = np.lexsort((np.random.random(mean_list.size),ucb))
                    elif alloc_order == 'random':
                        order = np.random.permutation(max_budget)
                    elif alloc_order == 'flipped':
                        order = np.lexsort((np.random.random(mean_list.size),(-1)*mean_list))
                    elif alloc_order == 'opt':
                        order = opt_order


                    if alloc_order != 'opt':
                        x_lower_perish = helper.x_lower_line_search(perish_dist, lambda n: demand_dist(n, mean_size, var_size), n, max_budget, n_upper, order)
                    else:
                        x_lower_perish = max_x_lower_perish

                    if DEBUG: print(f"Alloc Order: {alloc_order} and order: {order} and x_lower: {x_lower_perish}")
                    if setup == 'A' and alloc_order == 'lcb':
                        x_lower_perish = x_lower_no_perish
                    elif setup == 'A' and alloc_order == 'cv':
                        x_lower_perish = x_lower_no_perish
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
                    elif algo == 'og_hope_guardrail_12':
                        perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_original(demands, resource_perish, max_budget, xopt, x_lower_no_perish, n, mean_size*(n**(-1/2)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
                    elif algo == 'og_hope_guardrail_13':
                        perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout = algorithms.hope_guardrail_original(demands, resource_perish, max_budget, xopt, x_lower_no_perish, n, mean_size*(n**(-1/3)), lambda n: demand_dist(n, mean_size, var_size), perish_dist, n_upper, order)
                    
                    data_dict = {'NumGroups': n, 'Algorithm': algo, 'Order': alloc_order, 'Norm': 'Delta_Perish', 'Value': x_lower_no_perish - x_lower_perish}
                    data.append(data_dict)
                    data_dict = {'NumGroups': n, 'Algorithm': algo, 'Order': alloc_order, 'Norm': 'Hindsight_Envy', 'Value': hindsight_envy}
                    data.append(data_dict)
                    data_dict = {'NumGroups': n, 'Algorithm': algo, 'Order': alloc_order, 'Norm': 'Counterfactual_Envy', 'Value': counterfactual_envy}
                    data.append(data_dict)
                    data_dict = {'NumGroups': n, 'Algorithm': algo, 'Order': alloc_order, 'Norm': 'Waste', 'Value': waste}
                    data.append(data_dict)
                    data_dict = {'NumGroups': n, 'Algorithm': algo, 'Order': alloc_order, 'Norm': 'Perished_Un_Allocated', 'Value': perish_un_allocate}
                    data.append(data_dict)
                    data_dict = {'NumGroups': n, 'Algorithm': algo, 'Order': alloc_order, 'Norm': 'Stockout', 'Value': stockout}
                    data.append(data_dict)

    # df = pd.DataFrame.from_records(data)
    # df.to_csv('./data/'+file_name+'.csv', index=False)