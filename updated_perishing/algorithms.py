import numpy as np
import helper

DEBUG = False


def fixed_threshold(size, resource_perish, max_budget, xopt, x_lower, n):
    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    for t in range(n):
        to_allocate = size[t] * x_lower
        alloc_tracker = 0
        
        for b in np.arange(current_index, max_budget):
            (frac, perish_time) = resource_dict[str(b)]
            current_index = b
            if perish_time >= t: # perishing in the future from this round
                alloc_amt = min(1 - frac, to_allocate - alloc_tracker)
                alloc_tracker += alloc_amt
                resource_dict[str(b)] = (frac+alloc_amt, perish_time)
            if alloc_tracker >= to_allocate:
                break

        if alloc_tracker < to_allocate: # run out of resources
            print('Out of resources')
            flag = True
            waste = 0
            counterfactual_envy = np.abs(xopt - alloc_tracker / size[t])
            hindsight_envy = np.abs(x_lower - alloc_tracker / size[t])

    if flag == False: # did not run out of resources
        hindsight_envy = 0
        counterfactual_envy = np.abs(xopt - x_lower)
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])
    perish_un_alloc_vec = []
    for b in range(max_budget):
        if resource_dict[str(b)][1] < n:
            perish_un_alloc_vec.append(1 - resource_dict[str(b)][0])
        else:
            perish_un_alloc_vec.append(0)
    perish_un_allocate = np.sum(perish_un_alloc_vec)
    
    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy


def hope_guardrail_perish(size, resource_perish, max_budget, xopt, x_lower, n, Lt, demand_dist, perish_dist, n_upper):

    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    
    x_upper = x_lower + Lt
    if DEBUG: print(f"X_lower: {x_lower}, X_upper: {x_upper}")
    current_budget = max_budget

    for t in range(n):
        if t != n-1:
            n_upper_future = n_upper[t+1]
            perish_future = helper.perish_future(t, current_index, resource_dict, x_lower, perish_dist, demand_dist, n, max_budget)        
        else:
            perish_future = 0
            n_upper_future = 0


        if DEBUG: print(f"t: {t}, current_budget: {current_budget}, n_upper: {n_upper_future}, perish_future {perish_future}")
        if DEBUG: print(f"Check without perish: {size[t]*x_upper + n_upper_future*x_lower}")

        if current_budget - size[t]*x_upper - n_upper_future*x_lower - perish_future >= 0:
            to_allocate = size[t] * x_upper
            if DEBUG: print(f"t: {t} allocating x_upper")
        else:
            to_allocate = size[t] * x_lower
        
        alloc_tracker = 0
        
        for b in np.arange(current_index, max_budget):
            (frac, perish_time) = resource_dict[str(b)]
            current_index = b
            if perish_time >= t: # perishing in the future from this round
                alloc_amt = min(1 - frac, to_allocate - alloc_tracker)
                alloc_tracker += alloc_amt
                resource_dict[str(b)] = (frac+alloc_amt, perish_time)
            if alloc_tracker >= to_allocate:
                break
                
        # Update current budget
        current_budget = max_budget - np.sum([1 if resource_dict[str(b)][1] <= t else resource_dict[str(b)][0] for b in range(max_budget)])
        if alloc_tracker < to_allocate: # run out of resources
            print(f'Out of resources at timestep: {t}')
            flag = True
            waste = 0
            counterfactual_envy = np.abs(xopt - alloc_tracker / size[t])
            hindsight_envy = np.abs(x_upper - alloc_tracker / size[t])

    if flag == False: # did not run out of resources
        hindsight_envy = np.abs(x_upper - x_lower)
        counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper)])
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_alloc_vec = []
    for b in range(max_budget):
        if resource_dict[str(b)][1] < n:
            perish_un_alloc_vec.append(1 - resource_dict[str(b)][0])
        else:
            perish_un_alloc_vec.append(0)
    perish_un_allocate = np.sum(perish_un_alloc_vec)
    
    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy

