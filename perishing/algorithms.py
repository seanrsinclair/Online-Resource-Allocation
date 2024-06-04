import numpy as np
import helper

DEBUG = False


def fixed_threshold(size, resource_perish, max_budget, xopt, x_lower, n):
    """
        Fixed threshold allocation algorithm. Takes as input a fixed allocation value x_lower
        to allocate across all timesteps until running out of budget

        Returns:
            waste: B - \sum_t N_t X_t
            counterfactual envy: |xopt - x_t|
            hindsight envy: \max_t |X_t - X_t'|
            PUA: \sum_b fraction of b perished and un-allocated
    """

    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0


    for t in range(n):
        if flag == False:
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
                if DEBUG: print(f'FIXED THRESHOLD: Out of resources at timestep: {t}')
                flag = True
                waste = 0
                if t != n-1:
                    counterfactual_envy = xopt
                    hindsight_envy = x_lower
                else:    
                    counterfactual_envy = np.abs(xopt - alloc_tracker / size[t])
                    hindsight_envy = np.abs(x_lower - alloc_tracker / size[t])

    if flag == False: # did not run out of resources
        hindsight_envy = 0
        counterfactual_envy = np.abs(xopt - x_lower)
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_allocate = np.sum([0 if resource_dict[str(b)][1] < n else (1 - resource_dict[str(b)][0]) for b in range(max_budget)])

    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy


def hope_guardrail_perish(size, resource_perish, max_budget, xopt, x_lower, n, Lt, demand_dist, perish_dist, n_upper):
    """
        Perishing guardrail allocation algorithm. Takes as input a fixed allocation value x_lower
        as well as a value of L_t dictating X_upper that it oscilates between while allocating

        Returns:
            waste: B - \sum_t N_t X_t
            counterfactual envy: |xopt - x_t|
            hindsight envy: \max_t |X_t - X_t'|
            PUA: \sum_b fraction of b perished and un-allocated
    """

    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    
    x_upper = x_lower + Lt
    
    if DEBUG: print(f"X_opt: {xopt}, X_lower: {x_lower}, X_upper: {x_upper}")
    
    current_budget = max_budget

    for t in range(n):
        if flag == False: # only continue when the algorithm has not run out of resources

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
                if DEBUG: print(f'PERISH GUARDRAIL: Out of resources at timestep: {t} versus horizon {n}')
                flag = True
                waste = 0
                if t != n-1:
                    counterfactual_envy = xopt
                    hindsight_envy = x_upper
                else:    
                    counterfactual_envy = np.abs(xopt - alloc_tracker / size[t])
                    hindsight_envy = np.abs(x_upper - alloc_tracker / size[t])

    if flag == False: # did not run out of resources
        hindsight_envy = np.abs(x_upper - x_lower)
        counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper)])
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_allocate = np.sum([0 if resource_dict[str(b)][1] < n else (1 - resource_dict[str(b)][0]) for b in range(max_budget)])
    
    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy



def hope_guardrail_original(size, resource_perish, max_budget, xopt, x_lower, n, Lt, demand_dist, perish_dist, n_upper):
    """
        Original guardrail allocation algorithm. Takes as input a value of L_t dictating X_upper that it 
        oscilates between while allocating.

        NOTE: 1) X_lower will be B / N_upper, regardless of perishing
        2) Does not take into account perishing when checking to allocate x_upper.

        Returns:
            waste: B - \sum_t N_t X_t
            counterfactual envy: |xopt - x_t|
            hindsight envy: \max_t |X_t - X_t'|
            PUA: \sum_b fraction of b perished and un-allocated
    """

    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    
    x_lower = max_budget / n_upper[0]

    x_upper = x_lower + Lt
    if DEBUG: print(f"X_lower: {x_lower}, X_upper: {x_upper}")
    
    current_budget = max_budget

    for t in range(n):
        if flag == False:
                
            if t != n-1:
                n_upper_future = n_upper[t+1]
            else:
                n_upper_future = 0


            if DEBUG: print(f"t: {t}, current_budget: {current_budget}, n_upper: {n_upper_future}")
            if DEBUG: print(f"Check without perish: {size[t]*x_upper + n_upper_future*x_lower}")

            if current_budget - size[t]*x_upper - n_upper_future*x_lower >= 0:
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
                if DEBUG: print(f'OG GUARDRAIL: Out of resources at timestep: {t} versus horizon {n}')
                flag = True
                waste = 0
                if t != n-1:
                    counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper), np.abs(xopt)])
                    hindsight_envy = x_upper

                else:    
                    counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper), np.abs(xopt), np.max(xopt - (alloc_tracker / size[t]))])
                    hindsight_envy = np.max([np.abs(x_upper - (alloc_tracker / size[t])), np.abs(x_upper - x_lower)])
                    if DEBUG: print(f"To Allocate: {to_allocate}, Allocated: {alloc_tracker}")
                

    if flag == False: # did not run out of resources
        hindsight_envy = np.abs(x_upper - x_lower)
        counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper)])
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_allocate = np.sum([0 if resource_dict[str(b)][1] < n else (1 - resource_dict[str(b)][0]) for b in range(max_budget)])
    
    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy

