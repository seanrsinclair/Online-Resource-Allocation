import cvxpy as cp
import numpy as np


# Get utility function for the various models


def get_individual_utility(weights, allocation, utility, epsilon = 0):
    if utility == 'linear':
        return np.dot(weights, allocation)
    elif utility == 'leontief':
        return np.min(allocation / weights)
    elif utility == 'filling':
        return np.min(np.clip(allocation / weights, 0, 1))
    elif utility == 'leontief_epsilon':
        return (1 - epsilon)*np.min(allocation / weights) + epsilon * np.sum(allocation)



def get_utility(weights, allocation, utility, epsilon):
    utils = []
    for typ in range(weights.shape[0]):
        utils.append(get_individual_utility(weights[typ], allocation[typ], utility, epsilon))
    return utils


def get_waste(weights, allocation, sizes, budget, utility):
    waste = 0
    new_budget = np.copy(budget)

    if utility == 'filling':
        for resource in range(budget.shape[0]):
            new_budget[resource] = min(new_budget[resource], np.dot(np.sum(sizes, axis=0), weights[:,resource]))
    
    for resource in range(budget.shape[0]):
        waste += new_budget[resource] - np.sum([np.dot(sizes[t,:], allocation[t,:,resource]) for t in range(sizes.shape[0])])
    return waste


def get_proportionality(weights, allocation, sizes, budget, utility, epsilon):
    proportionality = 0
    equal_allocation = budget / np.sum(sizes)
    for t in range(sizes.shape[0]):
        for typ in range(sizes.shape[1]):
            if get_individual_utility(weights[typ], allocation[t,typ], utility, epsilon) < get_individual_utility(weights[typ], equal_allocation, utility, epsilon):
                val = get_individual_utility(weights[typ], equal_allocation, utility, epsilon) - get_individual_utility(weights[typ], allocation[t,typ], utility, epsilon)
                proportionality = max(proportionality, val)
    return proportionality


def get_envy(weights, allocation, sizes, budget, utility, epsilon=0):
    envy = 0
    for t_1 in range(sizes.shape[0]):
        for typ_1 in range(sizes.shape[1]):
            for t_2 in range(sizes.shape[0]):
                for typ_2 in range(sizes.shape[1]):
                    if get_individual_utility(weights[typ_1], allocation[t_1,typ_1], utility, epsilon) < get_individual_utility(weights[typ_1], allocation[t_2, typ_2], utility, epsilon):
                        val = get_individual_utility(weights[typ_1], allocation[t_2, typ_2], utility, epsilon) - get_individual_utility(weights[typ_1], allocation[t_1, typ_1], utility, epsilon)
                        envy = max(envy, val)
    return envy



def get_counterfactual_envy(weights, allocation, sizes, budget, utility, opt, epsilon):
    envy = 0
    for t in range(sizes.shape[0]):
        for typ in range(sizes.shape[1]):
            val = get_individual_utility(weights[typ], allocation[t, typ], utility, epsilon) - get_individual_utility(weights[typ], opt[t, typ], utility, epsilon)
            envy = max(envy, np.abs(val))
    return envy


# Helper function to verify that the fairness definitions are satisfied
# with the relaxation that Pareto-Efficiency = Leftover Waste
def verify_fairness(weights, allocation, sizes, budget, utility, opt, epsilon=0):
    # assert (allocation <= leontief_weights).any(), "Allocation cannot be larger than the weights"
    waste = get_waste(weights, allocation, sizes, budget, utility)
    envy = get_envy(weights, allocation, sizes, budget, utility, epsilon)
    orig_envy = get_envy(weights, allocation, sizes, budget, 'leontief')
    proportionality = get_proportionality(weights, allocation, sizes, budget, utility, epsilon)
    c_envy = get_counterfactual_envy(weights, allocation, sizes, budget, utility, opt, epsilon)
    return waste, envy, proportionality, c_envy, orig_envy





def generate_cvxpy_solve(num_types, num_resources, utility):
    if utility == 'linear':
        return generate_cvxpy_solve_linear(num_types, num_resources)
    elif utility == 'leontief':
        return generate_cvxpy_solve_leontief(num_types, num_resources)
    elif utility == 'filling':
        return generate_cvxpy_solve_filling(num_types, num_resources)
    elif utility == 'leontief_epsilon':
        return generate_cvxpy_solve_leontief_epsilon(num_types, num_resources)
    


def generate_cvxpy_solve_linear(num_types, num_resources):
    # print(f'New Solver.  Num Types: {num_types}, Num Resources: {num_resources}')
    x = cp.Variable(shape=(num_types,num_resources))

    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)


    objective = cp.Maximize(cp.log(cp.sum(cp.multiply(x, weights), axis=1)) @ sizes)


    constraints = []
    constraints += [0 <= x]
    for i in range(num_resources):
        constraints += [x[:, i] @ sizes <= budget[i]]

    prob = cp.Problem(objective, constraints)
    
    def solver(true_sizes, true_weights, true_budget):
        sizes.value = true_sizes
        weights.value = true_weights
        budget.value = true_budget
        
        try:
            prob.solve()
        except:
            print('Sizes!')
            print(true_sizes)
            print('Weights!')
            print(true_weights)
            print('Budget!')
            print(true_budget)
            print('Solve failed retyring verbose')
            prob.solve(solver=cp.SCS)
        
        return prob.value, np.around(x.value, 7)
    
    return prob, solver



def generate_cvxpy_solve_leontief(num_types, num_resources):
    # print(f'New Solver.  Num Types: {num_types}, Num Resources: {num_resources}')
    x = cp.Variable(shape=(num_types,num_resources))

    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)


    objective = cp.Maximize(cp.log(cp.min(cp.multiply(x, 1 / weights), axis=1)) @ sizes)


    constraints = []
    constraints += [0 <= x]
    for i in range(num_resources):
        constraints += [x[:, i] @ sizes <= budget[i]]

    prob = cp.Problem(objective, constraints)
    
    def solver(true_sizes, true_weights, true_budget):
        sizes.value = true_sizes
        weights.value = true_weights
        budget.value = true_budget
        
        try:
            prob.solve()
        except:
            print('Sizes!')
            print(true_sizes)
            print('Weights!')
            print(true_weights)
            print('Budget!')
            print(true_budget)
            print('Solve failed retyring verbose')
            prob.solve(solver=cp.SCS)
        
        return prob.value, np.around(x.value, 7)
    
    return prob, solver


def generate_cvxpy_solve_leontief_epsilon(num_types, num_resources):
    # print(f'New Solver.  Num Types: {num_types}, Num Resources: {num_resources}')
    x = cp.Variable(shape=(num_types,num_resources))

    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)

    objective = cp.Maximize(cp.log(cp.min(cp.multiply(x, 1 / weights), axis=1) + cp.sum(x, axis=1)) @ sizes)
    constraints = []
    constraints += [0 <= x]
    for i in range(num_resources):
        constraints += [x[:, i] @ sizes <= budget[i]]

    prob = cp.Problem(objective, constraints)
    
    def solver(true_sizes, true_weights, true_budget, epsilon):
        sizes.value = true_sizes
        weights.value = true_weights
        budget.value = true_budget
        objective = cp.Maximize(cp.log((1 - epsilon)*cp.min(cp.multiply(x, 1 / weights), axis=1) + epsilon * cp.sum(x, axis=1)) @ sizes)
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
        except:
            print('Sizes!')
            print(true_sizes)
            print('Weights!')
            print(true_weights)
            print('Budget!')
            print(true_budget)
            print('Solve failed retyring verbose')
            prob.solve(solver=cp.SCS)
        
        return prob.value, np.around(x.value, 7)
    
    return prob, solver





def generate_cvxpy_solve_filling(num_types, num_resources):
    # print(f'New Solver.  Num Types: {num_types}, Num Resources: {num_resources}')
    x = cp.Variable(shape=(num_types,num_resources))

    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)


    objective = cp.Maximize(cp.log(cp.min(cp.multiply(x, 1 / weights), axis=1)) @ sizes)


    constraints = []
    constraints += [0 <= x]
    for i in range(num_resources):
        constraints += [x[:, i] @ sizes <= budget[i]]
    constraints += [x <= weights]
    prob = cp.Problem(objective, constraints)
    
    def solver(true_sizes, true_weights, true_budget):
        sizes.value = true_sizes
        weights.value = true_weights
        budget.value = true_budget
        
        try:
            prob.solve()
        except:
            print('Sizes!')
            print(true_sizes)
            print('Weights!')
            print(true_weights)
            print('Budget!')
            print(true_budget)
            print('Solve failed retyring verbose')
            prob.solve(solver=cp.SCS)
        
        return prob.value, np.around(x.value, 7)
    
    return prob, solver








def offline_opt(budget, size, weights, solver, epsilon = 0):
    tot_size = np.sum(size, axis=0)
    _, x = solver(tot_size, weights, budget, epsilon)
    allocation = np.zeros((size.shape[0], weights.shape[0], weights.shape[1]))
    for i in range(size.shape[0]):
        allocation[i,:,:] = x
    return allocation




def hope_online(budget, size, mean, stdev, weights, solver, epsilon = 0):
    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)
    for i in range(num_locations):
        cur_size = size[i] + np.sum(mean[i+1:], axis=0)
        _, sol = solver(cur_size, weights, budget_remaining, epsilon)
        resource_index = budget_remaining - np.matmul(size[i,:], sol) > 0
        allocation[i, :, :] = resource_index * sol + (1 - resource_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)
        budget_remaining -= np.matmul(size[i,:], allocation[i])
    return allocation, budget_remaining



def hope_full(budget, size, mean, stdev, weights, solver, epsilon=0):
    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)
    for i in range(num_locations):
        cur_size = np.sum(size[:i+1],axis=0) + np.sum(mean[i+1:], axis=0)
        _, sol = solver(cur_size, weights, budget, epsilon)
        resource_index = budget_remaining - np.matmul(size[i,:], sol) > 0
        allocation[i, :, :] = resource_index * sol + (1 - resource_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)
        budget_remaining -= np.matmul(size[i,:], allocation[i])    
    return allocation, budget_remaining






def fixed_threshold(budget, size, mean, stdev, weights, solver, epsilon=0):
    
    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)


    
    
    future_size = size[0] + np.sum(mean[1:], axis=0)
    conf = np.max(np.sqrt(2*np.max(stdev, axis=0)*np.mean(mean, axis=0)*num_locations) / future_size)
    lower_exp_size = future_size*(1 + conf)
    _, lower_thresh = solver(lower_exp_size, weights, budget, epsilon) 
    
    for i in range(num_locations):

        resource_index = budget_remaining - np.matmul(size[i,:], lower_thresh) > 0
        
        allocation[i, :, :] = resource_index * lower_thresh + (1 - resource_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)
        
        budget_remaining -= np.matmul(size[i, :], allocation[i])
    
    
    return allocation, budget_remaining







def hope_guardrail(budget, size, mean, stdev, weights, solver, Lt, epsilon=0):

    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)


    future_size = size[0] + np.sum(mean[1:], axis=0)
        
    conf = np.max(np.sqrt(np.max(stdev, axis=0)*np.mean(mean, axis=0)*num_locations) / future_size)
    lower_exp_size = future_size*(1 + conf)
    _, lower_thresh = solver(lower_exp_size, weights, budget, epsilon) 
    
    c = (1 / (num_locations**(Lt)))*(1 +  conf) - conf

    upper_exp_size = future_size*(1 - c)
    _, upper_thresh = solver(upper_exp_size, weights, budget, epsilon)
    
    for i in range(num_locations):
        rem = num_locations - i
        conf_bnd = np.sqrt(np.max(stdev, axis=0)*np.mean(mean, axis=0)*rem)
        
        budget_required = budget_remaining - np.matmul(size[i, :], upper_thresh) - np.matmul(np.sum(mean[(i+1):, :], axis=0) + conf_bnd, lower_thresh) > 0
        budget_index = budget_remaining - np.matmul(size[i,:], lower_thresh) > 0
        
        
        allocation[i, :, :] = budget_required * upper_thresh \
                + (1 - budget_required)*budget_index*lower_thresh \
                + (1 - budget_required) * (1 - budget_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)

        budget_remaining -= np.matmul(size[i, :], allocation[i])
    
    return allocation, budget_remaining