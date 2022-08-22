import cvxpy as cp
import numpy as np

cthresh = 1
clow = np.sqrt(2)


def generate_cvxpy_solve(num_types, num_resources):
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
        
        return prob.value, np.around(x.value, 5)
    
    return prob, solver





def offline_opt(budget, size, weights, solver):
    tot_size = np.sum(size, axis=0)
    _, x = solver(tot_size, weights, budget)
    allocation = np.zeros((size.shape[0], weights.shape[0], weights.shape[1]))
    for i in range(size.shape[0]):
        allocation[i,:,:] = x
    return allocation




def hope_online(budget, size, mean, stdev, weights, solver):
    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)
    for i in range(num_locations):
        cur_size = size[i] + np.sum(mean[i+1:], axis=0)
        _, sol = solver(cur_size, weights, budget_remaining)
        resource_index = budget_remaining - np.matmul(size[i,:], sol) > 0
        allocation[i, :, :] = resource_index * sol + (1 - resource_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)
        budget_remaining -= np.matmul(size[i,:], allocation[i])
    return allocation, budget_remaining



def hope_full(budget, size, mean, stdev, weights, solver):
    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)
    for i in range(num_locations):
        cur_size = np.sum(size[:i+1],axis=0) + np.sum(mean[i+1:], axis=0)
        _, sol = solver(cur_size, weights, budget)
        resource_index = budget_remaining - np.matmul(size[i,:], sol) > 0
        allocation[i, :, :] = resource_index * sol + (1 - resource_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)
        budget_remaining -= np.matmul(size[i,:], allocation[i])    
    return allocation, budget_remaining






def fixed_threshold(budget, size, mean, stdev, weights, solver):
    
    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)


    
    
    future_size = size[0] + np.sum(mean[1:], axis=0)
    conf = np.max(np.sqrt(2*np.max(stdev, axis=0)*np.mean(mean, axis=0)*num_locations) / future_size)
    lower_exp_size = future_size*(1 + conf)
    _, lower_thresh = solver(lower_exp_size, weights, budget) 
    
    for i in range(num_locations):

        resource_index = budget_remaining - np.matmul(size[i,:], lower_thresh) > 0
        
        allocation[i, :, :] = resource_index * lower_thresh + (1 - resource_index) * np.array([budget_remaining / np.sum(size[i,:]),]*num_types)
        
        budget_remaining -= np.matmul(size[i, :], allocation[i])
    
    
    return allocation, budget_remaining







def hope_guardrail(budget, size, mean, stdev, weights, solver, Lt):

    num_locations = size.shape[0]
    num_types = weights.shape[0]
    num_resources = weights.shape[1]
    
    allocation = np.zeros((num_locations, num_types, num_resources))
    budget_remaining = np.copy(budget)


    future_size = size[0] + np.sum(mean[1:], axis=0)
        
    conf = np.max(np.sqrt(np.max(stdev, axis=0)*np.mean(mean, axis=0)*num_locations) / future_size)
    lower_exp_size = future_size*(1 + conf)
    _, lower_thresh = solver(lower_exp_size, weights, budget) 
    
    c = (1 / (num_locations**(Lt)))*(1 +  conf) - conf

    upper_exp_size = future_size*(1 - c)
    _, upper_thresh = solver(upper_exp_size, weights, budget)
    
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




