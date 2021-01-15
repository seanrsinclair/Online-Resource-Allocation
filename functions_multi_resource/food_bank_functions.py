#  Food Bank Problem

import sys
import importlib
import numpy as np
from scipy.optimize import minimize
import scipy

# ## OPT - via Convex Programming
# Calculates the optimal solution for the offline problem with convex programming
def solve(W, n, k, budget, size):

    # Objective function in the nash social welfare
    # Note we take the negative one to turn it into a minimization problem
    def objective(x, w, n, k, size):
        X = np.reshape(x, (n,k))
        W = np.reshape(w, (n, k))
        value = np.zeros(n)
        for i in range(n):
            value[i] = np.log(np.dot(X[i,:], W[i,:]))
        return (-1) * np.dot(size, value)


    w = W.flatten()

    obj = lambda x: objective(x, w, n, k, size)

    # Ensures that the allocations are positive
    bds = scipy.optimize.Bounds([0 for _ in range(n*k)], [np.inf for _ in range(n*k)])

    B = np.zeros((k, n*k))
    for i in range(n):
        B[:,k*i:k*(i+1)] = size[i]*np.eye(k)
    # print(B)
    # Enforces the budget constraint
    constr = scipy.optimize.LinearConstraint(B, np.zeros(k), budget)

    x0 = np.zeros(n*k)

    # Initial solution starts out with equal allocation B / S
    index = 0
    for i in range(n):
        for j in range(k):
            x0[index] = budget[j] / np.sum(size)
            index += 1

    sol = minimize(obj, x0, bounds=bds, constraints = constr, tol = 10e-8)
    return sol.x, sol


# Calculates the optimal solution for the offline problem with convex programming
# Note that this program instead solves for the optimization problem in a different form, where now
# the histogram is used directly in the original optimization problem instead of rewriting the problem
# as maximizing over types.  This was used for investigation, and not as a primary proposed heuristic in the paper.
def solve_weights(weight_matrix, weight_distribution, n, k, budget, size):

    # Similar objective, but now multiplying by the probability agent i has type j
    def objective(x, weight_matrix, n, k, size, weight_distribution):
        num_types = weight_distribution.shape[1]
        X = np.reshape(x, (n,k))
        value = np.zeros(n)
        for i in range(n):
            value[i] = np.sum([weight_distribution[i,j] * np.log(np.dot(X[i,:], weight_matrix[j,:])) for j in range(num_types)])
        return (-1) * np.dot(size, value)

    obj = lambda x: objective(x, weight_matrix, n, k, size, weight_distribution)

    # Constraints are the same as before, along with initial solution
    bds = scipy.optimize.Bounds([0 for _ in range(n*k)], [np.inf for _ in range(n*k)])

    B = np.zeros((k, n*k))
    for i in range(n):
        B[:,k*i:k*(i+1)] = size[i]*np.eye(k)

    constr = scipy.optimize.LinearConstraint(B, np.zeros(k), budget)

    x0 = np.zeros(n*k)

    index = 0
    for i in range(n):
        for j in range(k):
            x0[index] = budget[j] / np.sum(size)
            index += 1

    sol = minimize(obj, x0, bounds=bds, constraints = constr, tol = 10e-8)
    return sol.x, sol



# proportional solution, i.e. equal allocation B / S
def proportional_alloc(n, k, budget, size):
    allocations = np.zeros((n,k))
    for i in range(n):
        allocations[i, :] = budget / np.sum(size)
    return allocations

# Calculates the offline optimal solution just utilizing the distribution and not adapting to realized types
def offline_alloc(weight_matrix, weight_distribution, n, k, budget, size):
    allocations = np.zeros((n,k))
    weight_dist = np.asarray([weight_distribution for i in range(n)])
    alloc, _ = solve_weights(np.asarray(weight_matrix), np.asarray(weight_dist), n, k, budget, size)
    allocations = np.reshape(alloc, (n,k))
    return allocations



# Implements the ET - Online heuristic algorithm
def et_online(expected_weights, observed_weights, n, k, budget, size):

    allocations = np.zeros((n,k))
    current_budget = np.copy(budget)

    for i in range(n):
        if i == n-1: # Last agent gets the maximum of earlier allocations or the remaining budget
            allocations[i, :] = [max(0, min(np.max(allocations[:, j]), current_budget[j] / size[i])) for j in range(k)]
            current_budget -= size[i] * allocations[i,:]
        else:
            cur_n = n - i # Solves the eisenbergt gale program with future weights taken to be their expectation
            weights = expected_weights[i:,:]
            weights[0, :] = observed_weights[i, :]
            alloc, _ = solve(weights, cur_n, k, current_budget, size[i:])
            alloc = np.reshape(alloc, (cur_n, k))
            allocations[i, :] = [max(0, min(alloc[0, j], current_budget[j] / size[i])) for j in range(k)] # solves the eisenberg gale
            current_budget -= size[i]*allocations[i, :] # reduces budget for next iteration
    return allocations


# Implements the ET - Full heuristic algorithm
def et_full(expected_weights, observed_weights, n, k, budget, size):
    allocations = np.zeros((n,k))
    current_budget = np.copy(budget)
    weights = np.copy(expected_weights)
    for i in range(n):
        if i == n-1:
            allocations[i, :] = [max(0, min(np.max(allocations[:, j]), current_budget[j] / size[i])) for j in range(k)]
            current_budget -= size[i] * allocations[i,:]
        else:
            weights[i, :] = observed_weights[i, :] # Replaces the weights with the observed one
            alloc, _ = solve(weights, n, k, budget, size) # Solves for the allocation, and makes it
            alloc = np.reshape(alloc, (n,k))
            allocations[i, :] = [max(0, min(current_budget[j] / size[i], alloc[i,j])) for j in range(k)]
            current_budget -= size[i]*allocations[i,:] # Reduces the budget
    return allocations

# Implements the Hope-Full heuristic algorithm
def hope_full(weight_matrix, weight_distribution, observed_types, n, k, budget, size):
    num_types = len(weight_distribution)

    allocations = np.zeros((n,k))
    current_budget = np.copy(budget)

    for i in range(n):
        size_factors = np.zeros(num_types) # Calculates the number of types and the N_\theta terms
        for m in range(n):
            if m <= i:
                size_factors[observed_types[m]] += size[m]
            elif m > i:
                size_factors += size[m] * weight_distribution

        obs_type = observed_types[i]

        alloc, _ = solve(weight_matrix, num_types, k, budget, size_factors) # Solves for the allocation
        alloc = np.reshape(alloc, (num_types, k))

        allocations[i,:] = [max(0,min(current_budget[j] / size[i], alloc[obs_type, j])) for j in range(k)]

        current_budget -= size[i] * allocations[i,:] # Reduces budget
    return allocations

# Implements the Hope-Online heuristic algorithm
def hope_online(weight_matrix, weight_distribution, observed_types, n, k, budget, size):
    num_types = len(weight_distribution)

    allocations = np.zeros((n,k))
    current_budget = np.copy(budget)

    for i in range(n):
        if i == n-1:
            allocations[i, :] = [max(0, min(np.max(allocations[:, j]), current_budget[j] / size[i])) for j in range(k)]

        else:
            size_factors = np.zeros(num_types)
            for m in range(n):
                if m == i:
                    size_factors[observed_types[m]] += size[m]
                elif m > i:
                    size_factors += size[m] * weight_distribution
            obs_type = observed_types[i]

            alloc, _ = solve(weight_matrix, num_types, k, current_budget, size_factors)
            alloc = np.reshape(alloc, (num_types, k))
            allocations[i,:] = [max(0, min(current_budget[j] / size[i], alloc[obs_type, j])) for j in range(k)]


            current_budget -= size[i] * allocations[i,:]

    return allocations


# Implements the Hope-Full heuristic algorithm of a different form, by solving the original Eisenberg-Gale over agents
# taking the expectation of the utility with the histogram on types.
def hope_full_v2(weight_matrix, weight_distribution, observed_types, n, k, budget, size):
    num_types = len(weight_distribution)
    allocations = np.zeros((n,k))
    current_budget = np.copy(budget)
    weight_dist = np.asarray([weight_distribution for i in range(n)])
    for i in range(n):
        weight_dist[i, :] = np.zeros(num_types)
        weight_dist[i, observed_types[i]] += 1
        obs_type = observed_types[i]
        alloc, _ = solve_weights(weight_matrix, weight_dist, n, k, budget, size)
        alloc = np.reshape(alloc, (n,k))
        allocations[i, :] = [max(0,min(current_budget[j] / size[i], alloc[i, j])) for j in range(k)]
        current_budget -= size[i] * allocations[i, :]

    return allocations

# Similarly for the Hope-Online heuristic algorithm.
def hope_online_v2(weight_matrix, weight_distribution, observed_types, n, k, budget, size):
    num_types = len(weight_distribution)

    allocations = np.zeros((n,k))
    current_budget = np.copy(budget)
    weight_dist = np.asarray([weight_distribution for i in range(n)])
    for i in range(n):
        weight_dist[i, :] = np.zeros(num_types)
        weight_dist[i, observed_types[i]] += 1
        cur_dist = weight_dist[i:, :]
        obs_type = observed_types[i]
        alloc, _ = solve_weights(weight_matrix, cur_dist, n-i, k, budget, size[i:])
        alloc = np.reshape(alloc, (n-i,k))
        allocations[i, :] = [max(0,min(current_budget[j] / size[i], alloc[0, j])) for j in range(k)]
        current_budget -= size[i] * allocations[i, :]

    return allocations



### FAIRNESS MEASURES!

# Returns the amount of excress for each resource
def excess(allocation, budget, size):
    true_alloc = np.zeros(allocation.shape[1])
    for i in range(allocation.shape[0]):
        true_alloc += size[i] * allocation[i,:]
    return (1/allocation.shape[0])*(budget-true_alloc)

# Calculates envy-ness for each agent
def envy_utility(X, W):
    n = X.shape[0]
    envy = np.zeros(n)
    for i in range(n):
        u_i = np.dot(X[i,:], W[i,:])
        max_env = (-1)*np.inf
        for j in range(n):
            if j != i and np.dot(X[j,:], W[i,:]) - u_i > max_env:
                max_env = np.dot(X[j,:], W[i,:]) - u_i
        envy[i] = max_env
    return envy

def utility(allocation, observed_weights):
    n = allocation.shape[0]
    utility_vec = np.zeros(n)
    for i in range(n):
        utility_vec[i] = np.dot(allocation[i,:], observed_weights[i,:])
    return utility_vec


# Calculates envy to proportional for each agent
def proportionality_utility(X, W, size, budget):
    n = X.shape[0]
    max_prop = np.zeros(n)
    for i in range(n):
        prop = (np.asarray(budget) / np.sum(size))
        max_prop[i] = np.dot(prop, W[i,:]) - np.dot(X[i,:], W[i,:])
    return max_prop
