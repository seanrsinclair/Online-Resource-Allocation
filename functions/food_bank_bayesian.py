import sys
import numpy as np
import plotly.express as px
import pandas as pd
import scipy.optimize as optimization

# Auxilary function that defines the optimal policy as a function of the threshold, budget, and demand.
def policy(threshold, budget, demand):
    indicator = threshold <= min(budget, demand)
    if threshold <= min(budget, demand):
        return threshold
    else:
        return min(budget, demand)

# Calculates the Bayes Optimal solution to the problem by solving the dynamic programming
# Note that this method was specified for the demand distribution where it is 1 with probability 1/2
# and 2 with probability 1/2

def bayes_opt(n, budget, b_grid, grid_size):

    # Stores the optimal thresholds for each stage in the algorithm
    opt_policy = np.zeros((n,len(b_grid)))

    # Stores the expected value function - where the expectation is taken with respect to the demand distribution
    v_fn = np.zeros((n, len(b_grid)))

    # Solves the Bellman recursion by first looping backwards through time
    for t in np.arange(n-1,-1,-1):
        print('Step number: ' + str(t))
        # Then loops over each discretized budget
        for b in range(len(b_grid)):

            # Determines the current budget
            current_budget = b*grid_size

            # the optimal policy is to give out the rest
            if t == n-1:
                opt_policy[t,b] = float('inf')
                v_fn[t,b] = (1/2)*np.log(policy(opt_policy[t,b],current_budget,1)/1) + (1/2)*np.log(policy(opt_policy[t,b],current_budget,2)/2)


            else:

                # Computes the q values for each possible allocation
                q_vals = np.log(b_grid[0:(b+1)]) + np.flip(v_fn[t+1,0:(b+1)])

                # Takes the maximum index in order to get the optimal allocation threshold
                opt_policy[t,b] = np.argmax(q_vals)*grid_size

                # Uses this threshold in order to discretize new budgets for the next town given the different demands
                new_budget_one = int(np.floor((current_budget - policy(opt_policy[t,b],current_budget,1))/grid_size))
                new_budget_two = int(np.floor((current_budget - policy(opt_policy[t,b],current_budget,2))/grid_size))

                # Evaluates the value function using this optimal policy
                v_fn[t,b] = (1/2)*(np.log(policy(opt_policy[t,b],current_budget,1)/1)+v_fn[t+1, new_budget_one]) \
                        + (1/2)*(np.log(policy(opt_policy[t,b],current_budget,2)/2)+v_fn[t+1, new_budget_two])

    return opt_policy, v_fn


# Solves the problem with more general discrete distributions.
def bayes_opt_general_dist(n, budget, full_distribution, full_support, b_grid, grid_size):

    # Stores the optimal thresholds for each stage in the algorithm
    opt_policy = np.zeros((n,len(b_grid)))

    # Stores the expected value function - where the expectation is taken with respect to the demand distribution
    v_fn = np.zeros((n, len(b_grid)))

    # Solves the Bellman recursion by first looping backwards through time
    for t in np.arange(n-1,-1,-1):

        # Then loops over each discretized budget
        for b in range(len(b_grid)):

            distribution = full_distribution[t]
            support = full_support[t]

            # Determines the current budget
            current_budget = b*grid_size

            # the optimal policy is to give out the rest
            if t == n-1:
                opt_policy[t,b] = float('inf')
                v_fn[t,b] = np.dot(distribution, [np.log(policy(opt_policy[t,b], current_budget, x) / x) for x in support])


            else:

                # Computes the q values for each possible allocation
                q_vals = np.log(b_grid[0:(b+1)]) + np.flip(v_fn[t+1,0:(b+1)])

                # Takes the maximum index in order to get the optimal allocation threshold
                opt_policy[t,b] = np.argmax(q_vals)*grid_size

                # Uses this threshold in order to discretize new budgets for the next town given the different demands
                new_budget = [int(np.floor((current_budget - policy(opt_policy[t,b], current_budget, x))/x)) for x in support]

                # Evaluates the value function using this optimal policy
                v_fn[t,b] = np.dot(distribution, [np.log(policy(opt_policy[t,b], current_budget, support[i])/support[i]) + v_fn[t+1, new_budget[i]] for i in range(len(support))])

    return opt_policy, v_fn





# Online allocation algorithm which utilizes the Bayesian optimal solution to the dynamic program

def waterfilling_bayesian(demands, opt_policy, budget, b_grid, grid_size):
    #print(demands, weights,budget,index, width)
    n = np.size(demands)
    allocations = np.zeros(n)
    budget_remaining = budget
    for i in range(n):
        index = int(np.floor(budget_remaining/grid_size))
        allocations[i] = policy(opt_policy[i,index], budget_remaining, demands[i])
        budget_remaining -= allocations[i]
    return allocations
