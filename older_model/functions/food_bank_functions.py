#  Food Bank Problem

import sys
import numpy as np
import plotly.express as px
import pandas as pd
import scipy.optimize as optimization
import scipy

# ## OPT - Waterfilling

## Water-filling Algorithm for sorted demands
def waterfilling_sorted(d,b):
    n = np.size(d)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        equal_allocation = bundle_remaining/(n-i)
        if d[i]<equal_allocation:
            allocations[i] = min(d[i], bundle_remaining) if i==n-1 else d[i]
        else:
            allocations[i] = equal_allocation
        bundle_remaining -= allocations[i]
    return allocations

## Water-filling Algorithm for sorted demands
def waterfilling_sorted_waste(d,b):
    n = np.size(d)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        equal_allocation = bundle_remaining/(n-i)
        if d[i]<equal_allocation:
            allocations[i] = d[i]
        else:
            allocations[i] = equal_allocation
        bundle_remaining -= allocations[i]
    return allocations

## Water-filling Algorithm for sorted demands and weights for bucket width
def waterfilling_sorted_weights(demands, weights, budget):
    n = np.size(demands)
    allocations = np.zeros(n)
    budget_remaining = budget
    width = np.sum(weights)

    for i in range(n):
        if width < .0001:
            equal_allocation = 0
        else:
            equal_allocation = budget_remaining / width

        if demands[i]<=equal_allocation:
            allocations[i] = min(budget_remaining, demands[i])
        else:
            allocations[i] = equal_allocation

        budget_remaining -= allocations[i]*weights[i]
        width -= weights[i]

    return allocations



## Water-filling Algorithm for general demands
def waterfilling(d,b):
    n = np.size(d)
    sorted_indices = np.argsort(d)
    sorted_demands = np.sort(d)
    sorted_allocations = waterfilling_sorted(sorted_demands, b)
    allocations = np.zeros(n)
    for i in range(n):
        allocations[sorted_indices[i]] = sorted_allocations[i]
    return allocations

## Water-filling Algorithm for general demands
def waterfilling_waste(d,b):
    n = np.size(d)
    sorted_indices = np.argsort(d)
    sorted_demands = np.sort(d)
    sorted_allocations = waterfilling_sorted_waste(sorted_demands, b)
    allocations = np.zeros(n)
    for i in range(n):
        allocations[sorted_indices[i]] = sorted_allocations[i]
    return allocations


def insert_sorted(lst, element):
    n = np.size(lst)
    if n==0:
        return np.array([element]),0
    if element<=lst[0]:
        return np.append(element,lst),0
    if element>=lst[n-1]:
        return np.append(lst,element),n
    left = 0
    right = n-1
    while left<right-1:
        mid_ind = int((left+right)/2)
        if element<lst[mid_ind]:
            right = mid_ind
        elif element > lst[mid_ind] :
            left = mid_ind
        if element == lst[mid_ind] or (element>lst[mid_ind] and element<lst[mid_ind+1]):
            return np.append(np.append(lst[:mid_ind+1],element),lst[mid_ind+1:]), mid_ind+1
    return np.append(np.append(lst[:left],element),lst[left:]), left


def delete_sorted(lst,element):
    n = np.size(lst)
    if element==lst[0]:
        return lst[1:]
    if element==lst[n-1]:
        return lst[:-1]
    left = 0
    right = n-1
    while left<right-1:
        mid_ind = int((left+right)/2)
        if element<lst[mid_ind]:
            right = mid_ind
        elif element > lst[mid_ind] :
            left = mid_ind
        else:
            return np.append(lst[:mid_ind],lst[mid_ind+1:])




## Tests
assert list(waterfilling(np.zeros(0), 5)) == []
assert list(waterfilling(np.array([1,2,3,4]), 10)) == [1,2,3,4]
assert list(waterfilling(np.array([3,4,1,2]), 10)) == [3,4,1,2]
assert list(waterfilling(np.array([1,2,3,4]), 8)) == [1,2,2.5,2.5]
assert list(waterfilling(np.array([3,1,4,2]), 8)) == [2.5,1,2.5,2]
assert list(waterfilling(np.array([3,6,5,6]), 8)) == [2,2,2,2]

##############################
## ## Online Algorithms
##############################

## Online Water-filling taking minimum of realized demand and B/n
def waterfilling_proportional(demands_realized,b):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    eq = 1 if n==0 else b/n
    bundle_remaining = b
    for i in range(n):
        if i!=n-1:
            allocations[i] = min(eq, demands_realized[i])
        else:
            allocations[i] = min(demands_realized[i], bundle_remaining)
        bundle_remaining -= allocations[i]
    return allocations

## Online Water-filling taking minimum of realized demand and B/n
# (also called adaptive-threshold in the paper)
def waterfilling_proportional_remaining(demands_realized,b):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        if i!=n-1:
            allocations[i] = min(bundle_remaining/(n-i), demands_realized[i])
        else:
            allocations[i] = min(bundle_remaining, demands_realized[i])
        bundle_remaining -= allocations[i]
    return allocations


# In[18]:


## O(n^2) version of online algorithm that needs waterfilling evaluated multiple times
def waterfilling_et_online(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted(sorted_demands,demands_realized[i])
        if i<n-1:
            allocations[i] = min((waterfilling_sorted(new_sorted_list, bundle_remaining))[index],demands_realized[i])
        else:
            allocations[i] = bundle_remaining
        bundle_remaining -= allocations[i]
    return allocations

## O(n^2) version of online algorithm that needs waterfilling evaluated multiple times
def waterfilling_et_waste(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(np.copy(demands_predicted))
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted(sorted_demands,demands_realized[i])
        allocations[i] = min((waterfilling_sorted(new_sorted_list, bundle_remaining))[index],demands_realized[i], bundle_remaining)
        bundle_remaining -= allocations[i]
    return allocations


def waterfilling_et_full_waste(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    demands_solve = np.copy(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        demands_solve[i] = demands_realized[i]
        sorted_demands = np.sort(demands_solve)
        index = np.argmin(np.abs(sorted_demands - demands_realized[i]))

        allocations[i] = min((waterfilling_sorted(sorted_demands, b))[index],demands_realized[i], bundle_remaining)
        bundle_remaining -= allocations[i]
    return allocations

# Implements Hope Online algorithm with i.i.d. dem,ands
def waterfilling_hope_waste_iid(weights_orig, supports_orig, demands_realized, budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    support = np.copy(supports_orig)
    weight = np.copy(weights_orig)*n

    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand
        if i<n-1:
            obs_demand = demands_realized[i]
            weight -= weights_orig
            index = np.argmin(np.abs(support - obs_demand))
            weight[index] += 1

            waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget_remaining)
            allocations[i] = min(max(waterfilling_alloc), demands_realized[i], budget_remaining)
            weight[index] -= 1
        else:
            allocations[i] = min(budget_remaining,demands_realized[i], max(allocations))

        budget_remaining -= allocations[i]
    return allocations

# Implements Hope Online algorithm with i.i.d. dem,ands
def waterfilling_hope_waste_iid_delta(weights_orig, supports_orig, demands_realized, budget, delta):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    support = np.copy(supports_orig)
    weight = np.copy(weights_orig)*n

    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand
        if i<n-1:
            obs_demand = demands_realized[i]
            weight -= weights_orig
            index = np.argmin(np.abs(support - obs_demand))
            weight[index] += 1

            waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget_remaining)
            allocations[i] = min(max(waterfilling_alloc)-delta, demands_realized[i], budget_remaining)
            weight[index] -= 1
        else:
            allocations[i] = min(budget_remaining,demands_realized[i], max(allocations))

        budget_remaining -= allocations[i]
    return allocations

def waterfilling_hope_waste(weights_orig, supports_orig, demands_realized, budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    support = np.copy(supports_orig)
    weight = np.sum(weights_orig, axis=0)
    # print(weight)
    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand
        if i<n-1:
            obs_demand = demands_realized[i]
            weight -= weights_orig[i,:]
            index = np.argmin(np.abs(support - obs_demand))
            weight[index] += 1


            waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget_remaining)
            allocations[i] = min(max(waterfilling_alloc), demands_realized[i], budget_remaining)
            weight[index] -= 1
        else:
            allocations[i] = min(budget_remaining,demands_realized[i])

        budget_remaining -= allocations[i]
    return allocations

def waterfilling_hope_full_waste(weights_orig, supports_orig, demands_realized, budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    support = np.copy(supports_orig)
    weight = np.sum(weights_orig, axis=0)

    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand

        obs_demand = demands_realized[i]
        weight -= weights_orig[i, :]
        index = np.argmin(np.abs(support - obs_demand))
        weight[index] += 1


        waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget)
        allocations[i] = min(max(waterfilling_alloc), demands_realized[i], budget_remaining)


        budget_remaining -= allocations[i]
    return allocations


def waterfilling_hope_full_waste_iid(weights_orig, supports_orig, demands_realized, budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    support = np.copy(supports_orig)
    weight = np.copy(weights_orig)*n

    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand

        obs_demand = demands_realized[i]
        weight -= weights_orig
        index = np.argmin(np.abs(support - obs_demand))
        weight[index] += 1


        waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget)
        allocations[i] = min(waterfilling_alloc[index],demands_realized[i], budget_remaining)

        budget_remaining -= allocations[i]

    return allocations


# def waterfilling_guardrail(weights_orig, supports_orig, demands_realized, budget, delta):
#     n = np.size(demands_realized)
#     allocations = np.zeros(n)
#     budget_remaining = budget

#     support_full = np.copy(supports_orig)
#     weight_full = np.copy(weights_orig)*n

#     support_online = np.copy(supports_orig)
#     weight_online = np.copy(weights_orig)*n
    
#     for i in range(n):
#         # need to collect distribution on weights
#         # and add in one for observed demand
        
#         # Calculate hope_full
#         obs_demand = demands_realized[i]
#         weight_full -= weights_orig
#         index_full = np.argmin(np.abs(support - obs_demand))
#         weight[index_full] += 1
#         waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget)
#         # Calculate budget process?
        
        
#         # Calculate hope_Online
        
        
         
        
        
#         weight_online -= weights_orig
#         index_online = np.argmin(np.abs(support - obs_demand))
#         weight[index_online] += 1


#         allocations[i] = min(waterfilling_alloc[index],demands_realized[i], budget_remaining)

#         budget_remaining -= allocations[i]

#     return allocations



def waterfilling_hope_full_waste_iid_delta(weights_orig, supports_orig, demands_realized, budget, delta):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    support = np.copy(supports_orig)
    weight = np.copy(weights_orig)*n

    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand

        obs_demand = demands_realized[i]
        weight -= weights_orig
        index = np.argmin(np.abs(support - obs_demand))
        weight[index] += 1


        waterfilling_alloc = waterfilling_sorted_weights(support, weight, budget)
        allocations[i] = min(waterfilling_alloc[index]-delta,demands_realized[i], budget_remaining)

        budget_remaining -= allocations[i]

    return allocations



# Greedy allocation strategy
def greedy(demands_realized,budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget
    for i in range(n):
        if demands_realized[i] <= budget_remaining:
            allocations[i] = demands_realized[i]
            budget_remaining -= demands_realized[i]
        else:
            allocations[i] = budget_remaining
            budget_remaining = 0
    return allocations

# MaxMin heuristic algorithm from another paper (see readme for citation)
def max_min_heuristic(demands_realized, median_demands, mean_demands, variance_demands, budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget
    min_fill = 1
    for i in range(n):

        # At the last step the person gets the remaining budget or their demand
        if i == n-1:
            allocations[i] = min(demands_realized[i], budget_remaining)

        else:
            delta = (median_demands[i] - median_demands[i+1]) / ((1/2)* (median_demands[i] + median_demands[i+1]))
            budget_portion = budget_remaining * (mean_demands[i] + mean_demands[i+1]) / np.sum(mean_demands[i:])
            heuristic_threshold = budget_portion * (demands_realized[i] / (demands_realized[i] + median_demands[i+1] + delta * np.sqrt(variance_demands[i+1])))
            allocations[i] = min(heuristic_threshold, min_fill*demands_realized[i], budget_remaining)

            if allocations[i] / demands_realized[i] <= min_fill:
                min_fill = allocations[i] / demands_realized[i]
            budget_remaining -= allocations[i]

    return allocations

# Allocates according to a constant threshold
def constant_threshold(demands_realized,budget,threshold):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget
    for i in range(n):
        allocations[i] = min(budget_remaining,demands_realized[i], threshold)
        budget_remaining -= allocations[i]
    return allocations



######
# FAIRNESS METRICS
######

# vector returning the maximum envy every town feels by  utility
def envy_vector(allocation, demands):
    n = np.size(demands)
    allocation_max = np.amax(allocation)
    envy = np.zeros(n)
    for i in range(n):
        envy[i] = min(allocation_max/demands[i], 1) - min(allocation[i]/demands[i],1)
    return envy


def proportionality(allocation,demands, budget):
    n = np.size(demands)
    prop = budget/n
    max_prop = np.zeros(n)
    for i in range(n):
        if allocation[i]<demands[i]:
            max_prop[i] = max(0,min(prop/demands[i],1) - min(allocation[i]/demands[i],1))
    return max_prop


def excess(allocation, budget):
    return (1/len(allocation))*(budget-np.sum(allocation))


def envy_utility(allocation, demands):
    n = np.size(demands)
    allocation_max = np.amax(allocation)
    envy = np.zeros(n)
    for i in range(n):
        envy[i] = min(1,allocation_max/demands[i]) - min(1,allocation[i]/demands[i])
    return envy

def proportionality_utility(allocation,demands, budget):
    n = np.size(demands)
    prop = budget/n
    max_prop = np.zeros(n)
    for i in range(n):
        max_prop[i] = min(1,prop/demands[i]) - min(1,allocation[i]/demands[i])
    return max_prop

def utility_ratio(allocation, demands, budget):
    utility = np.zeros(len(allocation))
    for i in range(len(allocation)):
        utility[i] = min(allocation[i]/demands[i], 1)
    return utility


####
# TEST CASES
####

demands1 = np.array([1,2,1,2,1])
demands2 = np.array([1,1,1,1,1])
demands3 = np.array([2,2,2,2,2])
demands4 = np.array([2,2,1,1,1])
demands4 = np.array([1,1,2,2,2])
demands4 = np.array([2,1,2,1,2])
budget = 7.5

assert list(envy_vector(np.array([1,1,1,1,1]),demands1)) == [0,0,0,0,0]
assert list(envy_vector(demands1,demands1)) == [0,0,0,0,0]
assert list(envy_vector(np.array([1,1,1,2,1]),demands1)) == [0,0.5,0,0,0]
assert list(envy_vector(np.array([1,1,1,2,0]),demands1)) == [0,0.5,0,0,1]
assert list(envy_vector(np.array([0,0,0,0,5]),demands1)) == [1,1,1,1,0]
assert list(envy_vector(np.array([2,2,2,2,2]),demands1)) == [0,0,0,0,0]
#############
assert list(proportionality(np.array([1,2,1,2,1]),demands1,budget)) == [0,0,0,0,0]
assert list(proportionality(np.array([1,1,1,1,1]),demands1,budget)) == [0,0.25,0,0.25,0]
assert list(proportionality(np.array([1.5,1.5,1.5,1.5,1.5]),demands1,budget)) == [0,0,0,0,0]
##############
assert excess(np.array([1,1,1,1,1]),10) == 5/5
assert excess(np.array([1,1,1,1,1]),1) == (-4)/5
assert excess(np.array([1,1,1,1,1]),5) == 0


####
# Calculating objective functions to test for C.R. guarantees
####

## Calculate log of Nash welfare for objective function
def objective_nash_log(demands, allocation):
    welfare_sum = 0
    for i in range(np.size(demands)):
        welfare_sum += np.log(min(1,allocation[i]/demands[i]))
    return welfare_sum

def objective_nash_log_normalized(demands, allocation):
    welfare_sum = 0
    n = np.size(demands)
    for i in range(n):
        welfare_sum += np.log(min(1,allocation[i]/demands[i]))
    return welfare_sum/n

def objective_nash_log_vector(demands, allocation):
    n = np.size(demands)
    welfare_vector = np.zeros(n)
    for i in range(n):
            welfare_vector[i] = np.log(min(1,allocation[i]/demands[i]))
    return welfare_vector


## Calculate log of Nash welfare for objective function
def objective_nash(demands, allocation):
    welfare_product = 1
    n=np.size(demands)
    for i in range(n):
        welfare_product = welfare_product*min(1,allocation[i]/demands[i])
    return welfare_product**1/n

## Calculate log of Nash welfare for objective function
def objective_nash_mod(demands, allocation):
    welfare_product = 1
    n=np.size(demands)
    for i in range(n):
        welfare_product = welfare_product*min(1,allocation[i]/demands[i])
    return welfare_product

## Calculate log of Nash welfare for objective function
def objective_sum(demands, allocation):
    welfare_sum = 0
    n=np.size(demands)
    for i in range(n):
        welfare_sum = welfare_sum+min(1,allocation[i]/demands[i])
    return welfare_sum



####
# MISC Code
####

# MISC Code used to calculate median and variance of a discrete distribution
def median(support, weights):
    tot = 0
    for i in range(len(support)):
        tot += weights[i]
        if tot >= .5:
            return support[i]

def variance(support, weights):
    mean = np.dot(support, weights)
    return np.dot(weights, (support - mean)**2)
