import numpy as np


cthresh = 1
cup = 1
clow = 1





def offline_opt(budget, size, mean, stdev):
    return [budget / np.sum(size) for i in range(len(size))]



def hope_online(budget, size, mean, stdev):
    allocation = np.zeros(len(size))
    budget_remaining = budget
    
    rem = len(size)
    size_future = size[0] + np.sum(mean[1:])
    
    for i in range(len(allocation)):
        size_future = size[i] + np.sum(mean[i+1:])
        allocation[i] = min(budget_remaining / size[i], budget_remaining / size_future)
        budget_remaining -= allocation[i] * size[i]
    return allocation


def hope_full(budget, size, mean, stdev):
    allocation = np.zeros(len(size))
    budget_remaining = budget
    
    rem = len(size)
    
    for i in range(len(allocation)):
        size_future = np.sum(size[0:i+1]) + np.sum(mean[i+1:])
        allocation[i] = min(budget_remaining / size[i], budget / size_future)
        budget_remaining -= allocation[i] * size[i]

    return allocation



def fixed_threshold(budget, size, mean, stdev):
    allocation = np.zeros(len(size))
    budget_remaining = budget
    
    rem = len(size)
    size_future = size[0] + np.sum(mean[1:])
    
    
    c = np.sqrt(np.max(stdev)*np.mean(mean)*rem) / size_future
    thresh_lower = (budget / size_future)*(1 / (1 + c))

    
    for i in range(len(allocation)):
        allocation[i] = min(budget_remaining / size[i], thresh_lower)
        budget_remaining -= allocation[i] * size[i]
        
    return allocation




def hope_guardrail(budget, size, mean, stdev, Lt):
    allocation = np.zeros(len(size))
    budget_remaining = budget
    rem = len(size)
    
    size_future = size[0] + np.sum(mean[1:])

    c = np.sqrt(np.max(stdev)*np.mean(mean)*rem) / size_future
    thresh_lower = (budget / size_future)*(1 / (1 + c))
    
    c = (1 / (rem**(Lt)))*(1 +  np.sqrt(np.max(stdev)*np.mean(mean)*rem) / size_future) \
                - np.sqrt(np.max(stdev)*np.mean(mean)*rem) / size_future
    
    thresh_upper = (budget / size_future)*(1 / (1- c))
    
    
    for i in range(len(allocation)):
    
        rem = len(allocation) - i - 1
        conf_bnd = np.sqrt(np.max(stdev)*np.mean(mean)*(rem))
        
        if rem == 0 and budget_remaining / size[i] >= thresh_lower and budget_remaining / size[i] <= thresh_upper:
            allocation[i] = budget_remaining / size[i]

        
        elif budget_remaining / size[i] < thresh_lower:
            # print(str(i) + ' giving rest of budget!')
            allocation[i] = budget_remaining / size[i]
        
        elif budget_remaining >= thresh_lower * (np.sum(mean[i+1:]) + clow*conf_bnd) + size[i] * thresh_upper:
            allocation[i] = thresh_upper

        else:
            allocation[i] = thresh_lower

            
        budget_remaining -= allocation[i] * size[i]

        
    if np.round(budget_remaining, 3) < 0:
        print(budget_remaining)
        print('Error: Negative Budget')
                
        
    return allocation


