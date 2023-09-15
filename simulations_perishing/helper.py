import numpy as np

def check_offset_expiry(perish_dist, demand_dist, n, max_budget, num_iters = 1000):
    
    num_valid = 0
    
    for _ in range(num_iters):
        
        sizes = demand_dist(n)
        resource_perish = np.asarray([time_dist(b,n) for b in range(max_budget)])
        check_optimality = [(max_budget / np.sum(size))*np.sum(size[:(t+1)]) 
                                - np.count_nonzero([resource_perish <= t]) for t in range(n)]
        
        if np.min(check_optimality) >= 0: # checks if B/N is feasible in hindsight
            num_valid += 1
    return (num_valid / num_iters)

def check_delta(x, perish_dist, demand_dist, n, max_budget, num_iters = 1000):
    
    total_mass = 0
    
    return (total_mass / num_iters)
