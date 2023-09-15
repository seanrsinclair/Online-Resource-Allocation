import numpy as np
import matplotlib.pyplot as plt

EPS = 10e-3

def export_legend(legend, filename="LABEL_ONLY.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def check_offset_expiry(perish_dist, demand_dist, n, max_budget, num_iters = 1000):
    """
        Takes as input a perishing and demand distribution and outputs the probability estimate
        that offset expiry is satisfied
    """
    num_valid = 0 
    for _ in range(num_iters):
        sizes = demand_dist(n)
        resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)])
        check_optimality = [(max_budget / np.sum(sizes))*np.sum(sizes[:(t+1)]) 
                                - np.count_nonzero([resource_perish <= t]) for t in range(n)]
        if np.min(check_optimality) >= 0: # checks if B/N is feasible in hindsight
            num_valid += 1
    return (num_valid / num_iters)


def x_lower_line_search(perish_dist, demand_dist, n, max_budget):
    """
        Takes as input a perishing and demand distribution and outputs an estimate of
        X_lower, the largest feasible X taking into account perishing    
    """

    valid = False # Indicator for whether a valid solution has been found

    tot_demand_samples = [np.sum(demand_dist(n)) for _ in range(1000)] # gets samples of the total demand N
    n_upper = np.mean(tot_demand_samples) + np.sqrt(np.var(tot_demand_samples) * n)
    x_lower = max_budget / n_upper + EPS # starts off as checking Delta(X) for X = B / N_upper (as in, the feasible X_lower with no perishing)

    while not valid: # loops down from x_lower until a valid solution is found
        x_lower = x_lower - EPS # takes off epsilon to search for lower values
        delta = estimate_delta(x_lower, perish_dist, demand_dist, n, max_budget) # estimates Delta(X)
        if x_lower <= (max_budget - delta) / (n_upper): # Checks if X <= (B - Delta) / n_upper
            valid = True
    return x_lower


def estimate_delta(x, perish_dist, demand_dist, n, max_budget, num_iters = 1000):
    """
        Estimates Delta(X) = |{b : T_b < \tau_b(X)}| through Monte-Carlo simulations
        NOTE: This is simplifying \tau_b to not take into account the demands N_t
        ADJUST?
    """
    total_mass = 0
    for _ in range(num_iters):
        sizes = demand_dist(n) # samples demands N_t
        resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)]) # samples perishing times T_b
        total_mass += np.sum([1 if resource_perish[b] < np.ceil(b/x) else 0 for b in range(max_budget)]) # checks whether
            # resources perish before tau_b
    avg_mass = (total_mass / num_iters) # takes expectation over the number of iterations
    return avg_mass + np.sqrt(avg_mass * np.log(n)) # returns an estimate for \overline{\Delta}

def perish_future(current_index, resource_dict, x_lower, perish_dist, demand_dist, n, max_budget, num_iters = 1000):
    """
        Estimates P_upper_t through Monte-Carlo simulations
        NOTE: This is done in a state-dependent way
    """

    total_mass = 0
    for _ in range(num_iters):
        sizes = demand_dist(n) # samples demands N_t
        resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)])
        
        for b in np.arange(current_index, max_budget):
            total_mass += 1 if resource_perish[b] < np.ceil(b/x_lower) else 0
    avg_mass = (total_mass / num_iters)
    return avg_mass + np.sqrt(avg_mass + np.log(n))