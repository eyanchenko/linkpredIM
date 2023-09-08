# Load libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import statistics
import timeit
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from multiprocessing import Pool
from multiprocessing import get_context
from itertools import repeat
from itertools import chain
from node2vec import Node2Vec
import networkx as nx
from multiprocessing import Pool
from multiprocessing import get_context
from igraph import Graph
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable


# Function to simulate SI process
# E:     input graph (as edge list)
# S:     seed nodes
# lam:   susceptibility parameter
# TEMPORAL
def sigma(E, S, lam):
    # I will keep track of infected nodes
    I = S.copy()
    # Unique time snapshots of the network
    T = pd.unique(E["t"])

    # Run process over time steps
    for t in T:
        # Function to see if new nodes are infected
        # Apply to each row of edge data frame
        def infect(x):
            # Check if one node is infected and the other isn't
            if x[0] in I and x[1] not in I:
                if random.random() < (lam * x[3]):
                    return x[1]
                else:
                    return np.nan
            elif x[0] not in I and x[1] in I:
                if random.random() < (lam * x[3]):
                    return x[0]
                else:
                    return np.nan
            else:
                return np.nan

        out = E[E["t"]==t].apply(infect, axis=1) # those infected during this step
        out = [item for item in out if not(math.isnan(item)) == True]
        out = [round(elem) for elem in out]
        I.extend(out)
        I = list(set(I))

    return(len(I))

# STATIC
def sigma_s(E, S, lam, T):
    # I will keep track of infected nodes
    I = S.copy()

    # Run process over time steps
    for t in range(T):
        # Function to see if new nodes are infected
        # Apply to each row of edge data frame
        def infect(x):
            # Check if one node is infected and the other isn't
            if x[0] in I and x[1] not in I:
                if random.random() < (lam * x[3]):
                    return x[1]
                else:
                    return np.nan
            elif x[0] not in I and x[1] in I:
                if random.random() < (lam * x[3]):
                    return x[0]
                else:
                    return np.nan
            else:
                return np.nan

        out = E.apply(infect, axis=1) # those infected during this step
        out = [item for item in out if not(math.isnan(item)) == True]
        out = [round(elem) for elem in out]
        I.extend(out)
        I = list(set(I))

    return(len(I))


# For a fixed E, S, and lam
# finds average influence over nsims simulations
# TEMPORAL
def sigma_mc(E, S, lam, nsims=100, se=False, probs=False):
    # Define function to loop over
    if probs: # Generate new graph for each iteration
        def apply_fun(i):
            E_i = E.copy()
            E_i["pp"] = bernoulli.rvs(size=E_i.shape[0], p=E_i["p"])
            E_i = E_i[E_i["pp"]==1]
            return(sigma(E_i, S, lam))
    else: # Use same edge graph for each iteration
        def apply_fun(i):
            return(sigma(E, S, lam))

    # Repeat function nsims times
    out = map(apply_fun, range(nsims))
    out = list(out)

    if se:
        return(statistics.mean(out), math.sqrt(statistics.variance(out)/nsims))
    else:
        return(statistics.mean(out))


# STATIC
def sigma_mc_s(E, S, lam, T, nsims=100, se=False, probs=False):
    # Define function to loop over
    if probs: # Generate new graph for each iteration
        def apply_fun(i):
            E_i = E.copy()
            E_i["pp"] = bernoulli.rvs(size=E_i.shape[0], p=E_i["p"])
            E_i = E_i[E_i["pp"]==1]
            return(sigma_s(E_i, S, lam, T))
    else: # Use same edge graph for each iteration
        def apply_fun(i):
            return(sigma_s(E, S, lam, T))


    # Repeat function nsims times
    out = map(apply_fun, range(nsims))
    out = list(out)

    if se:
        return(statistics.mean(out), math.sqrt(statistics.variance(out)/nsims))
    else:
        return(statistics.mean(out))

# Greedy algorithm to find optimal seed nodes
# Uses MC simulations to estimate influence function
# E:     edge list
# n:     number of vertices
# k:     size of seed set
# lam:   susceptibility parameter in SI model
# nsims: number of MC sims to compute average influence

# TEMPORAL
def greedy(E, k, lam, S_init=[], nsims=100, probs=False, mc_cores):
    #Initialize seed set
    S = S_init
    # Initialize max influence
    value = 0
    # Possible nodes to consider
    V = set(E["v1"]).union(set(E["v2"]))
    V = list(V - set(S))

    if k > len(V):
        return[V, 0]

    # Continue process until k nodes in the seed set
    for i in range(len(S),k):

        # Potential seed sets
        S_pot = [0]*len(V)
        for i in range(len(V)):
            S_pot[i] = S+[V[i]]

        # Loop over nodes to see which leads to largest increase in influence
        pool = get_context("fork").Pool(mc_cores)
        out = pool.starmap(sigma_mc, zip(repeat(E), S_pot, repeat(lam), repeat(nsims), repeat(False), repeat(probs)) )
        pool.close()

        idx   = out.index(max(out))
        value = out[idx]

        # Add max node to seed set and remove it from possible nodes to consider set
        S.append(V[idx])
        V.pop(idx)


    return([S, value])

# STATIC
def greedy_s(E, k, lam, T, S_init=[], nsims=100, probs=False, mc_cores):
    #Initialize empty seed set
    S = S_init
    # Initialize max influence
    value = 0
    # Possible nodes to consider
    V = set(E["v1"]).union(set(E["v2"]))
    V = list(V - set(S))
    if k > len(V):
        return[V, 0]

    # Continue process until k nodes in the seed set
    for i in range(len(S),k):

        # Potential seed sets
        S_pot = [0]*len(V)
        for i in range(len(V)):
            S_pot[i] = S+[V[i]]

        # Loop over nodes to see which leads to largest increase in influence
        pool = get_context("fork").Pool(mc_cores)
        out = pool.starmap(sigma_mc_s, zip(repeat(E), S_pot, repeat(lam), repeat(T), repeat(nsims), repeat(False), repeat(probs)) )
        pool.close()

        idx   = out.index(max(out))
        value = out[idx]

        # Add max node to seed set and remove it from possible nodes to consider set
        S.append(V[idx])
        V.pop(idx)


    return([S, value])

## Implement method from Murata and Koga (2018)
## Dynamic degree Discount 

def dyndeg(E, k, lam):
    #Initialize empty seed set
    S = []
    # Possible nodes to consider
    V = list(set(E["v1"]) or set(E["v2"]))
    n = len(V)

    ######## Step 1: Compute dynamic degrees

    # Initialize dynamic degree
    DD = [0]*n
    td = [0]*n
    # Temporal snapshots
    T = pd.unique(E["t"])

    # Create array where i th entry is the neighbors of nodes i at any time
    nodes_total = [[] for Null in range(n)]
    # Create array where i th entry is the neighbors of node i at time t-1
    nodes_prev = [[] for Null in range(n)]

    for v in range(n):
        # Find neighbors of node v at t=0 and probability of being neighbor (probability of an edge)
        nodes1        = list(E[(E["v1"]==V[v]) & (E["t"]==T[0])]["v2"])
        nodes2        = list(E[(E["v2"]==V[v]) & (E["t"]==T[0])]["v1"])
        nodes = nodes1 + nodes2

        if len(nodes)>0:
            nodes_prev[v] = list(set(nodes))
            nodes_total[v] = list(set(nodes))
    # Compute dynamic degree
    for t in T[1:len(T)]:
        # Create array where i th entry is the neighbors of node i at time t
        nodes_curr = [[] for Null in range(n)]
        for v in range(n):
            # Find neighbors at current time
            nodes1        = list(E[(E["v1"]==V[v]) & (E["t"]==t)]["v2"])
            nodes2        = list(E[(E["v2"]==V[v]) & (E["t"]==t)]["v1"])
            nodes = nodes1 + nodes2

            if len(nodes)>0:
                nodes_curr[v] = list(set(nodes))
                nodes_total[v] = list(set(nodes_total[v] + nodes_curr[v]))

            # Formula for dynamic degree from page 6 of Murata and Koga (2018)
            if len(nodes_prev[v]) > 0 or len(nodes_curr[v]) > 0:
                DD[v] += len(nodes_curr[v]) * (len(set(nodes_prev[v]) - set(nodes_curr[v]))) / (len(set(nodes_prev[v] + nodes_curr[v])))
        nodes_prev = nodes_curr


    ######## Step 2: Add node to seed with largest dynamic degrees and update values
    dd = DD
    for i in range(k):
        idx = dd.index(max(dd)) # Find max dd
        v = V[idx]
        S.append(v)             # and add it to seed set

        # Find which nodes were neighbors with v and update dd
        for u in range(n):
            if v in nodes_total[u]:
                td[u] += 1
                dd[u] = DD[u] - 2*td[u] - (DD[u]-td[u])*td[u]*lam

        # Remove node v from future consideration
        nodes_total[idx] = []
        dd[idx] = -100




    return(S)

# Find influential nodes on degree (static version of dynamic degree method)
def statdeg(E, k):
    E = E.reset_index(drop=True)
    S = []
    # Possible nodes to consider
    V = list(set(E["v1"]).union(set(E["v2"])))
    n = len(V)

    # Initialize degrees and neighbors
    dd     = [0] * n
    neighs = [[] for Null in range(n)]

    # Find degrees and neighbors
    for i in range(E.shape[0]):
        v1 = E["v1"][i]
        v2 = E["v2"][i]

        id1 = V.index(v1)
        id2 = V.index(v2)

        dd[id1] += 1
        dd[id2] += 1

        neighs[id1].append(v2)
        neighs[id2].append(v1)

    for i in range(k):
        idx = dd.index(max(dd)) # Find max dd
        v   = V[idx]
        S.append(v)             # and add it to seed set

        # Find which nodes were neighbors with v and update dd
        for u in range(n):
            if v in neighs[u]:
                dd[u] -= 1

        # Remove node v from future consideration
        neighs[idx] = []
        dd[idx] = -100

    return(S)
