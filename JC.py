# Jaccard coefficient link prediction
def cn_linkpred(E, pct=0.05, steps=1, remove=False, probs=True):
    # Convert to networkx and find Jaccard Coefficient for each edge pair without a link
    E_ret = E.copy()

    G = nx.from_pandas_edgelist(E_ret, source="v1", target="v2").to_undirected()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out = nx.jaccard_coefficient(G)

    jc = pd.DataFrame({"v1":[0], "v2":[0], "p":[0]})
    for u, v, p in out:
        if p > 0:
            jc = jc.append({"v1":int(u), "v2":int(v), "p":p}, ignore_index=True)

    # Randomly remove pct % of nodes if remove=True
    if remove:
        drop_indices = np.random.choice(E_ret.index, round(m*pct), replace=False)
        E_ret = E_ret.drop(drop_indices)

    E_ret["t"] = 0
    E_ret["p"] = 1

    # Add top pct % of nodes based on JC
    jc = jc.sort_values(by="p", axis=0, ascending=False)
    jc = jc.head(round(m*pct))
    jc["t"] = 0
    E_ret= E_ret.append(jc)

    for i in range(1, steps):
        E_prev = E_ret[(E_ret["t"]==(i-1))]
        G = nx.from_pandas_edgelist(E_prev, source="v1", target="v2").to_undirected()
        n = G.number_of_nodes()
        m = G.number_of_edges()
        out = nx.jaccard_coefficient(G)

        jc = pd.DataFrame({"v1":[0], "v2":[0], "p":[0]})
        for u, v, p in out:
            if p > 0:
                jc = jc.append({"v1":int(u), "v2":int(v), "p":p}, ignore_index=True)

        # Randomly remove pct % of nodes if remove=True
        if remove:
            drop_indices = np.random.choice(E_prev.index, round(m*pct), replace=False)
            E_prev = E_prev.drop(drop_indices)

        E_prev["t"] = i
        E_prev["p"] = 1

        # Add top pct % of nodes based on JC
        jc = jc.sort_values(by="p", axis=0, ascending=False)
        jc = jc.head(round(m*pct))
        jc["t"] = i
        E_prev = E_prev.append(jc)
        E_ret  = E_ret.append(E_prev)

    # Keep or don't keep edge probabilities based on JC
    if probs==False:
        E_ret["p"]=1

    return E_ret
