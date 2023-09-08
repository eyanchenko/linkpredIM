def expit(x):

    if x > 100:
        return 1
    elif x < -100:
        return 0
    else:
        return 1/(1+math.exp(-x))

def map_auc(i, X_train, X_test, Y_train, Y_test, CC):
    if len(pd.unique(Y_train[i])) > 1 and len(pd.unique(Y_test[i])) > 1:
        fit = LogisticRegression(penalty="l1", C=CC, solver="saga").fit(X_train, Y_train[i])

        y_true = Y_test[i]
        y_pred = fit.predict_proba(X_test).transpose()[1]
        return roc_auc_score(y_true, y_pred)
    else:
        return 0


def map_pred(i, X_fit, X_pred, Y_fit, Cstar):

    if len(pd.unique(Y_fit[i])) > 1:
        fit = LogisticRegression(penalty="l1", C=Cstar, solver="saga").fit(X_fit, Y_fit[i])
        Y_pred = fit.predict_proba(X_pred.reshape(1, -1)).transpose()[1]
        return Y_pred

    else:
        return float(pd.unique(Y_fit[i]))

def map_beta(i, X_fit, Y_fit, Cstar):

    if len(pd.unique(Y_fit[i])) > 1:
        fit = LogisticRegression(penalty="l1", C=Cstar, solver="saga").fit(X_fit, Y_fit[i])
        return fit.intercept_, fit.coef_

    elif int(pd.unique(Y_fit[i]))==1:
        return 10, [10]*X_fit.shape[1]

    elif int(pd.unique(Y_fit[i]))==0:
        return -10, [-10]*X_fit.shape[1]

def net_pred_lasso_mult(edges, no_seeds=1, pct_train=0.75, mc_cores=7, steps=1, rho=0.8, penalty=0, probs=False, keep_edge="prob"):

    # Create dataframe which has the edges observed across all times
    # Will use this to create the feature vector
    edge_vec = edges.copy()
    del edge_vec["t"]
    edge_vec = edge_vec.drop_duplicates().reset_index()
    del edge_vec["index"]
    M = edge_vec.shape[0]
    edge_vec["idx"] = range(M)

    E = edges.copy()
    E = E.drop_duplicates().reset_index()
    del E["index"]

    # Create feature vector for each time step
    T = pd.unique(E["t"])     # unique time snapshot
    X = np.zeros((len(T), M)) # feature vector X_ij = 1 if edge pair j has edge at time i

    for i in range(E.shape[0]):
        # Extract edge pair and time
        v1       = E["v1"][i]
        v2       = E["v2"][i]
        # Find which index this edge pair corresponds to in the edge vector
        idx = int(edge_vec["idx"].loc[(edge_vec["v1"] == v1)* (edge_vec["v2"] == v2)])
        X[T==E["t"][i], idx] = 1

    print("Data in right format.")


    n = max(E["v2"]) + 1
    # Compute (weighted) edge density
    norm_fact = 0
    phat      = 0
    for t in T:
        phat      = phat + (rho**(max(T)-t))*(E[E["t"]==t].shape[0]/(0.5*n*(n-1)))
        norm_fact = norm_fact + rho**(max(T)-t)

    phat = phat/norm_fact
    print(phat)

    # Note that Y at time t+1 regresses on X at time t
    # Ignore last time step because we can't regress it on anything as we don't know links at next time step
    X_pred = X[-1,]
    X_fit  = X[range(0,len(T)-1),]

    # Seperate into train/test split
    n_train   = round(pct_train * X_fit.shape[0] )
    idx_train = random.sample(range(X_fit.shape[0]), n_train)
    idx_test  = list(set(range(X_fit.shape[0])) - set(idx_train))

    X_train   = X_fit[idx_train, ]
    X_test    = X_fit[idx_test, ]

    # Create response vector
    # Y_i = [0, 1, 1, 0] corresponds to the activation/deactivation of link pair i across times
    Y = X.copy()
    # Drop first time step because we don't know it's features
    Y_fit   = Y[range(1, len(T))]

    Y_train = Y_fit[idx_train, ].transpose()
    Y_test  = Y_fit[idx_test, ].transpose()

    # Now, we want to regress each entry of Y_train on X_train
    # Then test performance on test set to find optimal penalty parameter
    # Note that if edge pair i has an edge / no edges for all times t then we must predict it will/won't have an edge with
    # 100% certainty
    # Smaller C means more regularization

    if penalty==0:

        alpha = [0.01, 0.05, 0.1, 0.50, 1, 5, 10, 50, 100, 1000]
        auc   = [0]*len(alpha)

        for a in range(len(alpha)):
            pool = get_context("fork").Pool(mc_cores)
            out = pool.starmap(map_auc, zip(range(M), repeat(X_train), repeat(X_test), repeat(Y_train), repeat(Y_test), repeat(alpha[a])) )
            pool.close()

            if statistics.mean(out) > 0:
                auc[a] = statistics.mean([i for i in out if i >0])
            else:
                auc[a] = 0

            if auc[a]==1:
                auc[a]=0
            print("AUC for alpha =", round(1/alpha[a],3), "is", round(auc[a],3))
            Cstar = alpha[auc.index(max(auc))]
    else:
        Cstar = 1/penalty

    # Use penalty parameter which corresponded to best performance and re-train on entire dataset
    # Then compute probability of a link for each edge pair for next network in sequence
    # Note that if edge pair i has an edge / no edges for all times t then we must predict it will/won't have an edge with
    # 100% certainty
    # Smaller C means more regularization

    E_predict = [ edge_vec[["v1", "v2"]].copy() for x in range(steps)]

    Y_fit = Y_fit.transpose()

    pool = get_context("fork").Pool(mc_cores)
    out = pool.starmap(map_beta, zip(range(M), repeat(X_fit), repeat(Y_fit), repeat(Cstar)) )
    pool.close()

    print("Model fit")

    # Find seed nodes for IM
    d = {'node': range(n), "theta": 0.0}
    df = pd.DataFrame(data=d)

    for i in range(M):
        if out[i][0] > -10:
            pp = expit(out[i][0] + np.sum(X_pred * out[i][1]))
            v1 = edge_vec["v1"][i]
            v2 = edge_vec["v2"][i]

            df["theta"][v1] = df["theta"][v1] + pp
            df["theta"][v2] = df["theta"][v2] + pp

    S_sum = list(df.sort_values(by="theta", ascending=False).head(no_seeds)["node"])

    # Predict steps ahead
    INT  = np.zeros(M)
    BETA = np.zeros((M, M))

    for i in range(M):
        INT[i]    = out[i][0]
        BETA[i, ] = out[i][1]

    pool = get_context("fork").Pool(mc_cores)
    E_predict[0]["p"] = pool.map(expit, INT + X_pred @ BETA )
    pool.close()

    #E_predict[0]["p"] = INT + X_pred @ BETA

    for i in range(1,steps):
        pool = get_context("fork").Pool(mc_cores)
        E_predict[i]["p"] = pool.map(expit, INT + E_predict[i-1]["p"] @ BETA )
        pool.close()
        print("Prediction",i,"done")

    if probs==False:
        for t in range(steps):
            # Find cutoff to preserve edge density phat
            E_predict[t] = E_predict[t].sort_values(by="p", axis=0, ascending=False)

            if keep_edge == "prob":
                E_predict[t] = E_predict[t].head(round(0.5*n*(n-1)*phat))
            elif keep_edge == "zero":
                E_predict[t] = E_predict[t][E_predict[t]["p"] > 0.5]
            E_predict[t]["p"] = 1

    E_out = E_predict[0]
    E_out["p"] = 1
    E_out["t"] = 0
    for t in range(1, steps):
        E_predict[t]["t"] = t
        E_out = pd.concat((E_out, E_predict[t]))

    # Switch column order
    return E_out[["v1", "v2", "t", "p"]], S_sum
