def nmf_map(xxx, A, k, T, q=0.05, lam=0.8, rho=0.8, eps=0.0001, max_iter=250):

    n = A[:,:,0].shape[0]
    # Initialize matrices
    U = np.zeros((n,k,T))
    V = np.zeros((n,k,T))

    for i in range(n):
        for j in range(k):
            for t in range(T):
                U[i,j,t] = random.random()
                V[i,j,t] = random.random()


    Ustar = np.zeros((n,k))
    Vstar = np.zeros((n,k))
    M     = 0

    for t in range(T):
        Ustar += (lam**(T-t)) * U[:,:,t]
        Vstar += (lam**(T-t)) * V[:,:,t]
        M += lam**(T-t)

    Ustar /= M
    Vstar /= M


    loss = 0
    for t in range(T):
        loss += lam**(T-t) * np.linalg.norm(A[:,:,t].astype("float64") - np.matmul(U[:, :, t].astype("float64"), np.transpose(V[:,:,t]).astype("float64")), ord="fro")**2 + lam**(T-t) * np.linalg.norm(U[:,:,t].astype("float64")-Ustar.astype("float64"))**2 + lam**(T-t) * np.linalg.norm(V[:,:,t].astype("float64")-Vstar.astype("float64"))**2

    converge = False
    iters=0

    while converge==False and iters < max_iter:
        converge = True
        iters    += 1
        AV    = np.zeros((n,k,T))
        AtV   = np.zeros((n,k,T))

        AU    = np.zeros((k,n,T))
        AtU   = np.zeros((k,n,T))

        # Iteratively update values
        for t in range(T):
            AV[:, :, t]  = np.matmul(A[:, :, t].astype("float64"), V[:, :, t].astype("float64"))
            AtV[:, :, t] = np.matmul(U[:, :, t].astype("float64"), np.matmul(np.transpose(V[:, :, t]).astype("float64"), V[:, :, t]).astype("float64")  )
            U[:,:,t] = np.multiply(U[:,:,t].astype("float64"), np.divide(AV[:, :, t].astype("float64") + Ustar.astype("float64"), AtV[:, :, t].astype("float64") + U[:,:,t].astype("float64")) )

            AU[:, :, t]  = np.matmul(np.transpose(U[:, :, t]).astype("float64"), A[:, :, t].astype("float64"))
            AtU[:, :, t] = np.matmul(np.transpose(U[:, :, t]).astype("float64"), np.matmul(U[:, :, t].astype("float64"), np.transpose(V[:, :, t]).astype("float64")  ))
            V[:,:,t] = np.multiply(V[:,:,t].astype("float64"), np.divide(np.transpose(AU[:, :, t]).astype("float64") + Vstar.astype("float64"), np.transpose(AtU[:, :, t]).astype("float64") + V[:,:,t].astype("float64")) )


        Ustar = np.zeros((n,k))
        Vstar = np.zeros((n,k))
        for t in range(T):
            Ustar += (lam**(T-t)) * U[:,:,t] / M
            Vstar += (lam**(T-t)) * V[:,:,t] / M

        loss_new = 0
        for t in range(T):
            loss_new += lam**(T-t) * np.linalg.norm(A[:,:,t].astype("float64") - np.matmul(U[:, :, t].astype("float64"), np.transpose(V[:,:,t]).astype("float64")), ord="fro")**2 + lam**(T-t) * np.linalg.norm(U[:,:,t].astype("float64")-Ustar.astype("float64"))**2 + lam**(T-t) * np.linalg.norm(V[:,:,t].astype("float64")-Vstar.astype("float64"))**2
        if abs(loss-loss_new)/loss_new > eps:
            converge = False
        loss = loss_new

    return U, AU, AtU, Ustar, V, AV, AtV, Vstar, loss

# Link prediction with non-negative matrix factorization
# Outputs adjacency matrix with link probabilties for each edge pair at next time step
def nmf_linkpred(E, mem=False, steps=1, no_seeds=1, q=0.05, lam=0.8, rho=0.8, eps=0.0001, max_iter=250, num_inits=25, mc_cores=6):

    # Convert edge list to adjacency matrix
    n = max(pd.unique(E["v2"])) + 1
    T = max(pd.unique(E["t"])) + 1
    A = np.zeros((n,n,T))

    for index, row in E.iterrows():
        A[row[0], row[1], row[2]] = 1
        A[row[1], row[0], row[2]] = 1


    # Compute (weighted) edge density
    norm_fact = 0
    phat      = 0
    for t in range(1,T+1):
        phat      = phat + (rho**(T-t))*(E[E["t"]==(t-1)].shape[0]/(0.5*n*(n-1)))
        norm_fact = norm_fact + rho**(T-t)

    phat = phat/norm_fact

    # Dimension of latent space
    k = round(q*n)

    # Normalizing constatn
    M     = 0
    for t in range(T):
        M += lam**(T-t)

    # Find U, V, Ustar, Vstar for different starting values
    pool = get_context("fork").Pool(mc_cores)
    out = pool.starmap(nmf_map, zip(range(num_inits), repeat(A), repeat(k), repeat(T), repeat(q), repeat(lam), repeat(rho), repeat(eps), repeat(max_iter)))
    pool.close()

    # Keep result with lowest value of loss function
    vec_out = np.zeros(num_inits)
    for i in range(num_inits):
        vec_out[i] = out[i][8]

    idx = np.argmin(vec_out)

    U = out[idx][0]
    AU = out[idx][1]
    AtU = out[idx][2]
    Ustar = out[idx][3]
    V = out[idx][4]
    AV = out[idx][5]
    AtV = out[idx][6]
    Vstar = out[idx][7]
    loss_ret = loss = out[idx][8]

    # Output
    print("Model fit.")

    # Compute sum of each row to see which nodes are most likely to have edges
    P = np.matmul(Vstar.astype("float64"), np.transpose(Vstar).astype("float64"))
    X = np.array(np.linalg.norm(Vstar.astype("float64"), axis=1))
    X = np.matmul(X[:, None], X[None, :])
    P = np.divide(P, X)

    d = {'node': range(0,n), 'sum':np.sum(P, axis=0)}
    df_node = pd.DataFrame(data=d)

    S_sum = list(df_node.sort_values(by="sum", ascending=False).head(no_seeds)["node"])

    # Predict steps ahead
    Ahat = np.zeros((n,n,1))

    # Only consider edge pairs with a last link if mem. Otherwise, consider all edge pairs
    if mem:
        for i in range(1,n):
            for j in range(i):
                if ((E["v1"]==j) & (E["v2"]==i)).any():
                    Ahat[i,j,0] = Ahat[j,i,0] = np.dot(Vstar[i,:].astype("float64"), Vstar[j,:].astype("float64")) / (np.linalg.norm(Vstar[i,:].astype("float64")) * np.linalg.norm(Vstar[j,:].astype("float64")))
    else:
        for i in range(1,n):
            for j in range(i):
                Ahat[i,j,0] = Ahat[j,i,0] = np.dot(Vstar[i,:].astype("float64"), Vstar[j,:].astype("float64")) / (np.linalg.norm(Vstar[i,:].astype("float64")) * np.linalg.norm(Vstar[j,:].astype("float64")))


    # Keep edge if similarity is greater than some cutoff which preserves density
    C = np.quantile(Ahat[np.triu_indices(n, k=1)], 1-phat)

    for i in range(1,n):
        for j in range(i):
            if Ahat[i,j,0] > C:
                Ahat[i,j,0] = Ahat[j, i,0] = 1
            else:
                Ahat[i,j,0] = Ahat[j, i,0] = 0

    if steps==1:
        Aout = Ahat

    else:
        Aout = Ahat

        for step in range(1,steps):
            A = np.concatenate((A, Ahat), axis=2)
            T += 1
            converge = False
            iters=0

            # Append to existing matrices
            Uadd = np.zeros((n,k,1))
            Vadd = np.zeros((n,k,1))

            for i in range(n):
                for j in range(k):
                    Uadd[i,j,0] = random.random()
                    Vadd[i,j,0] = random.random()

            U   = np.concatenate((U, Uadd), axis=2)
            V   = np.concatenate((V, Vadd), axis=2)
            AV  = np.concatenate((AV, np.zeros((n,k,1))), axis=2)
            AtV = np.concatenate((AtV, np.zeros((n,k,1))), axis=2)
            AU  = np.concatenate((AU, np.zeros((k,n,1))), axis=2)
            AtU = np.concatenate((AtU, np.zeros((k,n,1))), axis=2)

            while converge==False and iters < max_iter:
                converge = True
                iters    += 1

                for t in range(T):
                    AV[:, :, t]  = np.matmul(A[:, :, t].astype("float64"), V[:, :, t].astype("float64"))
                    AtV[:, :, t] = np.matmul(U[:, :, t].astype("float64"), np.matmul(np.transpose(V[:, :, t]).astype("float64"), V[:, :, t]).astype("float64")  )
                    U[:,:,t] = np.multiply(U[:,:,t].astype("float64"), np.divide(AV[:, :, t].astype("float64") + Ustar.astype("float64"), AtV[:, :, t].astype("float64") + U[:,:,t].astype("float64")) )

                    AU[:, :, t]  = np.matmul(np.transpose(U[:, :, t]).astype("float64"), A[:, :, t].astype("float64"))
                    AtU[:, :, t] = np.matmul(np.transpose(U[:, :, t]).astype("float64"), np.matmul(U[:, :, t].astype("float64"), np.transpose(V[:, :, t]).astype("float64")  ))
                    V[:,:,t] = np.multiply(V[:,:,t].astype("float64"), np.divide(np.transpose(AU[:, :, t]).astype("float64") + Vstar.astype("float64"), np.transpose(AtU[:, :, t]).astype("float64") + V[:,:,t].astype("float64")) )


                Ustar = np.zeros((n,k))
                Vstar = np.zeros((n,k))
                for t in range(T):
                    Ustar += (lam**(T-t)) * U[:,:,t] / M
                    Vstar += (lam**(T-t)) * V[:,:,t] / M

                loss_new = 0
                for t in range(T):
                    loss_new += lam**(T-t) * np.linalg.norm(A[:,:,t].astype("float64") - np.matmul(U[:, :, t].astype("float64"), np.transpose(V[:,:,t]).astype("float64")), ord="fro")**2 + lam**(T-t) * np.linalg.norm(U[:,:,t].astype("float64")-Ustar.astype("float64"))**2 + lam**(T-t) * np.linalg.norm(V[:,:,t].astype("float64")-Vstar.astype("float64"))**2
                if abs(loss-loss_new)/loss_new > eps:
                    converge = False
                loss = loss_new

            # Output
            Ahat = np.zeros((n,n,1))

            # Only consider edge pairs with a last link if mem. Otherwise, consider all edge pairs
            if mem:
                for i in range(1,n):
                    for j in range(i):
                        if ((E["v1"]==j) & (E["v2"]==i)).any():
                            Ahat[i,j,0] = Ahat[j,i,0] = np.dot(Vstar[i,:].astype("float64"), Vstar[j,:].astype("float64")) / (np.linalg.norm(Vstar[i,:].astype("float64")) * np.linalg.norm(Vstar[j,:].astype("float64")))
            else:
                for i in range(1,n):
                    for j in range(i):
                        Ahat[i,j,0] = Ahat[j,i,0] = np.dot(Vstar[i,:].astype("float64"), Vstar[j,:].astype("float64")) / (np.linalg.norm(Vstar[i,:].astype("float64")) * np.linalg.norm(Vstar[j,:].astype("float64")))

            # Keep edge if similarity is greater than some cutoff which preserves density
            C = np.quantile(Ahat[np.triu_indices(n, k=1)], 1-phat)
            for i in range(1,n):
                for j in range(i):
                    if Ahat[i,j,0] > C:
                        Ahat[i,j,0] = Ahat[j, i,0] = 1
                    else:
                        Ahat[i,j,0] = Ahat[j, i,0] = 0



            Aout = np.concatenate((Aout, Ahat), axis=2)

    # Convert adjacency matrix to edge list
    edges = []
    for t in range(steps):
        for i in range(1,n):
            for j in range(i):
                if Aout[i,j,t]==1:
                    edges.append([j, i, t, 1])

    edges = pd.DataFrame(edges)
    edges = edges.rename({0: "v1", 1:"v2", 2:"t", 3:"p"}, axis=1)
    return edges, S_sum
