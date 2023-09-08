def evolveGCN_linkpred(E, steps=1, no_seeds=1, node2vec_dim = 16, epochs=200, num_cores=6, rho=0.8):
    n = max(E["v2"]) + 1
    T = list(pd.unique(E["t"]))
    # Create Temporal Data Snapshots
    # Since we have no node-level features, we must use node2vec to generate them
    # Remember to make the edges undirected
    # All edges have weight one
    # No targets
    features      = [ np.zeros((n, node2vec_dim)) for x in range(len(T)) ]
    edge_indices = [ [] for x in range(len(T)) ]
    edge_weights = [ [] for x in range(len(T)) ]
    targets      = [ np.zeros(n) for x in range(len(T)) ]

    for t in range(len(T)):
        # Get node2vec embeddings
        G = nx.from_pandas_edgelist(E[E["t"]==T[t]], source="v1", target="v2").to_undirected()
        node2vec = Node2Vec(G, dimensions=node2vec_dim, workers=num_cores)
        model = node2vec.fit()
        nodes = pd.unique(G.nodes())
        for i in range(len(nodes)):
            features[t][nodes[i],] = model.wv[i]

        # Get edges
        edge_indices[t] = np.array([ list(E["v1"][E["t"]==T[t]])+ list(E["v2"][E["t"]==T[t]]),
                                list(E["v2"][E["t"]==T[t]])+ list(E["v1"][E["t"]==T[t]]) ])
        edge_weights[t] = np.ones(len(edge_indices[t][0]))

    # Create dataset
    dataset = DynamicGraphTemporalSignal(edge_indices = edge_indices,
                                         edge_weights = edge_weights,
                                         features     = features,
                                         targets      = targets)
    dataset.num_features = node2vec_dim

    # Compute (weighted) overall edge density
    norm_fact = 0
    phat      = 0
    tt        = 0
    for snapshot in dataset:
        phat      = phat + (rho**(max(T)-tt))*snapshot.edge_index.shape[1]/(n*(n-1))
        norm_fact = norm_fact + rho**(max(T)-tt)
        tt        += 1
    phat = phat/norm_fact
    print(phat)

    # Train/test split (use 100% of the data for training)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=1.00)

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features):
            super().__init__()
            self.recurrent = EvolveGCNO(node_features)
            self.linear = torch.nn.Linear(node_features, node2vec_dim)

        def encode(self, x, edge_index):
            h = self.recurrent(x, edge_index).relu()
            h = self.linear(h)
            return h

        # Just taking a dot product on node embeddings to get edge output
        def decode(self, z, edge_label_index):
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

        #def decode_all(self, z):
        #    prob_adj = z @ z.t()
        #    return (prob_adj > 0).nonzero(as_tuple=False).t()

        def decode_all(self, z):
            #prob_adj = (z @ z.t()).sigmoid()
            prob_adj = (z @ z.t())
            # Set diagonal to zero
            n = prob_adj.shape[0]
            for i in range(n):
                prob_adj[i,i] = 0
            return prob_adj

    model = RecurrentGCN(node2vec_dim).to("cpu")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in tqdm(range(1,epochs+1)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            z = model.encode(snapshot.x, snapshot.edge_index)

            # Can modify this part to only look at edge pairs with at least one link, instead of negative sampling
            # Make this method similar to LR LASSO method

            # We perform a new round of negative sampling for every training epoch:
            neg_edge_index = negative_sampling(
                edge_index=snapshot.edge_index, num_nodes=snapshot.num_nodes,
                num_neg_samples=int(snapshot.edge_index.size(1)/2), method='sparse')
                #num_neg_samples=snapshot.edge_label_index.size(1), method='sparse')

            # Make edges undirected
            source = neg_edge_index[0]
            target = neg_edge_index[1]

            target = torch.cat((neg_edge_index[0], target))
            source = torch.cat((neg_edge_index[1], source))

            neg_edge_index = torch.stack((target, source))

            edge_label_index = torch.cat(
                [snapshot.edge_index, neg_edge_index],
                dim=-1,
            )

            edge_label = torch.cat([
                torch.ones(len(snapshot.edge_index[0])),
                torch.zeros(neg_edge_index.size(1))
            ], dim=0).long()

            out = model.decode(z, edge_label_index).view(-1)

            loss = criterion(out, edge_label.float())
            cost = cost + loss

        cost = cost / (time+1)
        cost.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    # Now that model is fit, find the most important nodes and predict "steps" future time steps
    model.eval()

    # Find important nodes
    z = model.encode(snapshot.x, snapshot.edge_index)
    P = (z @ z.t())
    P = P.detach().numpy()
    # Set diagonal to zero
    for i in range(n):
        P[i,i] = 0

    # Compute sum of each row to see which nodes are most important
    d = {'node': range(0,n), 'sum':np.sum(P, axis=0)}
    df_node = pd.DataFrame(data=d)
    S_sum = list(df_node.sort_values(by="sum", ascending=False).head(no_seeds)["node"])


    # Predict "steps" ahead
    for step in range(steps):
        if step==0:
            z = model.encode(snapshot.x, snapshot.edge_index) # encode/decode most recent time step to predict next step
            out_all = model.decode_all(z).detach().numpy()

            # Extract edge indices of kept links
            C = np.quantile(out_all[np.triu_indices(n, k=1)], 1-phat)

            edge_indices_new = np.where(out_all > C)

            E_out = pd.DataFrame(np.transpose(edge_indices_new), columns=["v1", "v2"])
            # Remove duplicates
            E_out = E_out[E_out["v1"] < E_out["v2"]]
            E_out["t"] = 0


        else:
            z = model.encode(snapshot.x, snapshot.edge_index) # encode/decode most recent time step to predict next step
            out_all = model.decode_all(z).detach().numpy()

            # Extract edge indices of kept links
            C = np.quantile(out_all[np.triu_indices(n, k=1)], 1-phat)

            edge_indices_new = np.where(out_all > C)



            E_add = pd.DataFrame(np.transpose(edge_indices_new), columns=["v1", "v2"])
            # Remove duplicates
            E_add = E_add[E_add["v1"] < E_add["v2"]]

            E_add["t"] = step
            E_out = pd.concat([E_out, E_add])

        if step==(steps-1): # return output once number of steps has been reached
            E_out["p"] = 1
            return E_out, S_sum

        else: # otherwise, get dataset ready for next prediction
            edge_weights_new = np.ones(len(edge_indices_new[0]))
            features_new     = np.zeros((n, node2vec_dim))
            targets_new      = np.zeros(n)

            # Get node2vec embeddings
            G = nx.from_pandas_edgelist(E_out[E_out["t"]==step], source="v1", target="v2").to_undirected()
            node2vec = Node2Vec(G, dimensions=node2vec_dim, workers=num_cores)
            modeln2v = node2vec.fit()
            nodes = pd.unique(G.nodes())
            for i in range(len(nodes)):
                features_new[nodes[i],] = modeln2v.wv[i]
            # Update dataset with predicted edges
            edge_indices.append(edge_indices_new)
            edge_weights.append(edge_weights_new)
            features.append(features_new)
            targets.append(targets_new)

            dataset = DynamicGraphTemporalSignal(edge_indices = edge_indices,
                                                 edge_weights = edge_weights,
                                                 features     = features,
                                                 targets      = targets)
            dataset.num_features = node2vec_dim

            # Access the most recent snapshot
            for time, snapshot_prob in enumerate(dataset):
                0
