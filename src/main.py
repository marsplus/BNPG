import os
import time
import pickle
import argparse
import numpy as np
import networkx as nx
from model import *
from multiprocessing import Pool
from collections import defaultdict

np.random.seed(123)


## (c_i, lambda_i, g)
def task_Homogeneous_concave(network, beta, numExp):
    if network == 'BA-1':
        G = nx.barabasi_albert_graph(128, 3) # 1: exponent =  2.7167
    elif network == 'BA-2':
        G = nx.barabasi_albert_graph(128, 4) # 2: exponent =  2.2789
    elif network == 'BA-3':
        G = nx.barabasi_albert_graph(128, 5) # 3: exponent =  2.0374

    elif network == 'Small-World-1':
        G = nx.watts_strogatz_graph(128, 10, 0.2) # 1 local clustering = 0.3667
    elif network == 'Small-World-2':
        G = nx.watts_strogatz_graph(128, 15, 0.2) # 1 local clustering = 0.3893
    elif network == 'Small-World-3':
        G = nx.watts_strogatz_graph(128, 20, 0.2) # 1 local clustering = 0.4070

    elif network == 'Facebook':
        G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
    else:
        raise ValueError()

    numAgent = len(G.nodes())
    logPath = 'log.txt'
    mapping = {1.2: '1_2', 1.5: '1_5', 2: '2'}

    ret = defaultdict(list)
    running_iterations = defaultdict(list)
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        g = lambda x: -alpha * np.power(x, beta) 
        for iteration in range(numExp):
            print("alpha=%.2f     iteration=%.i" % (alpha, iteration))
            AgentList = []
            for i in range(numAgent):
                c_i = np.random.uniform()
                Lambda_i = np.random.uniform()
                AgentList.append(GameAgent(i, c_i, Lambda_i, g))
            NetworkGame = Game(G, AgentList, logPath)
            t1 = time.time()
            NetworkGame.simulate("evolutionary")
            NetworkGame.output_log()
            t2 = time.time() - t1
            r_ratio = np.sum(NetworkGame.globalState) * 1.0 / numAgent
            ret[alpha].append(r_ratio)
            running_iterations[alpha].append(NetworkGame.iterations)
    with open("result/homogeneous_g_concave_beta_%s_%s.p" % (mapping[beta], network), 'wb') as fid:
        pickle.dump(dict(ret), fid, protocol=2)
    return ret



def task_Homogeneous_convex(network, beta, numExp):
    if network == 'BA-1':
        G = nx.barabasi_albert_graph(128, 3) # 1: exponent =  2.7167
    elif network == 'BA-2':
        G = nx.barabasi_albert_graph(128, 4) # 2: exponent =  2.2789
    elif network == 'BA-3':
        G = nx.barabasi_albert_graph(128, 5) # 3: exponent =  2.0374

    elif network == 'Small-World-1':
        G = nx.watts_strogatz_graph(128, 10, 0.2) # 1 local clustering = 0.3667
    elif network == 'Small-World-2':
        G = nx.watts_strogatz_graph(128, 15, 0.2) # 1 local clustering = 0.3893
    elif network == 'Small-World-3':
        G = nx.watts_strogatz_graph(128, 20, 0.2) # 1 local clustering = 0.4070

    elif network == 'Facebook':
        G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
    else:
        raise ValueError()

    numAgent = len(G.nodes())
    logPath = 'log.txt'
    mapping = {1.2: '1_2', 1.5: '1_5', 2: '2'}

    ret = defaultdict(list)
    running_iterations = defaultdict(list)
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        ## make sure the mapping g is non-negative all the time
        g = lambda x: - alpha * np.log(np.power(x + 1, beta)) 

        for iteration in range(numExp):
            print("alpha=%.2f     iteration=%.i" % (alpha, iteration))
            AgentList = []
            for i in range(numAgent):
                c_i = np.random.uniform()
                Lambda_i = np.random.uniform()
                AgentList.append(GameAgent(i, c_i, Lambda_i, g))
            NetworkGame = Game(G, AgentList, logPath)
            t1 = time.time()
            NetworkGame.simulate('evolutionary')
            NetworkGame.output_log()
            t2 = time.time() - t1
            r_ratio = np.sum(NetworkGame.globalState) * 1.0 / numAgent
            ret[alpha].append(r_ratio)
            running_iterations[alpha].append(NetworkGame.iterations)
    with open("result/homogeneous_g_convex_beta_%s_%s.p" % (mapping[beta], network), 'wb') as fid:
        pickle.dump(dict(ret), fid, protocol=2)
    return ret



def HeterogeneousBNPG_general(ratioConvex, G):
    # G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
    # G = nx.barabasi_albert_graph(128, 3)
    numAgent = len(G.nodes())
    logPath = 'log.txt'
    maxDegree = np.max([G.degree(i) for i in G.nodes()])
    
    alpha_list_convave = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    beta_list_convave = [1.2, 1.5, 2.0]

    alpha_list_convex = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    beta_list_convex = [1.2, 1.5, 2.0]

    agentList = []
    for i in range(numAgent):
        c_i = np.random.uniform()
        Lambda_i = np.random.uniform()

        isConvex = (np.random.uniform() <= ratioConvex)
        if isConvex:
            Alpha = np.random.choice(alpha_list_convave)
            Beta = np.random.choice(beta_list_convave)
            g = lambda x: Alpha * np.power(x, Beta) + np.abs(Alpha * np.power(maxDegree, Beta))
        else:
            Alpha = np.random.choice(alpha_list_convex)
            Beta = np.random.choice(beta_list_convex)
            g = lambda x: Alpha * np.log(np.power(x + 1, Beta)) + np.abs(Alpha * np.log(np.power(maxDegree + 1, Beta)))
        agentList.append(GameAgent(i, c_i, Lambda_i, g))   
    NetworkGame = Game(G, agentList, logPath)
    NetworkGame.simulate('evolutionary')
    return NetworkGame


def HeterogeneousBNPG_BestShot_Threshold(ratioConvex, G):
    numAgent = len(G.nodes())
    logPath = 'log.txt'
    maxDegree = np.max([G.degree(i) for i in G.nodes()])

    agentList = []
    for i in range(numAgent):
        isConvex = (np.random.uniform() <= ratioConvex)
        if isConvex:
            Lambda = np.random.uniform()
            c = np.random.uniform()
            agentList.append(ThresholdAgent(i, c, Lambda))
        else:
            c = np.random.uniform()
            agentList.append(BestShotAgent(i, c))
    NetworkGame = Game(G, agentList, logPath)
    NetworkGame.simulate('evolutionary')
    return NetworkGame



def dispatch(args):
    ratioConvex, G = args
    ret = HeterogeneousBNPG_BestShot_Threshold(ratioConvex, G)
    epsilon = ret.epsilon
    foundPSNE = ret.foundPSNE
    Iter = ret.numAsynStep
    return (foundPSNE, epsilon, ratioConvex, Iter)


# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('graph_type', type=str)
# parser.add_argument('numExp', type=int)
# args = parser.parse_args()
# graph_type, numExp = args.graph_type, args.numExp


numExp = 100
G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
pool = Pool(processes=7)
ret = []
for i in range(numExp):
    args = []
    for ratioConvex in np.arange(0, 1.1, 0.1):
        args.append((ratioConvex, G))
    ret.extend(pool.map(dispatch, args))


# for beta in [1.2, 1.5, 2]:
#     task_Homogeneous_concave(graph_type, beta, numExp)
#     task_Homogeneous_convex(graph_type, beta, numExp)

