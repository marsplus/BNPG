import numpy as np
import numpy.linalg as lin
from abc import ABC, abstractmethod


## abstract class for agent
class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_id(self):
        pass

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def utility(self):
        pass




## the actual game agent used in our experiments
class GameAgent(Agent):
    def __init__(self, unique_id, c, Lambda, g):
        super().__init__()
        self.id = unique_id
        self.c = c
        self.Lambda = Lambda

        # g is the crime rate mapping in the paper
        assert(callable(g) is True)
        self.g = g

    def get_id(self):
        return self.id

    def utility(self, decision, numNeighborsReport):
        U =  self.Lambda * self.g(decision + numNeighborsReport) - self.c * decision
        return U

    # we consider best response as an agent's action
    def action(self, numNeighborsReport):
        if self.utility(1, numNeighborsReport) > self.utility(0, numNeighborsReport):
            return 1
        elif self.utility(1, numNeighborsReport) < self.utility(0, numNeighborsReport):
            return 0
        else:
            if np.random.uniform() <= 0.5:
                return 1
            else:
                return 0


## game agent for threshold game
## where the incentive structure is 
## strategic complement
class ThresholdAgent(Agent):
    def __init__(self, unique_id, c, Lambda):
        super().__init__()
        self.id = unique_id
        self.c = c
        self.Lambda = Lambda

    def get_id(self):
        return self.id

    def utility(self, decision, numNeighborsReport):
        if decision == 1:
            return self.Lambda * numNeighborsReport - self.c
        else:
            return 0

    # we consider best response as an agent's action
    def action(self, numNeighborsReport):
        if self.utility(1, numNeighborsReport) > self.utility(0, numNeighborsReport):
            return 1
        elif self.utility(1, numNeighborsReport) < self.utility(0, numNeighborsReport):
            return 0
        else:
            if np.random.uniform() <= 0.5:
                return 1
            else:
                return 0


## game agent for best-shot game
## where the incentive structure is 
## strategic substitute
class BestShotAgent(Agent):
    def __init__(self, unique_id, c):
        super().__init__()
        self.id = unique_id
        self.c = c

    def get_id(self):
        return self.id

    def utility(self, decision, numNeighborsReport):
        if decision == 1:
            return 1 - self.c
        elif decision == 0 and numNeighborsReport != 0:
            return 1
        elif decision == 0 and numNeighborsReport == 0:
            return 0
        else:
            raise ValueError("Best-shot agent: cannot compute utility\n")


    # we consider best response as an agent's action
    def action(self, numNeighborsReport):
        if self.utility(1, numNeighborsReport) > self.utility(0, numNeighborsReport):
            return 1
        elif self.utility(1, numNeighborsReport) < self.utility(0, numNeighborsReport):
            return 0
        else:
            if np.random.uniform() <= 0.5:
                return 1
            else:
                return 0


## class for network game environment
class Game(object):
    def __init__(self, G, AgentList, logPath, maxIter=10000):
        super().__init__()
        self.graph = G
        # AgentList: a set of initialized agents 
        self.players = AgentList

        self.numPlayer = len(self.players)
        self.globalState = np.zeros(self.numPlayer)
        self.epsilon = 0

        # number of iterations simulated
        self.iterations = 0
        self.logPath = logPath

        self.maxIter = maxIter
        self.foundPSNE = False

        self.numAsynStep = 0


    def revert(self):
        """
        Revert back to the initial global state
        """
        self.globalState = np.zeros(self.numPlayer)


    def isPSNE(self, state):
        """
        Check if a global state is a epsilon-PSNE
        """
        flag = True
        for Player in self.players:
            player_id = Player.get_id()
            player_decision = state[player_id]
            neighborhood = self.graph.neighbors(player_id)
            numNeighborsReport = np.sum([state[i] for i in neighborhood])
            if Player.utility(player_decision, numNeighborsReport) >= Player.utility(1 - player_decision, numNeighborsReport):
                flag = False
                break

        return True if flag == True else False


    def update(self, x, player_id, decision):
        """
        Update global state
        """ 
        self.globalState[player_id] = decision


    def asynchronous_step(self, x):
        """
        Iterate throught each agent and ask for its response and update immediatedly
        """
        x_copy = x.copy()
        for Player in self.players:
            player_id = Player.get_id()
            neighborhood = self.graph.neighbors(player_id)
            # the line below is the difference with synchronous_step
            numNeighborsReport = np.sum([x_copy[i] for i in neighborhood])
            player_action = Player.action(numNeighborsReport)
            x_copy[player_id] = player_action
        return x_copy


    def synchronous_step(self, x):
        """
        Iterate through each agent and ask for its response and update after every agent makes decision
        """
        tmp_x = x.copy()
        for Player in self.players:
            player_id = Player.get_id()
            neighborhood = self.graph.neighbors(player_id)
            # the line below is the difference with asynchronous_step
            numNeighborsReport = np.sum([x[i] for i in neighborhood])
            player_action = Player.action(numNeighborsReport)
            tmp_x[player_id] = player_action
        return tmp_x



    def maximumEpsilon(self, x):
        """
        Given a state x, calculate the maximum epsilon needed 
        to make x a PSNE.
        """
        maxEpsilon = 0
        for Player in self.players:
            player_id = Player.get_id()
            player_decision = x[player_id]
            neighborhood = self.graph.neighbors(player_id)
            numNeighborsReport = np.sum([x[i] for i in neighborhood])

            U_a = Player.utility(player_decision, numNeighborsReport)
            U_b = Player.utility(1 - player_decision, numNeighborsReport)
            Epsilon = (U_b - U_a) / np.max([np.abs(U_a), np.abs(U_b)])
            
            ## a player who could potentially deviate
            if Epsilon > maxEpsilon:
                maxEpsilon = Epsilon
        return maxEpsilon



    def evolve(self, x, evol_iter):
        """
        Evolve to generate the next seed
        """
        candidate_pool = []
        x_seed = x.copy()
        for Iter in range(evol_iter):
            Epsilon = self.maximumEpsilon(x_seed)
            print("evolution step: %i    MaxEpsilon: %.4f\n" % (Iter, Epsilon))

            ## found a PSNE
            if Epsilon == 0:
                self.foundPSNE = True
                return x_seed

            candidate_pool.append((x_seed, Epsilon))
            # use asynchronous best-response to generate the next global state
            x_seed = self.asynchronous_step(x_seed)
            self.numAsynStep += 1

        x_next = min(candidate_pool, key=lambda x: x[1])[0]
        currentEpsilon = min(candidate_pool, key=lambda x: x[1])[1]
        self.epsilon = currentEpsilon
        print("Current Epsilon: ", currentEpsilon)
        return x_next



    def evolutionary_step(self, x):
        """
        One evolutionary step
        evol_iter: the number of iterations to generate the next seed
        """
        evol_iter = 200
        x_next = self.evolve(x, evol_iter)
        return x_next


    # output debug info
    def output_log(self):
        with open(self.logPath, 'w') as fid:
            for Player in self.players:
                player_id = Player.get_id()
                neighborhood = self.graph.neighbors(player_id)
                numNeighborsReport = np.sum([self.globalState[i] for i in neighborhood])   
                player_decision = self.globalState[player_id]
                delta_g = Player.g(1 + numNeighborsReport) - Player.g(numNeighborsReport)
                c_vs_lambda = Player.c * 1.0 / Player.Lambda
                msg = "%i      decision: %i      (%.2f    v.s.   %.2f)\n" % (player_id, player_decision, c_vs_lambda, delta_g)
                fid.write(msg)       
    

    def utilityDistribution(self):
        U = []
        for Player in self.players:
            player_id = Player.get_id()
            neighborhood = self.graph.neighbors(player_id)
            numNeighborsReport = np.sum([self.globalState[i] for i in neighborhood])   
            player_decision = self.globalState[player_id]     
            player_utility = Player.utility(player_decision, numNeighborsReport)
            U.append(np.abs(player_utility))
            # U.append(player_utility)
        return U



    def simulate(self, mode='asynchronous'):
        """
        Simulate the network game until convergence
        """
        while not self.isPSNE(self.globalState):
            if (self.iterations+1) % 100 == 0:
                print("Iterations: ", self.iterations)

            if self.iterations < self.maxIter:
                if mode == 'synchronous':
                    self.globalState = self.synchronous_step(self.globalState)

                elif mode == 'asynchronous':
                    self.globalState = self.asynchronous_step(self.globalState)

                elif mode == 'evolutionary':
                    ## first time run evolutionary step
                    if self.iterations == 0:
                        self.globalState = np.random.choice([0, 1], size=self.numPlayer)
                    updated_state = self.evolutionary_step(self.globalState)

                    if lin.norm(updated_state - self.globalState, 1) < 1e-7:
                        if not self.isPSNE(updated_state):
                            print("Converge to an approximate PSNE with epsilon=%.6f\n" % self.epsilon)
                            return    
                    self.globalState = updated_state
                else:
                    raise ValueError("Invalid Best Response Dynamics")

                self.iterations += 1
            else:
                print("Max iterations reached\n")
                return 

        self.epsilon = 0
        self.foundPSNE = True












