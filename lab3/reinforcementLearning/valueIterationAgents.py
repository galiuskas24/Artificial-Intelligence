# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()

        # --------------Train values (usefulness)------------------
        for iteration in range(self.iterations):
            iteration_values = util.Counter()

            for state in self.mdp.getStates():

                # We can not calculate for terminal/final state
                if self.mdp.isTerminal(state): continue

                # Calculate max Q value for temp state
                q_values = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                iteration_values[state] = 0. if len(q_values) == 0 else max(q_values)

            # Update values after every iteration
            self.values = iteration_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0.
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += prob*(self.mdp.getReward(state, action, next_state) + self.discount*self.getValue(next_state))

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Terminal state do not have action
        if self.mdp.isTerminal(state): return None

        # Calculate best Q value
        actions = self.mdp.getPossibleActions(state)
        q_values = [self.getQValue(state, action) for action in actions]

        return actions[util.maxIndexWithUniformSelction(q_values)]
        # return actions[q_values.index(max(q_values))] # NOT uniform

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
