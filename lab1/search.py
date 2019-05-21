# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import copy

class SearchNode:
    """
    This class represents a node in the graph which represents the search problem.
    The class is used as a basic wrapper for search methods - you may use it, however
    you can solve the assignment without it.

    REMINDER: You need to fill in the backtrack function in this class!
    """

    def __init__(self, position, parent=None, transition=None, cost=0, heuristic=0):
        """
        Basic constructor which copies the values. Remember, you can access all the 
        values of a python object simply by referencing them - there is no need for 
        a getter method. 
        """
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.transition = transition

    def isRootNode(self):
        """
        Check if the node has a parent.
        returns True in case it does, False otherwise
        """
        return self.parent == None 

    def unpack(self):
        """
        Return all relevant values for the current node.
        Returns position, parent node, cost, heuristic value
        """
        return self.position, self.parent, self.cost, self.heuristic

    def backtrack(self):
        """
        Reconstruct a path to the initial state from the current node.
        Bear in mind that usually you will reconstruct the path from the 
        final node to the initial.
        """
        node = copy.deepcopy(self)

        if node.isRootNode(): 
            # The initial state is the final state
            return []

        # print self.cost, self.heuristic
        return [self.transition] + self.parent.backtrack()


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    return universalSearch(problem, util.Stack(), nullHeuristic)


def breadthFirstSearch(problem):
    return universalSearch(problem, util.Queue(), nullHeuristic)


def uniformCostSearch(problem):
    buffer = util.PriorityQueueWithFunction(lambda x: x[1])
    return universalSearch(problem, buffer, nullHeuristic)


def aStarSearch(problem, heuristic=nullHeuristic):
    buffer = util.PriorityQueueWithFunction(lambda x: x[1])
    return universalSearch(problem, buffer, heuristic)


def universalSearch(problem, buffer, heuristic):
    """
    This is universal greedy search method.
    There is no logic for already visited state. This method ignores them and continues search.

    :param problem:
    :param buffer:
    :param heuristic:
    :return:
    """
    root = SearchNode(position=problem.getStartState())
    buffer.push((root, 0))  # (node, priority)
    visited_states = []

    while not buffer.isEmpty():

        curr_node = buffer.pop()[0]
        curr_pos, curr_par, curr_cost, curr_heur = curr_node.unpack()

        # is finish?
        if problem.isGoalState(curr_pos):
            return list(reversed(curr_node.backtrack()))

        # is duplicate?
        if curr_pos in visited_states: continue

        # add to visited
        visited_states.append(curr_pos)

        # add next states to buffer
        for next_position, direction, cost in problem.getSuccessors(curr_pos):
            next_node = SearchNode(
                position=next_position,
                parent=curr_node,
                transition=direction,
                cost=(curr_cost + cost),
                heuristic=heuristic(next_position, problem)
            )
            buffer.push((next_node, next_node.cost + next_node.heuristic))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
