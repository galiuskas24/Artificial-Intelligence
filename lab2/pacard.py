
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from logic import * 

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


def miniWumpusSearch(problem): 
    """
    A sample pass through the miniWumpus layout. Your solution will not contain 
    just three steps! Optimality is not the concern here.
    """
    from game import Directions
    e = Directions.EAST 
    n = Directions.NORTH
    return  [e, n, n]

def logicBasedSearch(problem):
    """

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    print "Does the Wumpus's stench reach my spot?", 
               \ problem.isWumpusClose(problem.getStartState())

    print "Can I sense the chemicals from the pills?", 
               \ problem.isPoisonCapsuleClose(problem.getStartState())

    print "Can I see the glow from the teleporter?", 
               \ problem.isTeleporterClose(problem.getStartState())
    
    (the slash '\\' is used to combine commands spanning through multiple lines - 
    you should remove it if you convert the commands to a single line)
    
    Feel free to create and use as many helper functions as you want.

    A couple of hints: 
        * Use the getSuccessors method, not only when you are looking for states 
        you can transition into. In case you want to resolve if a poisoned pill is 
        at a certain state, it might be easy to check if you can sense the chemicals 
        on all cells surrounding the state. 
        * Memorize information, often and thoroughly. Dictionaries are your friends and 
        states (tuples) can be used as keys.
        * Keep track of the states you visit in order. You do NOT need to remember the
        tranisitions - simply pass the visited states to the 'reconstructPath' method 
        in the search problem. Check logicAgents.py and search.py for implementation.
    """
    # -----------------INITIALIZATION------------------------
    global database, indicator, visitedStates
    global safeStates, unclearStates, safeStatesTrack, unclearStatesTrack, forbiddenStates
    visitedStates = []
    database, safeStatesTrack, unclearStatesTrack, forbiddenStates = set(), set(), set(), set(),

    safeStates = util.PriorityQueueWithFunction(stateWeight)
    unclearStates = util.PriorityQueueWithFunction(stateWeight)

    indicator = {Labels.WUMPUS_STENCH: problem.isWumpusClose,
                 Labels.POISON_FUMES: problem.isPoisonCapsuleClose,
                 Labels.TELEPORTER_GLOW: problem.isTeleporterClose}

    # ---------------------START-----------------------------
    safeStates.push(problem.getStartState())
    safeStatesTrack.add(problem.getStartState())

    while True:

        # choose next state
        if not safeStates.isEmpty():
            curr_state = safeStates.pop()

        elif not unclearStates.isEmpty():
            curr_state = unclearStates.pop()

            # because we can not delete from PriorityQueueWithFunction
            if curr_state not in unclearStatesTrack: continue

            # maybe we have new knowledge for unclear state
            conclude = concludeWTP(curr_state)

            if conclude[Labels.POISON] or conclude[Labels.WUMPUS]:
                forbiddenStates.add(curr_state)
                continue
        else:
            print 'Game over: No more available positions!'
            return problem.reconstructPath(visitedStates)

        if curr_state in visitedStates: continue
        visitedStates.append(curr_state)
        print 'Visiting:', curr_state

        if problem.isGoalState(curr_state):
            print 'Game over: Teleported home!'
            return problem.reconstructPath(visitedStates)

        successors = problem.getSuccessors(curr_state)
        sniff(curr_state, successors)

        for position, _, _a in successors:
            if position in visitedStates: continue
            if generalConclude(position):
                print 'Game over: Teleported home!'
                return problem.reconstructPath(visitedStates)


def sniff(state, successors):
    """
    In this method, we add clauses in the database obtained by sniffing from the given state.
    :param state:
    :param successors:
    :return:
    """
    for indicator_type, obj in Labels.INDICATOR_OBJECT_PAIRS:
        isObjectNearby = indicator[indicator_type](state)

        if isObjectNearby:
            literals = {Literal(obj, position, negative=False) for position, _, _a in successors}
            database.add(Clause(literals))
        else:
            for position, _, _a in successors: database.add(Clause({Literal(obj, position, negative=True)}))

        print 'Sensed:', '~' + indicator_type if not isObjectNearby else indicator_type, state

    if not indicator[Labels.WUMPUS_STENCH](state) and not indicator[Labels.POISON_FUMES](state):
        # all successors are SAFE (concluded only by sniff from the current state -> without database)
        for position, _, _a in successors: database.add(Clause({Literal(Labels.SAFE, position, False)}))


def generalConclude(position):
    """
    This method try to do every possible conclusion for given position and database.
    :param position:
    :return:
    """
    concluded = concludeWTP(position)
    goal = Clause({Literal(Labels.SAFE, position, False)})

    if concluded[Labels.TELEPORTER]:
        visitedStates.append(position)
        return True

    if concluded['~' + Labels.POISON] and concluded['~' + Labels.WUMPUS]:
        # position is SAFE
        database.add(goal)
        print 'Concluded: o', position

        if position in unclearStatesTrack:
            unclearStatesTrack.remove(position)

        if position not in safeStatesTrack:
            safeStatesTrack.add(position)
            safeStates.push(position)

    else:
        # we do not know is position SAFE
        database.add(goal.negateAll().pop())

        # is it forbidden
        if concluded[Labels.POISON] or concluded[Labels.WUMPUS]:
            forbiddenStates.add(position)
        else:
            unclearStatesTrack.add(position)
            unclearStates.push(position)

    # conclude
    objPosition = concludeOneTimeOccurredObject(Labels.TELEPORTER)

    if objPosition is not None:
        visitedStates.append(objPosition)
        return True

    return False


def concludeWTP(position):
    """
    This method make a conclusion for given position.
    Using database and resolution, we try to detect WTP object on given position.
    :param position:
    :return: dict
    """
    WTP_objects = []
    concludeVector = []

    for obj in [Labels.WUMPUS, Labels.POISON, Labels.TELEPORTER]:
        for isNegative in [False, True]:
            WTP_objects.append('~' + obj if isNegative else obj)
            goal = Clause({Literal(obj, position, isNegative)})

            concludeResult = resolution(database, goal)
            concludeVector.append(concludeResult)

            if concludeResult:
                database.add(goal)
                print 'Concluded:', WTP_objects[-1], position

    return dict(zip(WTP_objects, concludeVector))


def concludeOneTimeOccurredObject(item):
    """
    This method try to conclude position of object which appears one time. (Wumpus or Teleporter)
    Logic behind this method is find two clauses which contains our object and find overlapping positions.
    Then try to conclude where is object in overlapping positions.
    :param item: only Labels.TELEPORTER or Labels.WUMPUS
    :return: position or None
    """
    objectClauses = []

    # find two clauses that contain positive literal (contain item)
    for clause in database:
        if item in [str(literal) for literal in clause.literals]:
            objectClauses.append(clause.literals)

        if len(objectClauses) == 2: break

    if len(objectClauses) != 2: return None

    # when we find them, do intersection and find overlapping position/s
    clause1, clause2 = objectClauses
    intersection = set.intersection(clause1, clause2)
    objectPosition = None

    if len(intersection) == 1:
        objectPosition = intersection.pop().state

    elif len(intersection) == 2:
        new_list = []

        # try to conclude object position with removing visited states
        for literal in intersection:
            if literal.state not in visitedStates: new_list.append(literal.state)

        objectPosition = new_list[0] if len(new_list) == 1 else None

    return objectPosition

# Abbreviations
lbs = logicBasedSearch
