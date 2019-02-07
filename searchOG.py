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

'''
    IMPORTANT NOTE:
    Derrick Neal and Brian Veber worked together on this project as was allowed by the pair programming rules. Our submissions will look very similar. We worked hand in hand in the creation of the search algorithms, and when writing the heuristics over the span of January 7th 2019 - January 24th 2019.
'''



"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import copy
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    fringe = util.Stack() #using a stack to be able to pop the last node and be able to process it in the algo
    path = util.Stack() # this will record the path that need to be taken to get to each node
    explored_nodes = []
    curr_state = problem.getStartState()
    fringe.push((curr_state, copy.deepcopy(path)))

    
    while True: # this can just go until the break is processed which means it is at the goalstate, there should always be a goalstate so ifeel comfortable with having a while True
        if problem.isGoalState(curr_state): #if its goal node end the loop
           break

        child_nodes = problem.getSuccessors(curr_state) #get the child nodes
        for child in child_nodes: #process the child nodes
            if child[0] not in explored_nodes: #this will go through the
                path.push(child[1]) # add the child's Direction value so that we know how to get to it.
                fringe.push((child[0], copy.deepcopy(path))) # have to deep copy so that each item in the fringe has its own indeoendent
                path.pop() #pop it off to reset it for the next child

        explored_nodes.append(curr_state) #save that you explored that node if you havent before

        while curr_state in explored_nodes:
            #this will allow for backtracking which is something we need
            curr_state_path = fringe.pop()
            curr_state = curr_state_path[0]
            path = curr_state_path[1]

    print(path.list)# for the autograder
    return path.list

def makeActionList(list):
    '''
        makes a parsable list of action for pacman
    '''
    from game import Directions
    action_list = []
    while not list.isEmpty():
        temp = list.pop()
        if temp == "North":
            action_list.append(Directions.NORTH)
        elif temp == "East":
            action_list.append(Directions.EAST)
        elif temp == "South":
            action_list.append(Directions.SOUTH)
        elif temp == "West":
            action_list.append(Directions.WEST)
    action_list.reverse()
    print(action_list)
    return action_list


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    '''
        SEE DFS FOR EXPLANATATIONS. As there isnt a lot of code difference ill explain most stuff in DFS and choice things is each search algo.
    '''
    fringe = util.Queue() #switch to using a queue so that we can have a different pop algorithm so we scan a single depth before moving on
    path = util.Stack()
    explored_nodes = []
    start = problem.getStartState()
    curr_state = start
    fringe.push((curr_state, copy.deepcopy(path)))
    explored_nodes.append(curr_state)
    
    while True:
        if problem.isGoalState(curr_state): #if its goal node end the loop
            break

        child_nodes = problem.getSuccessors(curr_state) #get the child nodes
        for child in child_nodes:
            if child[0] not in explored_nodes:
                path.push(child[1])
                fringe.push((child[0], copy.deepcopy(path)))
                path.pop()

        if curr_state not in explored_nodes:
            explored_nodes.append(curr_state)

        while curr_state in explored_nodes:
            curr_state_path = fringe.pop()
            curr_state = curr_state_path[0]
            path = curr_state_path[1]

    print(path.list)
    return path.list

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    '''
        SEE DFS FOR EXPLANATATIONS. As there isnt a lot of code difference ill explain most stuff in DFS and choice things is each search algo.
    '''
    fringe = util.PriorityQueue() # use a priority queue and make the cost of
    path = util.Stack()
    explored_nodes = []
    start = problem.getStartState()
    curr_state = start
    cost = 0
    fringe.push((curr_state, copy.deepcopy(path), 0), 0)
    explored_nodes.append(curr_state)
    
    while True:
        if problem.isGoalState(curr_state): #if its goal node end the loop
            break

        child_nodes = problem.getSuccessors(curr_state) #get the child nodes
        for child in child_nodes:
            if child[0] not in explored_nodes:
                cost+=child[2]
                path.push(child[1])
                fringe.push((child[0], copy.deepcopy(path), cost), cost)
                cost-=child[2]
                path.pop()
    
        if curr_state not in explored_nodes:
            explored_nodes.append(curr_state)

        while curr_state in explored_nodes:
            curr_state_path = fringe.pop()
            curr_state = curr_state_path[0]
            path = curr_state_path[1]
            cost = curr_state_path[2]
    
    print(path.list) 
    return path.list

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue() #keep using the priority queue but when we enqueue it we have to use the f(n) = g(n) + h(n) as is shown in the lecture slides
    path = util.Stack()
    explored_nodes = []
    start = problem.getStartState()
    curr_state = start
    cost = 0
    explored_nodes.append(curr_state)
    
    while True:
        if problem.isGoalState(curr_state): #if its goal node end the loop
            break

        child_nodes = problem.getSuccessors(curr_state) #get the child nodes
        for child in child_nodes: # can check DFS for explanations of the stuff
            if child[0] not in explored_nodes:
                cost = cost + child[2]
                priority = cost + heuristic(child[0], problem) #f(n) = g(n) + h(n)
                path.push(child[1])
                fringe.push((child[0], copy.deepcopy(path), cost), priority)
                cost = cost - child[2]
                priority = 0 #reset priority
                path.pop()
    
        if curr_state not in explored_nodes:
            explored_nodes.append(curr_state)

        while curr_state in explored_nodes:
            curr_state_path = fringe.pop()
            curr_state = curr_state_path[0]
            path = curr_state_path[1]
            cost = curr_state_path[2]
    
    print(path.list)
    return path.list


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
