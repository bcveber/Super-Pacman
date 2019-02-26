# multiAgents.py
# --------------
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

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        for ghost in newGhostStates:
            if newPos == ghost.getPosition():
                return float("-inf")

        shortestDis = 100000000

        for food in currentGameState.getFood().asList():
            curDis = manhattanDistance(food,newPos)
            if (curDis < shortestDis):
                shortestDis = curDis

        return (1/(1+shortestDis))

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            if depth + 1 == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(0)
            v = float("-inf")

            for act in actions:
                 successor = gameState.generateSuccessor(0, act)
                 minimum = min_value(successor, (depth+1), 1)
                 v = max(v, minimum)
                 
            return v

        def min_value(gameState, depth, index):
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(index)
            v = float("inf")

            for act in actions:
                successor = gameState.generateSuccessor(index, act)
                if index == (gameState.getNumAgents() - 1):
                    val = max_value(successor, depth)
                else:
                    val = min_value(successor, depth, index + 1)
                v = min(v, val)
                
            return v

        actions = gameState.getLegalActions(0)
        curScore = float("-inf")

        for act in actions:
            successor = gameState.generateSuccessor(0, act)
            minimum = min_value(successor, 0, 1)

            if curScore < minimum:
                move = act
                curScore = minimum
                
        return move
              

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, alpha, beta):
            if depth + 1 == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(0)
            v = float("-inf")

            for act in actions:
                 successor = gameState.generateSuccessor(0, act)
                 minimum = min_value(successor, alpha, beta, (depth+1), 1)
                 v = max(v, minimum)

                 if v > beta:
                     return v
                 alpha = max(v,alpha)

            return v


        def min_value(gameState, alpha, beta, depth, index):
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(index)
            v = float("inf")

            for act in actions:
                successor = gameState.generateSuccessor(index, act)
                if index == (gameState.getNumAgents() - 1):
                    val = max_value(successor, depth, alpha, beta)
                else:
                    val = min_value(successor, alpha, beta, depth, index + 1)

                v = min(v, val)

                if v < alpha:
                    return v
                beta = min(v,beta)

            return v

        actions = gameState.getLegalActions(0)
        alpha = float("-inf")
        curScore = float("-inf")
        beta = float("inf")
        index = 1

        for act in actions:
            successor = gameState.generateSuccessor(0, act)
            minimum =  min_value(successor, alpha, beta, 0, index)

            if curScore < minimum:
                move = act
                curScore = minimum

            if beta < minimum:
                break

            alpha = max(alpha, minimum)
            
        return move
         

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth):
            if depth + 1 == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(0)
            v = float("-inf")

            for act in actions: 
                 successor = gameState.generateSuccessor(0, act)
                 minimum = exp_value(successor, (depth+1), 1)
                 v = max(v, minimum)
            return v

        def exp_value(gameState, depth, index):
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(index)
            allVal = 0

            for act in actions:
                successor = gameState.generateSuccessor(index, act)
                if index == (gameState.getNumAgents() - 1):
                    val = max_value(successor, depth)
                else:
                    val = exp_value(successor, depth, index + 1)
                allVal += val/len(actions)

            return allVal

        actions = gameState.getLegalActions(0)
        curScore = float("-inf")

        for act in actions:
            successor = gameState.generateSuccessor(0, act)
            exp = exp_value(successor, 0, 1)

            if curScore < exp:
                move = act
                curScore = exp

        return move
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: see comments
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    shortestDis = float("-inf")
    capsule = len(currentGameState.getCapsules())
    position = currentGameState.getGhostPositions()
    score = currentGameState.getScore()
    totDis = 1
    run = 0
    dis1 = 0
    retScore = 0.0
    mh = 0.0
    other = 0.0

    for food in newFood.asList():
        dis = manhattanDistance(food, newPos)
        if dis <= shortestDis:
            shortestDis = dis
        if shortestDis  == -1:
            shortestDis = dis
     
    for pos in position:
        dis1 += manhattanDistance(pos, newPos)
        totDis += dis1
        if dis1 <= 1:
            run += 1

    mh = (1/shortestDis) - (1/totDis)
    retScore = score + mh - capsule - run
    return retScore

# Abbreviation
better = betterEvaluationFunction
