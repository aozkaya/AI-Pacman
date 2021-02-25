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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        eval = 0
        foodList = newFood.asList()
        oldGhostStates = currentGameState.getGhostStates()
        oldFoodList = currentGameState.getFood().asList()
        if len(oldFoodList) > len(foodList):
            eval += 1000
        if len(foodList) > 0:
            least = util.manhattanDistance(newPos, foodList[0])
        for i in foodList:
            temp = util.manhattanDistance(newPos, i)
            eval -= temp
            if least > temp:
                least = temp
        if len(foodList) > 0:
            eval -= least * (len(foodList) + 1)
        for j in newGhostStates:
            if j.scaredTimer > 0:
                eval -= util.manhattanDistance(newPos, j.getPosition())
            else:
                temp = util.manhattanDistance(newPos, j.getPosition())
                if temp == 0:
                    eval = -10000
                else:
                    eval += temp
        for j in oldGhostStates:
            if j.scaredTimer == 0 and util.manhattanDistance(newPos, j.getPosition()) == 0:
                eval = -10000
        if action == Directions.STOP:
            eval -= 500
        return eval

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
        """
        "*** YOUR CODE HERE ***"
        def maxi(gameState, depth):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            scores = []
            for i in gameState.getLegalActions():
                newState = gameState.generateSuccessor(0, i)
                scores.append(mini(newState, depth, gameState.getNumAgents() - 1))
            return max(scores)

        def mini(gameState, depth, counter):
            ghosts = gameState.getNumAgents() - 1
            ghost = ghosts - counter + 1
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            scores = []
            if counter == 1:
                for i in gameState.getLegalActions(ghost):
                    newState = gameState.generateSuccessor(ghost, i)
                    scores.append(maxi(newState, depth - 1))
            else:
                for i in gameState.getLegalActions(ghost):
                    newState = gameState.generateSuccessor(ghost, i)
                    scores.append(mini(newState, depth, counter - 1))
            return min(scores)

        scores = []
        for i in gameState.getLegalActions():
            newState = gameState.generateSuccessor(0, i)
            scores.append(mini(newState, self.depth, newState.getNumAgents() - 1))
        action = gameState.getLegalActions()[scores.index(max(scores))]
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxi(gameState, depth, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            scores = []
            for i in gameState.getLegalActions():
                newState = gameState.generateSuccessor(0, i)
                score = mini(newState, depth, gameState.getNumAgents() - 1, alpha, beta)
                scores.append(score)
                if score > beta:
                    return score
                alpha = max(alpha, score)
            return max(scores)

        def mini(gameState, depth, counter, alpha, beta):
            ghosts = gameState.getNumAgents() - 1
            ghost = ghosts - counter + 1
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            scores = []
            if counter == 1:
                for i in gameState.getLegalActions(ghost):
                    newState = gameState.generateSuccessor(ghost, i)
                    score = maxi(newState, depth - 1, alpha, beta)
                    scores.append(score)
                    if score < alpha:
                        return score
                    beta = min(beta, score)
            else:
                for i in gameState.getLegalActions(ghost):
                    newState = gameState.generateSuccessor(ghost, i)
                    score = mini(newState, depth, counter - 1, alpha, beta)
                    scores.append(score)
                    if score < alpha:
                        return score
                    beta = min(beta, score)
            return min(scores)

        scores = []
        alpha = -float("inf")
        beta = float("inf")
        for i in gameState.getLegalActions():
            newState = gameState.generateSuccessor(0, i)
            score = mini(newState, self.depth, newState.getNumAgents() - 1, alpha, beta)
            scores.append(score)
            if score > beta:
                return score
            alpha = max(alpha, score)
        action = gameState.getLegalActions()[scores.index(max(scores))]
        return action

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
        def maxi(gameState, depth):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            scores = []
            for i in gameState.getLegalActions():
                newState = gameState.generateSuccessor(0, i)
                scores.append(mini(newState, depth, gameState.getNumAgents() - 1))
            return max(scores)

        def mini(gameState, depth, counter):
            ghosts = gameState.getNumAgents() - 1
            ghost = ghosts - counter + 1
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            scores = []
            if counter == 1:
                for i in gameState.getLegalActions(ghost):
                    newState = gameState.generateSuccessor(ghost, i)
                    scores.append(maxi(newState, depth - 1))
            else:
                for i in gameState.getLegalActions(ghost):
                    newState = gameState.generateSuccessor(ghost, i)
                    scores.append(mini(newState, depth, counter - 1))
            return sum(scores)/len(scores)

        scores = []
        for i in gameState.getLegalActions():
            newState = gameState.generateSuccessor(0, i)
            scores.append(mini(newState, self.depth, newState.getNumAgents() - 1))
        action = gameState.getLegalActions()[scores.index(max(scores))]
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Pacman likes round objects, bigger the better, and hates being close proximity to ghosts.
    """
    "*** YOUR CODE HERE ***"

    eval = currentGameState.getScore()
    foodList = currentGameState.getFood().asList()
    capList = currentGameState.getCapsules()
    oldGhostStates = currentGameState.getGhostStates()
    position = currentGameState.getPacmanPosition()
    if len(foodList) > 0:
        least = util.manhattanDistance(position, foodList[0])
        eval -= len(foodList) * 200
    if len(capList) > 0:
        eval -= len(capList) * 200
    for i in foodList:
        temp = util.manhattanDistance(position, i) * 10
        eval -= temp
        if least > temp:
            least = temp
    for i in capList:
        temp = util.manhattanDistance(position, i) * 20
        eval -= temp
    if len(foodList) > 0:
        eval -= least * 20
    for j in oldGhostStates:
        if j.scaredTimer > 0:
            eval -= util.manhattanDistance(position, j.getPosition()) * 100
        else:
            temp = util.manhattanDistance(position, j.getPosition())
            if temp < 2:
                eval -= 10000
    return eval



# Abbreviation
better = betterEvaluationFunction

