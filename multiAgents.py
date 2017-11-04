from util import manhattanDistance
from game import Directions
import random, util, pacmanNet

from game import Agent

class NeuralAgent(Agent):
  """
    A NeuralAgent  attempts to use a neural network to figure out the moves.
  """
  WIDTH = 20
  HEIGHT = 7

  def __init__(self, index=0):
    self.index = index
    self.nSize = (self.WIDTH*self.HEIGHT) + 4
    self.net = pacmanNet.createNetwork(self.nSize, 5, 25)
    self.visited = [[0 for j in range(self.WIDTH-2)] for i in range(self.HEIGHT-2)] 
    self.expected = [1,1,1,1,0]

  def resetVisited(self):
    self.visited = [[0 for j in range(self.WIDTH-2)] for i in range(self.HEIGHT-2)] 
    
  def getAction(self, gameState):
    position = gameState.data.agentStates[0].getPosition()
    col = position[0]-1
    row = abs(position[1]-(self.HEIGHT-2))
    self.visited[row][col] += 1
    #print self.visited

    visitedSpaces = []
    #update expected
    self.expected = [0 for i in range(5)]
    if(row != 0):
      visitedSpaces.append(self.visited[row - 1][col])
      if(self.visited[row - 1][col] == 0):
        self.expected[Directions.NORTH] = 1
      else:
        self.expected[Directions.NORTH] = -1*(self.visited[row - 1][col])
    else:
      visitedSpaces.append(0)

    if(row != self.HEIGHT-3):
      visitedSpaces.append(self.visited[row + 1][col])
      if(self.visited[row + 1][col] == 0):
        self.expected[Directions.SOUTH] = 1
      else:
        self.expected[Directions.SOUTH] = -1*(self.visited[row + 1][col])
    else:
      visitedSpaces.append(0)

    if(col != 0):
      visitedSpaces.append(self.visited[row][col - 1])
      if(self.visited[row][col - 1] == 0):
        self.expected[Directions.WEST] = 1
      else:
        self.expected[Directions.WEST] = -1*(self.visited[row][col - 1])
    else:
      visitedSpaces.append(0)

    if(col != self.WIDTH-3):
      visitedSpaces.append(self.visited[row][col + 1])
      if(self.visited[row][col + 1] == 0):
        self.expected[Directions.EAST] = 1
      else:
        self.expected[Directions.EAST] = -1*(self.visited[row][col + 1])
    else:
      visitedSpaces.append(0)

    self.expected[Directions.STOP] = -1*self.visited[row][col]

    #print(self.expected)
    
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    #print(legalMoves)
    
    parsedState = gameState.data.parseState()
    for i in visitedSpaces:
      parsedState.append(i)
    #print parsedState

    ff = pacmanNet.feedforward(self.net, parsedState)
    #print ff[1]

    viableMoves = [(ff[1][i], i) for i in legalMoves]
    #print(viableMoves)

    bestMove = max(viableMoves)
    #print(bestMove)

    #raw_input()

    "Add more of your code here if you want to"

    #backprop
    pacmanNet.backprop(self.net, parsedState, self.expected, 10)
    
#    return legalMoves[chosenIndex]
    return bestMove[1]



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change it
    in any way you see fit.
  """

  def __init__(self, index=0):
    self.index = index
    self.net = pacmanNet.createNetwork()
  
    
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.
    
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """

    print("Getting action")
    
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    successors = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

    print("Get list")
    print(gameState.data.parseState())
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, succ) for succ in successors]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    
    "Add more of your code here if you want to"
    
    return legalMoves[chosenIndex]
  
  def evaluationFunction(self, currentGameState, successorGameState):
    """
    Design a better evaluation function here. 
    
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = successorGameState.getPacmanState().getPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates() 
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    "*** YOUR CODE HERE ***"
    return successorGameState.getScore()

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
    This abstract class** provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    
    **An abstract class is one that is not meant to be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.  
  """
  
  def __init__(self, evalFn = scoreEvaluationFunction):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = evalFn
    
  def setDepth(self, depth):
    """
      This is a hook for feeding in command line argument -d or --depth
    """
    self.depth = depth # The number of search plies to explore before evaluating
    
  def useBetterEvaluation(self):
    """
      This is a hook for feeding in command line argument -b or --betterEvaluation
    """
    self.evaluationFunction = betterEvaluationFunction
    

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
    
  def getAction(self, gameState, network):
    """
      Returns the minimax action from the current gameState using self.depth 
      and self.evaluationFunction.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
    
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

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
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    
    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

DISTANCE_CALCULATORS = {}
