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
    self.nSize = (self.WIDTH*self.HEIGHT)
    self.net = pacmanNet.createNetwork(self.nSize, 6, 100)
    self.visited = [[0 for j in range(self.WIDTH-2)] for i in range(self.HEIGHT-2)] 
    self.expected = [1,1,1,1,0]
    self.numDots = 1
    self.moves = 0

  def reset(self, num):
    self.initialDots = num
    self.moves = 0
    self.visited = [[0 for j in range(self.WIDTH-2)] for i in range(self.HEIGHT-2)] 

  def getFF(self, gameState):
    parsedState = gameState.data.parseState()
    ff = pacmanNet.feedforward(self.net, parsedState)

    return ff[1]

  def getAction(self, gameState):
    #training things and stuff
    self.moves += 1

    position = gameState.data.agentStates[0].getPosition()
    col = position[0]-1
    row = abs(position[1]-(self.HEIGHT-2))
    self.visited[row][col] += 1

    surroundingSpaces = [-1 for i in range(4)]
    m = gameState.data.matrix()
    if(row != 0): #row - 1, col, Directions.NORTH
      if(m[row][col + 1] != "%"):
        surroundingSpaces[0] = (self.visited[row - 1][col])
    if(row != self.HEIGHT-3): #row + 1, col, Directions.SOUTH
      if(m[row + 2][col + 1] != "%"):
        surroundingSpaces[1] = self.visited[row + 1][col]
    if(col != 0): #row, col - 1, Directions.WEST
      if(m[row + 1][col] != "%"):
        surroundingSpaces[3] = self.visited[row][col - 1]
    if(col != self.WIDTH-3): #row, col + 1, Directions.EAST
      if(m[row + 1][col + 2] != "%"):
        surroundingSpaces[2] = self.visited[row][col + 1]

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    parsedState = gameState.data.parseState()
    self.numDots = parsedState.count(0.1)

    ff = pacmanNet.feedforward(self.net, parsedState)
    self.expected = [ff[1][i] for i in range(5)]
    viableMoves = [(ff[1][i], i) for i in legalMoves]
    bestMove = max(viableMoves)

    ranking = list(set(surroundingSpaces))
    ranking.sort()
    try:
      ranking.remove(-1)
    except ValueError:
      pass

    #Get expected values
    for r,rank in enumerate(ranking):
      if r == 0:
        for j in range(len(self.expected)-1):
          if surroundingSpaces[j] == rank:
            self.expected[j] = 1
      elif r == 1:
        for j in range(len(self.expected)-1):
          if surroundingSpaces[j] == rank:
            self.expected[j] = ff[1][j] * 0.5
      elif r == 2:
        for j in range(len(self.expected)-1):
          if surroundingSpaces[j] == rank:
            self.expected[j] = ff[1][j] * -0.5
      else:
        for j in range(len(self.expected)-1):
          if surroundingSpaces[j] == rank:
            self.expected[j] = -1
    self.expected[Directions.STOP] = -1

    if(self.training):
      pacmanNet.backprop(self.net, parsedState, self.expected + [(self.numDots-1)/float(self.initialDots)], 1)

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
