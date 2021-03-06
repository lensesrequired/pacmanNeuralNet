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
  nSize = 9

  def __init__(self, index=0):
    self.index = index
    self.net = pacmanNet.createNetwork(self.nSize, 5, 7)
    self.moves = 0
    self.emptyMoves = 0
    self.iterations = 0

  def reset(self, num):
    #self.initialDots = num
    self.moves = 0
    self.emptyMoves = 0

  def getFF(self, gameState):
    parsedState = gameState.data.parseState()
    ff = pacmanNet.feedforward(self.net, parsedState)

    return ff[1]

  def getAction(self, gameState):
    #training things and stuff
    self.moves += 1

    position = gameState.data.agentStates[0].getPosition()
    col = position[0]
    row = abs(position[1]-(self.HEIGHT-1))
    m = gameState.data.matrix() 

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    inputs = gameState.data.getInput(col, row, m)

    ff = pacmanNet.feedforward(self.net, inputs)
    #print(ff[1])
    expected = [ff[1][i] for i in range(5)]
    viableMoves = [(ff[1][i], i) for i in legalMoves if i != Directions.STOP]
    bestMove = max(viableMoves)
    best = bestMove[1]

    #Get expected values
    #row - 1, col, Directions.NORTH
    if(m[row - 1][col] == "." or m[row - 1][col] == "o"):
      expected[0] = 1
    elif(m[row - 1][col] != "%"):
      expected[0] = 0
    #row + 1, col, Directions.SOUTH
    if(m[row + 1][col] == "." or m[row + 1][col] == "o"):
      expected[1] = 1
    elif(m[row + 1][col] != "%"):
      expected[1] = 0
    #row, col - 1, Directions.WEST
    if(m[row][col - 1] == "." or m[row][col - 1] == "o"):
      expected[3] = 1
    elif(m[row][col - 1] != "%"):
      expected[3] = 0
    #row, col + 1, Directions.EAST
    if(m[row][col + 1] == "." or m[row][col + 1] == "o"):
      expected[2] = 1
    elif(m[row][col + 1] != "%"):
      expected[2] = 0
    expected[Directions.STOP] = 0

    if(self.training):
      if(inputs[4]):
        expected = []
        for i in inputs[:4]:
          if i == 1:
            expected.append(1)
          else:
            expected.append(0)
        pacmanNet.backprop(self.net, inputs, expected + [0], .1)
      else:
        expected = [i for i in inputs[5:]]
        pacmanNet.backprop(self.net, inputs, expected+[0], .1)
      # print "Given:", ff[1]
      # print "Expected", expected
      # input()

      if gameState.data.score < gameState.data.scoreMax - 50:
        #print(ff[1])
        gameState.data.scoreMax = gameState.data.score
        self.emptyMoves += 1
        if self.emptyMoves == 10 or self.emptyMoves > 20: 
          best = viableMoves[random.randint(0, len(viableMoves)-1)][1]
          print "Given:", ff[1]
          print "Expected", expected
          raw_input()
      else:
        self.emptyMoves = 0

    return best



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
