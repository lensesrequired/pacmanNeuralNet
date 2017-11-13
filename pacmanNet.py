
import math
import random
import pacman

#inspired by Joel Grus:  Data Science from Scratch

def dot(x, w):
    if len(x) == len(w):
        d = sum([x[i]*w[i] for i in range(len(x))])
    else:
        raise ValueError("Mismatched Vector sizes")
    return d

def neuronOutput (inputs, weights):
    try:
        d = dot(inputs,weights)
    except ValueError:
        print("Value Error")
        print(len(inputs))
        print(len(weights))
    # note - last col of weights is bias.
    try:
        trans = 1/(1+math.exp(-d))
    except OverflowError:
        print(d)
        #print(inputs)
        #print(weights)
        #input()
        return 0
    return trans

def feedforward(network, inputVector):
    #network is a list of layers
    #layer is list of nodes
    #node is a list of weights

    outputs = []

    for layer in network:
        biasedInput = inputVector + [-1]
        output = [neuronOutput(biasedInput, neuron) for neuron in layer]
        outputs.append(output)
        inputVector = output
    return outputs

def backprop(network, inputVector, targets, lr):
    hidden, outputs= feedforward(network, inputVector)

  # get error for output layer    
    outputErrors = []
    for i in range(len(outputs)):
        #print(outputs[i], targets[i])
        outputErrors.append(outputs[i] * (1-outputs[i]) *(targets[i]-outputs[i]) )

    # outputErrors = [output * (1-output)*(target-output)
    #                  for output,target in zip(outputs, targets)]

    #get hidden error
    hiddenErrors = [hiddenOut * (1-hiddenOut) *
                    dot(outputErrors, [n[i] for n in network[-1]])
                    for i, hiddenOut in enumerate(hidden)]
    #print(hiddenErrors)

    #adjust output weights
    for i, outputNeuron in enumerate(network[-1]):
        for j, hiddenOutput in enumerate(hidden +[-1]):
            #print(outputErrors[i],hiddenOutput)
            #print(outputErrors[i]*hiddenOutput*lr)
            outputNeuron[j] += outputErrors[i]*hiddenOutput*lr

    #adjust hidden weights
    for i, hiddenNeuron in enumerate(network[0]):
        for j, inputval in enumerate(inputVector + [-1]):         
            hiddenNeuron[j] += hiddenErrors[i]*inputval*lr

def learn(network, trainSet, epoch):
    for i in range(epoch):
        for tInput, tOutput in trainSet:      
            backprop(network,tInput,tOutput, .5)           

def createNode(size):
    node = []
    for i in range(size):
        node.append(random.random()/float(size))
        #node.append(0.5/float(size))
    #add one more for bias
    node.append(random.random())

    return node

def createNetwork(input_size = 140, output_size = 4, hidlay_size = 30):
    hiddenLayer = []
    for i in range(hidlay_size):
        hiddenLayer.append(createNode(input_size))

    outputLayer = []    
    for i in range(output_size):
        outputLayer.append(createNode(hidlay_size))
                           
    return [hiddenLayer, outputLayer]

def predict(network, inputval):
    hidden, result = feedforward(network,inputval)
    print("Network Output: ")
    for i in range(len(result)):    
        print("{:.2f}".format(result[i]))
