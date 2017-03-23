from random import seed
from random import random
import math
import numpy as np

def initializeNetwork(inputWeights,hiddenNeurons,outputs):
	network = list()
	hiddenLayer = [{'weights':[random() for i in range(inputWeights + 1)]} for i in range(hiddenNeurons)]
	network.append(hiddenLayer)

	outputLayer = [{'weights':[random() for i in range(hiddenNeurons + 1)]} for i in range(outputs)]

	network.append(outputLayer)

	return network

seed(1)
network = initializeNetwork(2, 1, 2)
for layer in network:
	print(layer)



def activate(weights,inputs):
	print(weights)
	print(inputs)
	activationVal = np.dot(inputs,weights)

	return activationVal


def transfer(activation):
	return 1.0 / (1.0 + math.exp(-activation))



def forwardPropagation(network,row):
	inputs = row

	for layer in network:
		newInput = []
		for neuron in layer:
			activationVal = activate(neuron['weights'],inputs)
			neuron['output'] = transfer(activationVal)
			newInput.append(neuron['output'])
		inputs = newInput
		inputs.append(1)
	inputs = inputs[0:-1]
	return inputs




network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]

row = [1, 0, 1]

output = forwardPropagation(network,row)
print(output)
