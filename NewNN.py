import math
import random
import string
import copy
import numpy as np

def sigmoid(x):
	return ( np.exp(x) / (1.0 + np.exp(x)) )	
def desigmoid(x):
	return x*(1 - x)

sigmoid = np.vectorize(sigmoid)
desigmoid = np.vectorize(desigmoid)


def forwardPropagation(NN,row):
	inputs = row
	for k in range(len(NN.weights)):
		newInput = []
		weight = NN.weights[k]
		print(weight.shape)
		
		val = np.dot(weight,inputs)
		outputVal = list(sigmoid(val))
		print(outputVal)

		if(k < (len(NN.weights) - 1)):

			outputVal.append(1)
		NN.activationMatrix[k] = copy.deepcopy(outputVal)
		inputs = outputVal

	return outputVal


def backpropagation(NN,target):
	if(NN.structure[-1] != len(target)):
		raise ValueError('wrong number of target values')

	

	outputDeltas = [0.0]*NN.structure[-1]
	output = NN.activationMatrix[-1]
	print(output)
	error = np.subtract(target,output)
	outputDeltas = desigmoid(output)*error
	prevOutput = NN.activationMatrix[-2]
	updateValues = np.dot(outputDeltas,prevOutput)
	NN.weights[-1] -= updateValues

	print(NN.weights)


class NN(object):
	"""docstring for NN"""
	def __init__(self,structure):

		self.structure = structure
		
		
		self.activationMatrix = []
		for layer in structure[1:]:
			activateLayer = [1.0]*layer
			self.activationMatrix.append(activateLayer)

		self.weights = []

		for i in range(len(structure) - 1):
			
			weightMatrix = np.random.rand(structure[i+1],structure[i] + 1)  # 1 extra weight for bias in both layers
			 
			self.weights.append(weightMatrix)





network = NN([2,1,2])


print(network.activationMatrix)

val = forwardPropagation(network,[2,3,1])



print(network.activationMatrix)
print(network.weights)
backpropagation(network,[2,3])