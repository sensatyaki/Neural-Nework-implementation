
import math
import random
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast



random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2


def Normalize(inputData):
    return (inputData - inputData.min()) / (inputData.max() - inputData.min())

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi,self.wo = getWeights()
        
        

    def update(self, inputs):
        
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]




    def test(self, patterns):
        predictedResult = []
        for i in range(len(patterns)):
            val = self.update(patterns[i])
            if(val[0] < 0.35):
            	predictedResult.append(0)
            else:
            	predictedResult.append(1)
            
        return predictedResult       




def saveInFile(ids,predicted,fileName):
    
    result=np.column_stack((ids,predicted))
    
    np.savetxt(fileName,result, delimiter=',',fmt="%d,%d",header="id,salary",comments ='')



def getWeights():
	data = pd.read_csv('weights.txt')
	wi = []
	wo = []


	for i in range(len(data['wo'])):
		try:
			data['wo'][i] =  ast.literal_eval(data['wo'][i])
			wo.append(data['wo'][i])
				
			
		except Exception as e:
			pass
		
			
				
			
		    

		



	for i in range(len(data['wi'])):
		data['wi'][i] =  ast.literal_eval(data['wi'][i])
		wi.append(data['wi'][i])

		
			

	

	return wi,wo




def getData():
    numericalColumns = ('age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week')

    test=pd.read_csv('kaggle_test_data.csv')
	
    test=test.drop('race',1)

    test=test.drop('native-country',1)
    ids=test['id'].values
    
    for i in numericalColumns:
        test[i] = Normalize(test[i])
    test=pd.get_dummies(test)
    test=test.drop('workclass_ ?',1)
    
    test=test.drop('occupation_ ?',1)
    test=test.drop('id',1)
    #test = test.drop('native-country_ Holand-Netherlands',1)
    
    

    test = test.values.tolist()
    testData = []
    for i in range(len(test)):
        testData.append(test[i])

    return testData,ids


def demo():
	testData,ids = getData()
    
	n = NN(59, 30, 1)

	predicted = n.test(testData)
	saveInFile(ids,predicted,"predictions.csv")
	print("File created successfully predictions.csv.")



demo()