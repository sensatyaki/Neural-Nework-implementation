
# coding: utf-8

# In[ ]:




# In[93]:

import math
import random
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



random.seed(0)


def rand(a, b):
    return (b-a)*random.random() + a


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y**2

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
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
                #self.wi[i][j] = rand(-20.0, 20.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
                #self.wo[j][k] = rand(-20.0, 20.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

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


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                #self.wo[j][k] = self.wo[j][k] + N*change
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                #self.wi[i][j] = self.wi[i][j] + N*change
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        predictedResult = []
        for i in range(len(patterns)):
            val = self.update(patterns[i])
            if(val[0] < 0.35):
            	predictedResult.append(0)
            else:
            	predictedResult.append(1)
            
        return predictedResult

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.001, M=0.01):
        # N: learning rate
        # M: momentum factor
        print("-------------Training Neural Network---------------")
        prevError = 0.0
        for i in range(iterations):
            error = 0.0
            #print(self.wi)
            #print(self.wo)
            count = 0
            for p in patterns[0:20000]:
                inputs = p[0]
                targets = p[1]
                #count += 1
                #print(count)
                self.update(inputs)
                #print(self.ai)
                #print(self.ah)
                #print(self.ao)

                error = error + self.backPropagate(targets, N, M)
            #if i % 100 == 0:
            print(i,'error %-.5f' % error)
            if(abs(prevError - error) < 0.001):
                break
            prevError = error

def saveInFile(ids,predicted,fileName):
    #print(len(predicted))
    #print(len(ids))
    result=np.column_stack((ids,predicted))
    
    np.savetxt(fileName,result, delimiter=',',fmt="%d,%d",header="id,salary",comments ='')


def demo():
    # Teach network XOR function
    """
    pat = [
        [[0,0,1], [0]],
        [[0,-1,1], [1]],
        [[-1,0,1], [1]],
        [[-1,-1,1], [0]]
    ]
    """
    pat,testData,ids = getData()
    
    n = NN(59, 30, 1)
    

    n.train(pat)
    

    d  = dict(wi=n.wi, wo=n.wo)

    df = pd.DataFrame.from_dict(d, orient='index').transpose().fillna('')
    df.to_csv('weights.txt', index=False, header=["wi","wo"])
    print("File created weights.txt.")
    

    




    #test it
    #predicted = n.test(testData)
    #saveInFile(ids,predicted,"NNPredicted.csv")


def Normalize(inputData):
    return (inputData - inputData.min()) / (inputData.max() - inputData.min())
    #return (inputData - inputData.mean()) / inputData.std()

    

    








def getData():
    data = pd.read_csv('train.csv')

    numericalColumns = ('age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week')

    for i in numericalColumns:
        data[i] = Normalize(data[i])
    
    data=data.drop('race',1)
    data=data.drop('native-country',1)
    data=pd.get_dummies(data)

    trainY=data['salary']
    trainData=data.drop('salary',1)
    trainData=trainData.drop('workclass_ ?',1)
    
    trainData=trainData.drop('occupation_ ?',1)

    trainData=trainData.drop('id',1)
    features = list(trainData)
    trainData = trainData.values.tolist()
    trainY = trainY.values.tolist()
    trainTarget = []
    for i in trainY:
        temp = []
        temp.append(i)
        trainTarget.append(temp)
    
    






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
    
    
    
    pattern = []
    for i in range(len(trainData)):
        temp = []
        temp.append(trainData[i])
        temp.append(trainTarget[i])
        pattern.append(temp)

    test = test.values.tolist()
    testData = []
    for i in range(len(test)):
        testData.append(test[i])
    #print(testData)
    return pattern,testData,ids








demo()







