import math
import random
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def Normalize(inputData):
    return (inputData - inputData.min()) / (inputData.max() - inputData.min())
    #return (inputData - inputData.mean()) / inputData.std()




def saveInFile(ids,predicted,fileName):
    #print(len(predicted))
    #print(len(ids))
    result=np.column_stack((ids,predicted))
    
    np.savetxt(fileName,result, delimiter=',',fmt="%d,%d",header="id,salary",comments ='')   
    



#Fitting other Models
from sklearn.svm import SVC
def svm():
    print("-----------------------Applying SVM--------------------------")
    data,testData,ids = getData()
    X = [x[0] for x in data]
    Y = [x[1][0] for x in data]
    #print(X)
    #print(Y)   
    #print(testData)
    clf = SVC()
    clf.fit(X, Y) 
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    predicted = clf.predict(testData)
    saveInFile(ids,predicted,"predictions_3.csv")
    print("predictions_3.csv created.")


# In[95]:

from sklearn import linear_model
def logisticRegression():
    print("----------------Applying Logistic Regression-------------")
    data,testData,ids = getData()
    X = [x[0] for x in data]
    Y = [x[1][0] for x in data]
    clf = linear_model.SGDClassifier(loss='log',n_jobs= 1)
    clf.fit(X,Y)
    
    predicted = clf.predict(testData)
    saveInFile(ids,predicted,"predictions_1.csv")
    print("predictions_1.csv created.")


# In[96]:

from sklearn.naive_bayes import GaussianNB
def gaussianNaiveBayes():
    print("----------------Applying Gaussian Naive Bayes-------------")
    data,testData,ids = getData()
    X = [x[0] for x in data]
    Y = [x[1][0] for x in data]
    clf = GaussianNB()
    clf.fit(X, Y)
    GaussianNB(priors=None)
    predicted = clf.predict(testData)
    saveInFile(ids,predicted,"predictions_2.csv")
    print("predictions_2.csv created.")



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




logisticRegression()
gaussianNaiveBayes()
svm()