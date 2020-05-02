import numpy as np
import sys
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV


numCV = int(sys.argv[2])
fileName = sys.argv[1]
fullArray = np.genfromtxt(fileName, dtype=np.float64, delimiter=",", skip_header=1)

# data splitting

labels = np.zeros((len(fullArray), 8))
data = np.zeros((len(fullArray), 19))

for row in range(0, len(fullArray)):
    # create one hot vectors for label set
    label = fullArray[row, 19]
    if (label == 1.0):
        # benign
        labels[row, 0] = 1.0
    elif (label == 2):
        # GoldenEye
        labels[row, 1] = 1.0
    elif (label == 7):
        # SSHPatator
        labels[row, 2] = 1.0
    elif (label == 8):
        # FTPPatator
        labels[row, 3] = 1.0
    elif (label == 11):
        # Bot
        labels[row, 5] = 1.0
    elif (label == 12):
        # portscan
        labels[row, 6] = 1.0
    elif (label == 13):
        # DDOS
        labels[row, 7] = 1.0
        # print(labels[row,7])
    else:
        print("Label number outside approved range")

    # fill the data array
    data[row, 0:18] = fullArray[row, 0:18]

for step in range(0, numCV):
    testSize = len(fullArray) / numCV
    trainSize = len(fullArray) - testSize - 1
    testData = np.zeros((testSize, 18))
    testLabels = np.zeros((testSize, 8))

    trainData = np.zeros((trainSize, 18))
    trainLabels = np.zeros((trainSize, 8))

    testData[0:testSize, 0:17] = data[step * testSize:step * testSize + testSize, 0:17]
    testLabels[0:testSize, 0:7] = labels[step * testSize:step * testSize + testSize, 0:7]

    trainData[0:step * testSize, 0:17] = data[0:step * testSize, 0:17]
    trainData[step * testSize + 1:trainSize, 0:17] = data[step * testSize + testSize + 1:len(data) - 1, 0:17]

    trainLabels[0:step * testSize, 0:7] = labels[0:step * testSize, 0:7]
    trainLabels[step * testSize + 1:trainSize, 0:7] = labels[step * testSize + testSize + 1:len(labels) - 1, 0:7]

    
   
    fit_rf = RandomForestClassifier(random_state=42)	
    np.random.seed(42)
    start = time.time()
    param_dist = { 'max_features': ['auto','sqrt','log2', None],'bootstrap': [True, False],'criterion':['gini','entropy']}
    cv_rf = GridSearchCV(fit_rf, cv=10, param_grid=param_dist,n_jobs=3)
    cv_rf.fit(trainData,trainLabels)
    print('Best parameters using grid search: \n', cv_rf.best_params_)
    end=time.time()
    print('Time taken in grid search: {0: .2f}'.format(end - start))
    	

