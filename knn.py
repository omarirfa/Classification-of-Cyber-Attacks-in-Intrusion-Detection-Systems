import numpy as np
import sys
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


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


    print("begin debugging \n trainData array: ")
    print(trainData.shape)
    print("trainLabels array: ")
    print(trainLabels.shape)
    prediction = dict()
    file = open("/root/Downloads/rf/accuraciesknn.csv","a") 


    knn_model = KNeighborsClassifier(n_jobs=-1,n_neighbors=5)
    knn_model.fit(trainData, trainLabels)
    print("testData array: ")
    print(testData.shape)
    print("testLabels array: ")
    print(testLabels.shape)
    prediction["knn"] = knn_model.predict(testData)
    print(accuracy_score(testLabels,prediction["knn"]))
    acc=accuracy_score(testLabels,prediction["knn"])
    file.write('{}'.format(acc)+ '\n')
    file.close()
    
	# report = classification_report(testLabels, prediction['knn'],labels = [0,1,2,3,5,6,7],target_names = ["Benign","Goldeneye","SSHPatator","FTPPatator","Bot","PortScan","DDos"],output_dict=True)
    # print('\n')    	
    # df = pd.DataFrame(report).transpose()
    # print(df)
    # df.to_csv('/root/Downloads/rf/knntest1.csv',index=True,mode='a')

  

