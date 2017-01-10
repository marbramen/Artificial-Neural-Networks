from __future__ import print_function

import numpy as np
import tflearn
from collections import defaultdict
from tflearn.data_utils import load_csv

##Labels 
##unacc, acc, good, vgood 
##Attributes: 
##buying: vhigh, high, med, low. 
##maint: vhigh, high, med, low. 
##doors: 2, 3, 4, 5more. 
##persons: 2, 4, more. 
##lug_boot: small, med, big. 
##safety: low, med, high. 

mapLabels = defaultdict()
mapBuying = defaultdict(int)
mapMaint= defaultdict(int)
mapDoors = defaultdict(int)
mapPersons = defaultdict(int)
mapLugBoot = defaultdict(int)
mapSafety = defaultdict(int)

mapLabels['unacc'] = [1,0,0,0]
mapLabels['acc'] = [0,1,0,0]
mapLabels['good'] = [0,0,1,0]
mapLabels['vgood'] = [0,0,0,1]

mapBuying['vhigh'] = 0
mapBuying['high'] = 1
mapBuying['med'] = 2
mapBuying['low'] = 3

mapMaint['vhigh'] = 0
mapMaint['high'] = 1
mapMaint['med'] = 2
mapMaint['low'] = 3

mapDoors['2'] = 0
mapDoors['3'] = 1
mapDoors['4'] = 2
mapDoors['5more'] = 3

mapPersons['2'] = 0
mapPersons['4'] = 1
mapPersons['more'] = 2
mapPersons['5more'] = 3

mapLugBoot['small'] = 0
mapLugBoot['med'] = 1
mapLugBoot['big'] = 2

mapSafety['low'] = 0
mapSafety['med'] = 1
mapSafety['high'] = 2

numAttr = 4

# Preprocessing function
def preprocessLabel(labels):
    data = [[None]*numAttr]*len(labels)    
    for i in range(len(data)):
        data[i] = mapLabels[labels[i]]
    return np.array(data, dtype=np.float32)    
    
def preprocessData(data):
    for i in range(len(data)):
        data[i][0] = mapBuying[data[i][0]]
        data[i][1] = mapMaint[data[i][1]]
        data[i][2] = mapDoors[data[i][2]]
        data[i][3] = mapPersons[data[i][3]]
        data[i][4] = mapLugBoot[data[i][4]]
        data[i][5] = mapSafety[data[i][5]]
    return np.array(data, dtype=np.float32)

# Load CSV file
data,labels = load_csv('car.csv', target_column=6, has_header=False, categorical_labels=False, n_classes=4)

# Preprocess data
data = preprocessData(data)
labels = preprocessLabel(labels)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 150, activation='sigmoid')
net = tflearn.fully_connected(net, 60, activation='sigmoid')
net = tflearn.fully_connected(net, 4, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=32, batch_size=10, show_metric=True)

## Data test set
test = [[None]*6]*4
test[0] = ['low','low','4','2','big','high']
test[1] = ['low','low','5more','4','med','med']
test[2] = ['low','low','5more','more','big','low']
test[3] = ['high','vhigh','3','4','big','med']
testTemp = [[None]*6]*4
testTemp[0] = ['low','low','4','2','big','high']
testTemp[1] = ['low','low','5more','4','med','med']
testTemp[2] = ['low','low','5more','more','big','low']
testTemp[3] = ['high','vhigh','3','4','big','med']
## Preprocess test set
testTemp = preprocessData(testTemp)
## Predict test
pred = model.predict(testTemp)
print("valores de las predicciones ")

for i in range(len(test)):
    print(test[i])    
    print("unacc " + str(pred[i][0]))
    print("acc " + str(pred[i][1]))
    print("good " + str(pred[i][2]))
    print("vgood " + str(pred[i][3]))
    print("")

