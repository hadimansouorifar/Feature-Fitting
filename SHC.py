import numpy as np
import numpy
from FFitting import *
from sklearn.preprocessing import Imputer
import csv
filename = 'bc_survival.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
td = list(reader)
data = numpy.array(td).astype('str')



x = data[:, 0:3]  # select columns 1 through end

imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

x = numpy.array(x).astype('float')

w = data[:, 3]   # select column 0, the stock priceprint(w)
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

classname=['1','2']
dim = x.shape
a=0
b=30
cm=0
result=[]
for cross in range(1,11):
   # print(cross)
    trainx = []
    trainy = []

    testx = []
    testy = []

    for k in range(0, dim[0]):
        if k not in range(a, b):
            trainx.append(x[k])
            trainy.append(w[k])

        else:
            testx.append(x[k])
            testy.append(w[k])

    predicted = binary_classification(trainx, trainy, testx, testy, 4, -1.3, 2.9)
    c = 0
    fp = 0
    tn = 0
    tp = 0
    fn = 0

    #print(len(predicted))

    for i in range(0, len(testy)):

        if (predicted[i] == testy[i]):
            c = c + 1
            if (testy[i] == '1'):
                tp = tp + 1
            else:
                tn = tn + 1
        elif (testy[i] == '2'):
            fp = fp + 1
        elif (testy[i] == '1'):
            fn = fn + 1

    #print(c / len(testy))
    result.append(c / len(testy))

    #print(len(testy))
    #print(c)

    cm=cm+(c/len(testy))

    a=a+30
    b=b+30

    if (cross == 10):
        b = len(x)

tpr=tp/(tp+fn)
tnr=tn/(tn+fp)
print('Accuracy : ' + str(cm/10) +'--------TPR : ' + str(tpr) + ' --------- TNR : ' + str(tnr))
print('cross validation results :')
print(result)

