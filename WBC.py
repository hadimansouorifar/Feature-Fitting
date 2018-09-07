import numpy as np
import numpy
from FFitting import *
from sklearn.preprocessing import Imputer
import csv
cancer_data = np.genfromtxt(
 fname ='breast-cancer-wisconsin.data', delimiter= ',', dtype= float)

cancer_data = np.delete(arr = cancer_data, obj= 0, axis = 1)
x = cancer_data[:,range(0,9)]
w = cancer_data[:,9]
w = numpy.array(w).astype('int')
w = numpy.array(w).astype('str')

imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

classname=[2,4]
dim = x.shape
a=0
b=70
cm=0
result=[]
for cross in range(1,11):
    #print(cross)
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

    predicted = binary_classification(trainx, trainy, testx, testy, 5, -2, 3)
    c = 0
    fp=0
    tn=0
    tp=0
    fn=0

    #print(len(predicted))

    for i in range(0, len(testy)):

        if (predicted[i] == testy[i]):
            c = c + 1
            if (testy[i]=='2'):
                tp=tp+1
            else:
                tn=tn+1
        elif (testy[i]=='4'):
            fp=fp+1
        elif (testy[i]=='2'):
            fn=fn+1

    #print(c / len(testy))
    result.append(c / len(testy))

    #print(len(testy))
    #print(c)

    cm=cm+(c/len(testy))

    a=a+70
    b=b+70

    if (cross == 10):
        b = len(x)


tpr=tp/(tp+fn)
tnr=tn/(tn+fp)
print('Accuracy : ' + str(cm/10) +'--------TPR : ' + str(tpr) + ' --------- TNR : ' + str(tnr))
print('cross validation results :')
print(result)

