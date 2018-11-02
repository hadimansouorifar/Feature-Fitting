import numpy as np
import numpy
from LP import *
from sklearn.preprocessing import Imputer
import csv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
cancer_data = np.genfromtxt(
 fname ='breast-cancer-wisconsin.data', delimiter= ',', dtype= float)

cancer_data = np.delete(arr = cancer_data, obj= 0, axis = 1)
x = cancer_data[:,range(0,9)]
w = cancer_data[:,9]
w = numpy.array(w).astype('int')
w = numpy.array(w).astype('str')
y=w
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

precision=[]
recall=[]
sensitivity=[]
specificity=[]
accuracy=[]
f1=[]
kf = KFold(n_splits=10)
for train, test in kf.split(x):
    trainx = x[train, :]
    testx = x[test, :]
    trainy = y[train]
    testy = y[test]

    y_pred = lp(trainx, trainy, testx, testy, -2.2, 3, 4, 4, 2, 0.5)
    c = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(testy)):

        if (y_pred[i] == testy[i]):
            c = c + 1
            if (testy[i] == '4'):
                tp = tp + 1
            else:
                tn = tn + 1
        elif (testy[i] == '2'):
            fp = fp + 1
        elif (testy[i] == '4'):
            fn = fn + 1

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    prec = tp / (tp + fp)
    reca = tp / (tp + fn)
    fone=(2*(prec*reca))/(prec+reca)
    precision.append(prec)
    recall.append(reca)
    f1.append(fone)
    accuracy.append(accuracy_score(testy, y_pred))
    sensitivity.append(tpr)
    specificity.append(tnr)


print('accuracy : ' , numpy.mean(accuracy) )
print( 'precision : ' ,numpy.mean(precision) )
print( 'recall : ' , numpy.mean(recall) )
print( 'f1 : ' , numpy.mean(f1) )
print( 'sensitivity : ' , numpy.mean(sensitivity) )
print( 'specificity : ' , numpy.mean(specificity) )



