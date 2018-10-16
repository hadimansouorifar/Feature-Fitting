import numpy as np
import numpy
from LP import *
from sklearn.preprocessing import Imputer
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
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


classname=[2,4]

dim = x.shape
a=0
b=30
cm=0
result=[]




precision=[]
recall=[]
sensitivity=[]
specificity=[]
accuracy=[]
f1=[]

y=w
score=[]
kf = KFold(n_splits=10)
for train, test in kf.split(x):
    trainx = x[train, :]
    testx = x[test, :]
    trainy = y[train]
    testy = y[test]

    y_pred = lp(trainx, trainy, testx, testy, -1.3, 2.9, 1, 1, 0, 0.42)

    c = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(testy)):

        if (y_pred[i] == testy[i]):
            c = c + 1
            if (testy[i] == '1'):
                tp = tp + 1
            else:
                tn = tn + 1
        elif (testy[i] == '2'):
            fp = fp + 1
        elif (testy[i] == '1'):
            fn = fn + 1

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    prec = tp / (tp + fp)
    reca = tp / (tp + fn)
    fone = (2 * (prec * reca)) / (prec + reca)
    precision.append(prec)
    recall.append(reca)
    f1.append(fone)
    accuracy.append(accuracy_score(testy, y_pred))
    sensitivity.append(tpr)
    specificity.append(tnr)

print('accuracy : ', numpy.mean(accuracy))
print('precision : ', numpy.mean(precision))
print('recall : ', numpy.mean(recall))
print('f1 : ', numpy.mean(f1))
print('sensitivity : ', numpy.mean(sensitivity))
print('specificity : ', numpy.mean(specificity))



