# How to use Feature-Fitting function?
import it as follows
from FFitting import *
you can call the classifier like this:

predicted = binary_classification(trainx, trainy, testx, testy, degree, value1, value2)

where predicted is list of labels of test data returned from function
based on them you can judge the accuracy in K-fold cross validation
