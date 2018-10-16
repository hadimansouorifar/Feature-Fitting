# How to use Feature-Fitting function?
# import it as follows:


from LP import *

# you can call the classifier like this:

predicted = lp(trainx,trainy,testx,testy,p1,p2,dlb,dub,ite,threshold)

where predicted is list of labels of test data returned from function
 based on them you can judge the accuracy in K-fold cross validation
