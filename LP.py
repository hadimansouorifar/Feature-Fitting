import  numpy
import matplotlib.pyplot as plt
from random import randint
def lp(trainx,trainy,testx,testy,value1,value2,r1,r2,it,threshold):


    x=numpy.array(trainx)
    dim=x.shape
    r=[]
    for k in range(0,dim[1]):
        r.append(randint(r1, r2))

    degree = []
    for k in range(0, dim[1]):
        degree.append(r[k])

    predict = lptest(trainx, trainy, trainx, trainy, r, value1, value2,threshold)




    for u in range(1,it):


        degtemp=degree

        cand=randint(0, dim[1]-1)
        degtemp[cand] = degtemp[cand] + 1


        predicted = lptest(trainx, trainy, trainx, trainy, degtemp, value1, value2,threshold)

        c=0

        for i in range(0, len(trainy)):

            if (predicted[i] == trainy[i]):
                c = c + 1

        maxx=c/len(trainy)

        c = 0
        fp = 0
        tn = 0
        tp = 0
        fn = 0
        cm = 0

        # print(len(predicted))

        for i in range(0, len(trainy)):

            if (predicted[i] == trainy[i]):
                c = c + 1


        # print(c / len(testy))


        # print(len(testy))
        # print(c)

        cm =  (c / len(trainy))
        if ((cm > maxx) ):
            maxx = cm
            degree = degtemp
        else:
            degtemp=degree






    predict=lptest(trainx, trainy, testx, testy, degree, value1, value2,threshold)
    return predict



def lptest(trainx,trainy,testx,testy,degree,value1,value2,threshold):
    trainx = numpy.asarray(trainx)
    dim = trainx.shape
    y = []



    ynames=set(trainy)
    yn=list(ynames)



    yn3=sorted(yn)





    predicted = []

    v = []

    cm = 0
    v = numpy.zeros((len(yn3),len(trainx)), dtype=int)

    #initializing the Vs with value1
    for i in range(0, len(trainx)):
        for j in range(0, len(yn3)):
            v[j][i] = value1

    #Target value2
    for i in range(0, len(trainx)):
        for j in range(0, len(yn3)):
            if (trainy[i] == (yn3[j])):
                v[j][i]=value2






    vec = []

    for k in range(0, len(yn3)):
        vec.append([])
        for j in range(0, dim[1]):

            temp = []
            for i in range(0, len(trainx)):
                temp.append(trainx[i][j])
            vec[k].append(numpy.polyfit(temp, v[k], degree[j]))




    for i in range(0, len(testy)):
        maxx1 = 0
        ind=0
        pr=[]


        for k in range(0, len(yn3)):
            sum=0
            for j in range(0, dim[1]):



                p = numpy.poly1d(vec[k][j])
                a = (p(testx[i][j]))
                sum=sum+a

            tan=(1/(1+numpy.exp(-sum)))
            pr.append(tan)
            if (tan > threshold):

                ind=k


        predicted.append(yn3[ind])








                # print(str(sum) + '--' + str(testy[i]))





    return predicted




