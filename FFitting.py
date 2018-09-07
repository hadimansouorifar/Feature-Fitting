import  numpy
def binary_classification(trainx,trainy,testx,testy,degree,value1,value2):
    trainx = numpy.asarray(trainx)
    dim = trainx.shape
    y = []



    ynames=set(trainy)
    yn=list(ynames)



    yn3=sorted(yn)





    predicted = []

    v = []

    cm = 0

    for i in range(0, len(trainx)):
        if (trainy[i] == (yn3[0])):

            v.append(value1)
        else:
            v.append(value2)

    vec = numpy.zeros((dim[1], degree + 1), dtype=float)

    for j in range(0, dim[1]):
        temp = []
        for i in range(0, len(trainx)):
            temp.append(trainx[i][j])
        vec[j] = numpy.polyfit(temp, v, degree)

    s = 0
    c = 0

    for i in range(0, len(testy)):
        maxx1 = 0
        minn1 = 1000000

        minn2 = 1000000
        sum = 0

        for j in range(0, dim[1]):
            p = numpy.poly1d(vec[j])

            a = (p(testx[i][j]))
            sum = sum + a

            a1 = abs(p(testx[i][j]) - value1)
            a2 = abs(p(testx[i][j]) - value2)

            if (a1 < minn1):
                minn1 = a1
                or1 = j
            if (a2 < minn2):
                minn2 = a2
                or2 = j



                # print(str(sum) + '--' + str(testy[i]))


        if (sum > 0):
            predicted.append(str(yn3[1]))
        else:
            predicted.append(str(yn3[0]))

    return predicted
