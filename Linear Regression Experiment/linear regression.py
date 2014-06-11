#Using linear regression for binary classification, then 
#to seed a perceptron.


from random import uniform, randint
import numpy as np
from matplotlib import pylab as pl
from sklearn.linear_model import LinearRegression


#class for target function, e.g. creating a line in range
class targetf():
    x_1 = uniform(-1.0, 1.0)
    y_1 = uniform(-1.0, 1.0)
    x_2 = uniform(-1.0, 1.0)
    y_2 = uniform(-1.0, 1.0)

    slope = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - (slope * x_1)

def makeLine(slope, b, style):
    xs = [-2, 2]
    ys = []
    for x in xs:
        ys.append(slope * x + b)
    pl.plot(xs, ys, style)

#function to generate N random data points and evaluate against a target function
def generate_data(input, output, N, target):
    for i in range(0, N):
        x_coor = uniform(-1.0, 1.0)
        y_coor = uniform(-1.0, 1.0)
        input.append([x_coor, y_coor])

        if y_coor < (target.slope * x_coor) + target.b:
            output.append(-1)
        else:
            output.append(1)


def sign_func(w0, w1, w2, data_inputs):
    results = []
    for point in data_inputs:
        if w0 + (w1 * point[0]) + (w2 * point[1]) < 0:
            results.append(-1)
        else:
            results.append(1)
    return results

def getError(results, comparer, samples, input=None):
    errors = 0
    misses = []
    for j in range(0, len(comparer) - 1):
        if results[j] != comparer[j]:
            errors += 1
            if input:
                misses.append([input[j], comparer[j]])
    error = errors/samples
    return {'error': error, 'misses': misses}


def run(N):
    #generate and graph target function
    target = targetf()

    #generate data
    X = []
    Y = []

    generate_data(X, Y, N, target)

    #linear regression
    lin = LinearRegression()
    lin.fit(X, Y)
    lin.predict(X)

    #Weights - w0 is intercept, w1 and w2 are the coefs
    #use W to formulate g, our chosen hypothesis
    coefs = lin.coef_
    w0 = lin.intercept_
    w1 = coefs[0]
    w2 = coefs[1]


    #evalute based on g
    train_results = sign_func(w0, w1, w2, X)
    in_returned = getError(train_results, Y, float(N), X)
    E_in = in_returned['error']

    #generate test data
    X_test = []
    Y_test = []
    generate_data(X_test, Y_test, 1000, target)

    #evaluate g against test data
    test_results = sign_func(w0, w1, w2, X_test)
    out_returned = getError(test_results, Y_test, 1000.0)
    E_out = out_returned['error']


    #uncomment to plot for visualization, beware of reps...

    #target function
    #makeLine(target.slope, target.b, 'r--')

    #sample points
    #for point in X:
    #    pl.scatter(point[0], point[1], color='black')

    #hypothesis g
    #intercept = -w0/w2
    #lslope = -(w1/w2)
    #makeLine(lslope, intercept, 'b')

    #graph
    #pl.axis([-1, 1, -1, 1])
    #pl.xlabel('x1 or feature 1')
    #pl.ylabel('x2 or feature 2')
    #pl.title('Linear Regression for Classification Example')
    #pl.show()


    #seed perceptron with initial weights from linear regression
    training = True
    p0 = w0
    p1 = w1
    p2 = w2
    count = 0
    while training:
        per_results = sign_func(p0, p1, p2, X)
        per_returned = getError(per_results, Y, float(N), X)
        misses = per_returned['misses']

        if len(misses) == 0:
            training = False
        else:
            #choose 1 misclassified point at random
            n = randint(0, len(misses) - 1)
            rand_miss = misses[n]
            miss_xs = rand_miss[0]
            miss_y = rand_miss[1]


            p0 += miss_y
            p1 += float(miss_y) * miss_xs[0]
            p2 += float(miss_y) * miss_xs[1]

            #update count
            count += 1.0

    return {'E_in': E_in, 'E_out': E_out, 'iterations': count}


E_in_total = 0.0
E_out_total = 0.0
iterations = 0
reps = 1000
for k in range(0, reps):
    returned = run(10)
    E_in_total += returned['E_in']
    E_out_total += returned['E_out']
    iterations += returned['iterations']

E_in_avg = E_in_total/reps
E_out_avg = E_out_total/reps
iterations_avg = iterations/reps
print 'The average in-sample error is: %.3f' % E_in_avg
print 'The average out-of-sample error is: %.3f' % E_out_avg
print 'The average number of PLA iterations is: %d' % iterations_avg

