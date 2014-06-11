__author__ = 'alex'


#Given a non-linear target function, test out standard linear regression, then
#transform the data into a non-linear vector and test this. Written in Python 2.7.


from random import uniform, randint
from matplotlib import pylab as pl
from math import sqrt
from sklearn.linear_model import LinearRegression

#target function
def targetf(x1, x2, storage):
    tf = x1**2 + x2**2 - 0.6
    if tf >= 0:
        storage.append(-1)
    else:
        storage.append(1)

#generate sample and test data
def generate_data(input, output, N):
    for i in range(0, N):
        x_coor = uniform(-1.0, 1.0)
        y_coor = uniform(-1.0, 1.0)

        #non-linear transformation of data
        input.append([x_coor, y_coor, x_coor*y_coor, x_coor**2, y_coor**2])
        targetf(x_coor, y_coor, output)


#function to simulate noise by reversing a random 10% subset of data classifications
def simulate_noise(output):
    n = int(len(output) * 0.1)
    r = randint(0, len(output) - 1 - n)
    for j in range(r, r + n):
        output[j] = output[j] * (-1)

#sign function for binary classification
def sign_func(w0, w1, w2, w3, w4, w5, data_inputs):
    results = []
    for point in data_inputs:
        if w0 + (w1 * point[0]) + (w2 * point[1]) + (w3 * point[2]) + (w4 * point[3]) + (w5 * point[4]) < 0:
            results.append(-1)
        else:
            results.append(1)
    return results

#error checkng function - can also return X values for missed points if needed, e.g., to run through a perceptron later
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


#the experiement in repeatable form
def run(N):

    #generate training data
    X = []
    Y = []
    generate_data(X, Y, N)

    #generate testing data
    X_test = []
    Y_test = []
    generate_data(X_test, Y_test, 1000)

    #simulate noise
    simulate_noise(Y)
    simulate_noise(Y_test)

    #optional: leave uncommented to plot current run, turn off for mass runs
    plotty(X, Y)

    #linear regression
    lin = LinearRegression()
    lin.fit(X, Y)


    #Weights - w0 is intercept, w1 and w2 are the coefs
    #use W to formulate g, our chosen hypothesis
    coefs = lin.coef_
    w0 = lin.intercept_
    w1 = coefs[0]
    w2 = coefs[1]
    w3 = coefs[2]
    w4 = coefs[3]
    w5 = coefs[4]

    #evalute in sample data based on g
    train_results = sign_func(w0, w1, w2, w3, w4, w5, X)
    insamp_returned = getError(train_results, Y, float(N))
    E_in = insamp_returned['error']

    #evaluate out of sample data against g
    test_results = sign_func(w0, w1, w2, w3, w4, w5, X_test)
    outsamp_returned = getError(test_results, Y_test, 1000.0)
    E_out = outsamp_returned['error']

    #optional plotting of non-linear X vector vs. W vector
    pl.figure(2)
    pl.plot(X, lin.predict(X))
    pl.axis([-1, 1, -1, 1])
    pl.xlabel('X feature vector')
    pl.ylabel('Linear Regression Predictor')
    pl.title("Non-linear feature vector transformation of data")
    pl.show()

    #return error probability
    return {'E_in': E_in, 'E_out': E_out}


#function to plot initial data
def plotty(X, Y):
    pl.figure(1)
    for i in range(0, len(X) - 1):
        if Y[i] == 1:
            pl.scatter(X[i][0], X[i][1], color='blue')
        else:
            pl.scatter(X[i][0], X[i][1], color='red')

    circle1=pl.Circle((0,0), sqrt(0.6), edgecolor='k', facecolor='none')
    figure = pl.gcf()
    figure.gca().add_artist(circle1)
    pl.axis([-1, 1, -1, 1])
    pl.xlabel('x1 or feature 1')
    pl.ylabel('x2 or feature 2')
    pl.title('Generated Sample Data w/10% Simulated Noise')
    pl.show()




#experiment script controls
err_in_total = 0.0
err_out_total = 0.0

#number of runs
reps = 1

#run the experiement and get error totals
for i in range(0, reps):
    returned = run(1000)
    err_in_total += returned['E_in']
    err_out_total += returned['E_out']

#calculate averages
err_in_avg = err_in_total/float(reps)
err_out_avg = err_out_total/float(reps)

#print results
print 'The average in-sample error is %.3f' % err_in_avg
print 'The average out-of-sample error is %.3f' % err_out_avg



