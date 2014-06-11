__author__ = 'alex'

#Testing an RBF vs a SVM with RBF kernel

from math import pi, sin
from classes import My_Rbf
from random import uniform
from sklearn.svm import SVC


def signf(x):
    if (x[1] - x[0] + 0.25*sin(pi*x[0])) >= 0:
        return 1.0
    else:
        return -1.0

def generate_data(N, X_array, Y_array):
    for i in range(0, N):
        x1 = uniform(-1.0, 1.0)
        x2 = uniform(-1.0, 1.0)
        X = [x1, x2]
        X_array.append(X)
        Y_array.append(signf(X))


def run(N):
    X = []
    Y = []
    generate_data(N, X, Y)

    #RBF - runs Lloyd's algorithm for initial random K clusters
    rbf = []
    rbf = My_Rbf(k=12, gamma=1.5)
    rbf.fit(X, Y)
    rbf_Ein = rbf.test(X, Y)

    #SVM - hard margin - C is really infinity
    svc = []
    svc = SVC(C=10000, gamma=1.5, kernel='rbf')
    svc.fit(X, Y)
    svc_Ein = (1.0 - svc.score(X, Y))

    #generate test data
    X_test = []
    Y_test = []
    generate_data(1000, X_test, Y_test)

    #test rbf
    rbf_Eout = rbf.test(X_test, Y_test)

    #test svc
    svc_Eout = (1.0 - svc.score(X_test, Y_test))

    #return results
    return {'rbf_Ein': rbf_Ein, 'svc_Ein': svc_Ein, 'rbf_Eout': rbf_Eout, 'svc_Eout': svc_Eout}


def many_runs(R):
    avg_svc_Ein = 0.0
    avg_rbf_Ein = 0.0
    avg_svc_Eout = 0.0
    avg_rbf_Eout = 0.0
    perfects_svc = 0.0
    perfects_rbf = 0.0
    svc_wins = 0.0
    for i in range(0, R):
        returned = run(100)
        avg_svc_Ein += returned['svc_Ein']
        avg_rbf_Ein += returned['rbf_Ein']
        avg_svc_Eout += returned['svc_Eout']
        avg_rbf_Eout += returned['rbf_Eout']
        if returned['svc_Ein'] == 0.0:
            perfects_svc += 1
        if returned['rbf_Ein'] == 0.0:
            perfects_rbf += 1
        if returned['svc_Eout'] < returned['rbf_Eout']:
            svc_wins += 1
        print i
    avg_svc_Ein = avg_svc_Ein/float(R)
    avg_rbf_Ein = avg_rbf_Ein/float(R)
    avg_svc_Eout = avg_svc_Eout/float(R)
    avg_rbf_Eout = avg_rbf_Eout/float(R)
    svc_wins = svc_wins/float(R) * 100.0

    print 'SVC: E_in = %f, E_out = %f, perfects = %f.' % (avg_svc_Ein, avg_svc_Eout, perfects_svc)
    print 'RBF: E_in = %f, E_out = %f, perfects = %f.' % (avg_rbf_Ein, avg_rbf_Eout, perfects_rbf)
    print 'SVC w/RBF Kernel beat RBF %f percent of the time.' % svc_wins


many_runs(100)

