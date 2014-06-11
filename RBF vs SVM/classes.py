__author__ = 'alex'


from math import e
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans


class My_SVM():
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.no_svs = 0
        self.W = []
        self.b = 0.0
        self.sv_count = 0.0

    #implement rbf kernel
    def rbf(self, x, xn):
        s = np.asmatrix(x) - np.asmatrix(xn)
        return e**(np.multiply((-1.0 * self.gamma), np.linalg.norm(s)**2))

    #create matrix Q of quadratic coefficients using rbf kernel method
    def quad(self, X, Y):
        Q = None
        row = None
        for i in range(0, len(X)):
            for j in range(0, len(X)):
                if row is not None:
                    row = np.hstack((row, Y[i]*Y[j] * self.rbf(X[i], X[j])))
                else:
                    row = Y[i]*Y[j] * self.rbf(X[i], X[j])
            if Q is not None:
                Q = np.vstack((Q, row))
                row = None
            else:
                Q = row
                row = None
        Q = np.asmatrix(Q)
        return Q

    #Lagrange equation for SVM
    def L(self, alpha, Q):
        a = alpha.reshape((len(alpha), 1))
        a = np.asmatrix(a)
        c = np.ones_like(a)
        c = np.asmatrix(c)
        a_T = a.getT()
        return 0.5*((a_T*Q)*a) - c.getT()*a

    #constraints - hard margin
    def con(self, alpha, Y):
        a = alpha.reshape((len(alpha), 1))
        a = np.asmatrix(a)
        r = Y*a
        r = r.item(0)
        return r

    def solve(self, X, Y, Q):
        #minimize alphas w/constraints
        Ym = np.asmatrix(Y)

        #initial guess
        guess = np.ones((len(X), 1))

        #conditions
        cons = ({'type': 'eq', 'fun': self.con, 'args': Ym},)
        for j in range(0, len(guess)):
            cons = cons + ({'type': 'ineq', 'fun': lambda alpha: alpha[j]},)

        #bounds on alpha
        tup = (.000001, 100000.0)
        bnds = []
        for i in range(0, len(X)):
            bnds.append(tup)

        #minimization of alpha, the Lagrange multiplier - note, comma after Q tricks into reading as
        #1 value in a tuple vs. separate values
        res = minimize(self.L, guess, args=(Q,), bounds=bnds, constraints=cons, method='SLSQP')
        return res

    def weights(self, X, Y, res):
        alphas = res.x

        #get weights from valid support vectors - probably should check the math here?
        w1 = 0.0
        w2 = 0.0
        sv_indexes = []
        for i in range(0, len(alphas)):
            if alphas[i] > 1.0e-03:
                w1 += alphas[i] * Y[i] * X[i][0]
                w2 += alphas[i] * Y[i] * X[i][1]
                self.sv_count += 1.0
                sv_indexes.append(i)

        W = [w1, w2]
        self.W = W
        #solve for b, or w0, using any SV
        Wm = np.asmatrix(W)
        try:
            n = sv_indexes[0]
        except IndexError:
            self.no_svs += 1
            return self.fit(X, Y)

        xn = np.asmatrix(X[n])
        xn = xn.getT()
        self.b = (1/Y[n]) - Wm*xn

    def fit(self, X, Y):
        Q = self.quad(X, Y)
        res = self.solve(X, Y, Q)
        self.weights(X, Y, res)


    def sign_func(self, data_inputs):
        results = []
        for point in data_inputs:
            if self.b + (self.W[0] * point[0]) + (self.W[1] * point[1]) < 0:
                results.append(-1)
            else:
                results.append(1)
        return results

    def getError(self, results, comparer):
        errors = 0
        for j in range(0, len(comparer)):
            if results[j] != comparer[j]:
                errors += 1
        error = errors/float(len(comparer))
        return error

    def test(self, X, Y):
        results = self.sign_func(X)
        error = self.getError(results, Y)
        return error



class My_Rbf():
    def __init__(self, k, gamma=1.0):
        self.gamma = gamma
        self.k = k
        self.W = []
        self.mus = []
        self.b = 0.0

    def rbf(self, x, mu):
        s = np.asmatrix(x) - np.asmatrix(mu)
        return e**(np.multiply((-1.0 * self.gamma), np.linalg.norm(s)**2))

    def lloyd(self, X):
        lloyder = KMeans(n_clusters=self.k, init='random')
        lloyder.fit(X)
        mus = lloyder.cluster_centers_
        return mus

    #construct NxK matrix using call to rbf
    def phi(self, X, mus):
        phi = None
        row = [1]
        for i in range(0, len(X)):
            for k in range(0, len(mus)):
                row = np.hstack((row, self.rbf(X[i], mus[k])))
            if phi is not None:
                phi = np.vstack((phi, row))
                row = [1]
            else:
                phi = row
                row = [1]
        phi = np.asmatrix(phi)
        return phi

    #get W from pseudo-inverse
    def weights(self, phi, Y):
        Ym = np.asmatrix(Y)
        W = np.linalg.pinv(phi.getT()*phi)*phi.getT()*Ym.getT() #might have to play with shape here...
        self.b = W[0]
        self.W = W[1::]



    def fit(self, X, Y):
        self.mus = self.lloyd(X)
        phi = self.phi(X, self.mus)
        self.weights(phi, Y)


    #rbf sign func - Gaussian distance to centers
    def sign_func(self, data_inputs):
        results = []
        for point in data_inputs:
            a = 0.0
            for i in range(0, len(self.W)):
                a += self.W[i] * self.rbf(point, self.mus[i])
            a += self.b
            if a < 0:
                results.append(-1)
            else:
                results.append(1)
        return results

    def getError(self, results, comparer):
        errors = 0
        for j in range(0, len(comparer)):
            if results[j] != comparer[j]:
                errors += 1
        error = errors/float(len(comparer))
        return error

    def test(self, X, Y):
        results = self.sign_func(X)
        error = self.getError(results, Y)
        return error