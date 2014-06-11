__author__ = 'alex'


#Using Gradient Descent to minimize error
#In this example, error surface = E(u,v) = (ue**v - 2ve**(-u))**2
#starting point (u,v) = (1,1) => weight vector
#learning rate n = 0.1
#I've found the partial derivative with respect to u to be 2(e**v + 2ve**(-u))(ue**v - 2ve**(-u))

from math import e

n = 0.1


def error(u, v):
    return ((u*e**v) - (2.0*v*e**(-u)))**2.0

def du(u, v):
    return 2.0*(e**v + 2.0*v*e**(-u)) * (u*e**v - 2.0*v*e**(-u))

def dv(u, v):
    return 2.0*(u*e**v - 2.0*e**(-u)) * (u*e**v - 2.0*v*e**(-u))


#recursive gradient descent function
def gradient_descent(u, v, counter):
    err = error(u, v)
    if err < 10.0**(-14):
        return {'error': err, 'iterations': counter, 'u': u, 'v':v}
    else:
        u_next = u + (-1.0)*n*du(u, v)
        v_next = v + (-1.0)*n*dv(u, v)
        counter += 1
        return gradient_descent(u_next, v_next, counter)



count = 0
returned = gradient_descent(1.0, 1.0, count)
print 'The error is %f and it took %d iterations.' % (returned['error'], returned['iterations'])
print 'The final weights are u = %f and v = %f.' % (returned['u'], returned['v'])


#coordinate descent
def coor_descent(u, v):
    u -= n*du(u, v)
    v -= n*dv(u, v)
    return {'error': error(u, v), 'u': u, 'v': v}


u = 1.0
v = 1.0
for j in range(0, 15):
    returned = coor_descent(u, v)
    u = returned['u']
    v = returned['v']

print 'The final error for coordinate descent is %f.' % returned['error']
print 'The final weights are u = %f and v = %f' % (returned['u'], returned['v'])

