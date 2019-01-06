#TO FIND LOWER HALF TRIANGLE OF HAMILTONIAN MATRIX, FOR A GIVEN POTENTIAL AND DIMENSION

import numpy as np
import sympy as sym
from math import factorial
from sympy import diff, lambdify
from sympy.functions.special.polynomials import hermite as herm
from scipy.integrate import quad

x,t= sym.symbols('x t')

h=1
m=1
k= 0.5

dim=40
potential_no=2
bounds = [-np.inf, np.inf]
bounded = False

file = open("potential.txt","r")
for i in range(potential_no):
    potential = (file.readline()).rstrip()

V=sym.sympify(potential)    

def A(n):
    return np.sqrt( np.sqrt(np.pi) * 2**n * factorial(n)  )

def b(n,x):  #orthonormal hermit based basis
    A = sym.sqrt( sym.sqrt(sym.pi) * 2**n * sym.factorial(n)  )
    return sym.exp(-(x**2)/2) * herm(n,x) / A

def H(func):  #hamiltonian operator (on x wave space)
    return -diff(func,x,x)*(h**2)/(2*m) + V*func

def kDel(i,j): #kronecker delta
    if i==j : return 1
    else : return 0

H_mat = np.zeros( (dim,dim) )

if bounded : s = ", bound"
else : s=""

file = open("H_mat (V_no=%d, Dim=%d, nu%s).txt"%(potential_no, dim,s), "w+")

for i in range(dim):
    for j in range(i+1):
        if not(bounded) and i >= 2 and j>= 2 :
            try:
                I1 = -4*k*j*(j-1)*kDel(i,j-2) * ( A(j-2)/A(j) )
                I2 = k*kDel(i,j)
                I3 = ( 4*k*j*( 0.5*kDel(i+1,j-1)*A(i+1)*A(j-1)  + i*kDel(i-1,j-1)*A(i-1)*A(j-1) ) )/( A(i)*A(j) ) 
                I4 = ( -k*( 0.25*kDel(i+2,j)*A(i+2)*A(j) + (i+0.5)*kDel(i,j)*A(i)*A(j) + i*(i-1)*kDel(i-2,j)*A(i-2)*A(j) ) )/( A(i)*A(j) )
                
                func = lambdify(x, V * b(i,x) * b(j,x) )
                I5, err = quad(func, -np.inf , np.inf)
                
                H_mat[i,j] = I1 + I2 + I3 + I4 + I5
                H_mat[j,i] = np.conjugate(H_mat[i,j])
            except:
                func = lambdify(x, sym.conjugate( b(i,x) ) * H( b(j,x) ) )
            
                H_mat[i,j], err = quad(func, -np.inf , np.inf)
                H_mat[j,i] = np.conjugate(H_mat[i,j])
    
        else:
            func = lambdify(x, sym.conjugate( b(i,x) ) * H( b(j,x) ) )
            
            H_mat[i,j], err = quad(func, bounds[0] , bounds[1])
            H_mat[j,i] = np.conjugate(H_mat[i,j])
            
        if(j<i) : file.write("%.50f "%H_mat[i,j])
        elif(j==i) : file.write("%.50f\n"%H_mat[i,j])
        
        print("finding H_mat: ", i, " ", j)

file.close()
