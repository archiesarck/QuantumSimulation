import numpy as np
import sympy as sym
from sympy import oo, I, lambdify
from sympy.functions.special.polynomials import hermite as herm
from scipy.integrate import quad

#from scipy.special import hermite
#iota = np.complex(0,1)
#space limits
#a=-oo
#b=oo

x,t= sym.symbols('x t')

h=1
m=1

dim=40
Psi_0_no=1
potential_no=7

file = open("Psi_0.txt","r")
for i in range(Psi_0_no):
    Psi_0 = (file.readline()).rstrip()

#f_0 = sym.sympify(Psi_0)

file.close()

def b(n,x):  #orthonormal hermit based basis
    A = sym.sqrt( sym.sqrt(sym.pi) * 2**n * sym.factorial(n)  )
    return sym.exp(-(x**2)/2) * herm(n,x) / A

def innerProd(ket1, ket2, dtype = "ket"):
    if dtype =="ket" :
        sum=0
        for i in range(dim):
            sum+=np.conjugate(ket1[i]) * ket2[i]
        return sum
    
    elif dtype == "sym" :
        return sym.integrate(sym.conjugate(ket1)*ket2 , (x,-oo,oo) )
    
    elif dtype == "nu" :
        func = lambdify(x, sym.conjugate(ket1) * ket2 )
        return quad(func, -np.inf, np.inf)

def normalize(ket, dtype="ket"):
    if dtype=="ket" : 
        return ket/np.sqrt( innerProd(ket, ket,"ket") )
    
    elif dtype=="sym" :
        return ket/np.sqrt( innerProd(ket, ket,"sym") )
    
    elif dtype == "nu" :
        ip, err = innerProd(ket, ket,"nu")
        return ket/np.sqrt( ip )

f_0 = normalize(sym.sympify(Psi_0),"nu")

#obtaining previously computed hamiltonian matrix from file
H_mat = np.zeros( (dim,dim) )
file = open("H_mat (V_no=%d, Dim=%d, nu).txt"%(potential_no, dim) , "r")

for i in range(dim):
    row = np.fromstring( (file.readline()).rstrip(), dtype=float, sep=' ' )
    
    for j in range(i+1):
        H_mat[i,j]=row[j]
        H_mat[j,i]=np.conjugate(H_mat[i,j])

file.close()

#obtaining f_0_ket
f_0_ket = np.zeros( dim )
for i in range(dim): 
    f_0_ket[i], err = innerProd( b(i,x), f_0, "nu" )
    
    print("finding f_0_ket: ", i )

f_0_ket = normalize(f_0_ket, "ket")

#obtaining normalized eigen pairs
eVals, eKets = np.linalg.eig(H_mat)    
for i in range(dim):
    eKets[i] = normalize(eKets[i], "ket")
    
    print("normalizing eigenkets: ", i )

#obtaining waveFunc_ket
waveFunc_ket = eKets[0] * sym.exp(-I*eVals[0]*t/h) * innerProd(eKets[0], f_0_ket, "ket")
for i in range(1,dim):
    waveFunc_ket += eKets[i] * sym.exp(-I*eVals[i]*t/h) * innerProd(eKets[i], f_0_ket ,"ket")
    
    print("finding waveFunc_ket: ", i)
#hope that the waveFunc_ket obtained is near normalized at all times

waveFunc=0
for i in range(dim):
    waveFunc += waveFunc_ket[i]*b(i,x)


file = open("waveFunc(Psi_0_no=%d, V_no=%d, Dim=%d, nu).txt"%(Psi_0_no, potential_no, dim) , "w+")
file.write(str(waveFunc))
file.close()