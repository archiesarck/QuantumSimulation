import numpy as np
import sympy as sym
from sympy import oo, I, lambdify
from sympy.functions.special.polynomials import hermite as herm
from scipy.integrate import quad
from matplotlib import pyplot as plt

x,t= sym.symbols('x t')

h=1
m=1

dim=40
potential_no=2

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

#obtaining previously computed hamiltonian matrix from file
H_mat = np.zeros( (dim,dim) )
file = open("H_mat (V_no=%d, Dim=%d, nu).txt"%(potential_no, dim) , "r")

for i in range(dim):
    row = np.fromstring( (file.readline()).rstrip(), dtype=float, sep=' ' )
    
    for j in range(i+1):
        H_mat[i,j]=row[j]
        H_mat[j,i]=np.conjugate(H_mat[i,j])

file.close()

#obtaining normalized eigen pairs
eVals, eKets = np.linalg.eig(H_mat) 
   
for i in range(dim):
    eKets[i] = normalize(eKets[i], "ket")
    
    print("normalizing eigenkets: ", i )


file = open("potential.txt","r")
for i in range(potential_no):
    potential = (file.readline()).rstrip()

V=lambdify(x,sym.sympify(potential))

file.close()


for i in range(dim):
    for j in range(dim-1):  #or range(dim-i-1)
        if(eVals[j] > eVals[j+1]):
            temp = np.array(eKets[j], copy=True)
            eKets[j] = np.array(eKets[j+1], copy=True)
            eKets[j+1] = np.array(temp, copy=True)

eVals.sort()
plt.ylabel("eVal_i", fontsize=10)
plt.xlabel("i", fontsize=10)
plt.suptitle('eVal_i plot for V(x) = 0.5*x**2', fontsize=15)
plt.plot(range(dim) , eVals)
#plt.savefig("eFunc (V_no = %d, eVal_no = %d, Dim = %d).jpg"%(potential_no, i, dim),dpi=500)
#plt.gcf().clear() 
plt.show()
#print("plotting eFunc_no : ", i)