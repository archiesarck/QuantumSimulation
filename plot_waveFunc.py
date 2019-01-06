import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from sympy import lambdify, sympify, symbols

x,t = symbols('x t')

dim=40
Psi_0_no=1
potential_no=7

file = open("waveFunc(Psi_0_no=%d, V_no=%d, Dim=%d, nu).txt"%(Psi_0_no, potential_no, dim) , "r")
waveFunc = sympify(file.read())

file.close()

f=lambdify((x,t),waveFunc)

fig = plt.figure()
ax = plt.axes(xlim=(-7.5, 7.5), ylim=(-0.05,1.25) )
X = np.arange(-10, 10, 0.01)
line, = ax.plot(X, np.sin(X))


def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(X))
    return line,


def animate(t):
    line.set_ydata( abs(f(X,0.5*t)**2 ) )  # update the data.
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1, blit=True, save_count=50)


x=symbols('x')

file = open("potential.txt","r")
for i in range(potential_no):
    potential = (file.readline()).rstrip()

V=lambdify(x,sympify(potential))

file.close()

#plt.legend(["Psi(x,t)","Psi(x,0)" "V(x) = 0.5*x**2", "Psi(x,0) = exp(-(x-0.25)**2/2)"])

plt.xlabel( "waveFunc(Psi_0_no = %d, V_no = %d, Dim = %d)"%(Psi_0_no, potential_no, dim) )
ani.save( "waveFunc(Psi_0_no=%d, V_no=%d, Dim=%d, nu).htm"%(Psi_0_no, potential_no, dim) )

plt.show()
