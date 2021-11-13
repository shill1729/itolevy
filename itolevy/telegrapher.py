import itolevy.sde as sde
import numpy as np
import matplotlib.pyplot as plt

class telegraph:
    def __init__(self, x0, T, c):
        self.x0 = x0
        self.T = T
        self.c = c
        self._N = 500
        self._M = 500
        self._a = 1.5*(self.x0-self.c*self.T)
        self._b = 1.5*(self.x0+self.c*self.T)
        self._dx = (self._b-self._a)/self._M
        self._dt = self.T/self._M

    def setResolution(self, time_res, space_res):
        self._N = time_res
        self._M = space_res

    def _tridiag(self, a, b, c, d):
        """ Tridiagonal linear system solver using back-substitution, via the
        so-called Thomas algorithm.
        Here 'a' is the lower diagonal constant, 'b' the diagonal constant, and 'c' the upper diagonal constant
        of the tridiagonal matrix A in Ax=d, and this function assumes they are floats
        Finally, 'd' is the target vector of some given size greater than 1. Some
        systems will have NaN solutions or infinite.
        """
        n = d.size
        cc = np.zeros(n-1)
        dd = np.zeros(n)
        x = np.zeros(n)
        cc[0]=c/b
        for i in range(1, n-1, 1):
            cc[i]=c/(b-a*cc[i-1])
        dd[0] = d[0]/b
        for i in range(1, n, 1):
            dd[i] = (d[i]-a*dd[i-1])/(b-a*cc[i-1])
        x[n-1] = dd[n-1]
        for i in range(n-2, -1, -1):
            x[i]=dd[i]-cc[i]*x[i+1]
        return x

    def _initial_condition(self, x):
        return sde.indicator(np.abs(x - self.x0) < self._dx)/self._dx
    def _initial_velocity(self, x):
        return np.zeros(x.size)

    def implicit_scheme(self):
        k1 = 1+1/(self._dt*self.c**2)
        k2 = -1/(2*self._dt*self.c**2)
        alpha = 1-k2+self._dt/self._dx**2
        beta = -self._dt/(2*self._dx**2)
        x = np.linspace(self._a, self._b, self._M+1)
        u = np.zeros((self._N+1, self._M+1))
        # IC
        u[0,:] = self._initial_condition(x)
        u[1, ] = self._initial_velocity(x)*self.__dt+u[0, ]
        # BC
        u[:, 0] = self._initial_condition(x[0])
        u[:, self._M] = self._initial_condition(x[self._M])
        # BC are zero, default
        for i in range(2, self._N+1, 1):
            d = k1*u[i-1, 2:self._M]+k2*u[i-2, 2:self._M]
            u1 = self._tridiag(beta, alpha, beta, d)
            u[i, 2:self._M] = u1
        return u

    def plotPDE(self):
        """ Plot the PDE solution surface for a telegrapher PDE.
        """
        # Computing solution grid to PDE problem
        u = self.implicit_scheme()
        # 3D mesh
        time = np.linspace(0, self.T, self._N+1)
        space = np.linspace(self._a, self._b, self._M+1)
        # For 3D plotting
        time, space = np.meshgrid(time, space)
        fig = plt.figure()
        # Plotting solution surface
        ax = fig.add_subplot(111, projection = "3d")
        ax.plot_surface(time, space, u.transpose())
        plt.show();
