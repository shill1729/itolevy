import numpy as np     # installed with matplotlib
import matplotlib.pyplot as plt
import types

from mpl_toolkits.mplot3d import axes3d
from inspect import signature
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.stats as stats
# On windows: python -m pip install git+https://github.com/shill1729/itolevy.git

def indicator(event):
        
        if type(event) == np.bool_:
            if event:
                return 1
            else:
                return 0
        else:
            y = np.zeros(event.size)
            for i in range(event.size):
                if event[i]:
                    y[i] = 1 
            return y

# Generic base class (to be made abstract eventually and then...)
class sde:
    
    def __init__(self, x = 0.0, T = 1.0, drift = lambda t,x: 0, volat = lambda t,x: 1):
        """Initiailize an SDE solver with drift and volatility coefficient functions, initial point, and time-horizion.
        The parameters are 'x' the initial point, 'T', the time horizon to simulate a sample path over. 
        Default dynamics give a Brownian motion. The user must override the drift and volatility function themselves to 
        specify a more general SDE. These can be set using 'setCoef()' and passing two anonymous functions of '(t,x)'.
        A PDE-solver is provided as a method for computing conditional expecations of the process driven by the SDE, as well
        as an equivalent Monte-Carlo routine.
        """
        # Default dynamics are Brownian motion
        self.drift = drift
        self.volat = volat
        self.x = x
        self.T = T
        self.n = 5000
        # For PDE numerical scheme
        self.N = 100
        self.M = 100
        self.a = -1
        self.b = 1



    def __str__(self):
        return "SDE starting at "+str(self.x)+" simulated over [0, "+str(self.T)+"].\n"

    def setDrift(self, drift):
        self.drift = drift

    def setVolatility(self, volat):
        self.volat = volat

    def setCoef(self, drift, volat):
        self.drift = drift
        self.volat = volat

    def setInitialPoint(self, x):
        self.x = x

    def setTimeHorizon(self, T):
        self.T = T

    def setSamplePathRes(self, n):
        self.n = n

    def setGridRes(self, N, M):
        self.N = N
        self.M = M

    def setRegion(self, a, b):
        self.a = a
        self.b = b

    def tridiag(self, a, b, c, d):
        """ Tridiagonal linear system solver using back-substitution, via the
        so-called Thomas algorithm.
        Here 'a' is the lower diagonal, 'b' the diagonal, and 'c' the upper diagonal
        of the tridiagonal matrix A in Ax=d, and this function assumes they are
        vectors/arrays.
        Finally, 'd' is the target vector of some given size greater than 1. Some
        systems will have NaN solutions or infinite.
        """
        n = d.size
        cc = np.zeros(n-1)
        dd = np.zeros(n)
        x = np.zeros(n)
        cc[0]=c[0]/b[0]
        for i in range(1, n-1, 1):
            cc[i]=c[i]/(b[i]-a[i-1]*cc[i-1])
        dd[0] = d[0]/b[0]
        for i in range(1, n, 1):
            dd[i] = (d[i]-a[i-1]*dd[i-1])/(b[i]-a[i-1]*cc[i-1])
        x[n-1] = dd[n-1]
        for i in range(n-2, -1, -1):
            x[i]=dd[i]-cc[i]*x[i+1]
        return x

    # For passing user-defined drift and diffusion functions
    def euler_maruyama(self):
        """ Simulate a sample path of a stochastic process via solving an SDE it
        behaves via Euler-Maruyama stochastic integration.
        The initial point of the process, the time-horizion of the simulation, 
        and the time-grid resolution can all be set with sde.x, etc.
        Returned is a numeric array/vector of the sample-path or a time-series.
        """
        if type(self.drift) != types.FunctionType or type(self.volat) != types.FunctionType: 
            raise ValueError("Both drift function 'drift' and volatility function 'diffuse' must be a function of (t,x)")
        drift_args = len(signature(self.drift).parameters)
        vol_args = len(signature(self.volat).parameters)
        if drift_args != 2 or vol_args != 2:
            raise ValueError("Both drift function 'drift' and volatility function 'diffuse' must be a function of (t,x). Use 'milstein' for functions of one state variable")
        h = self.T/self.n
        y = np.zeros(self.n+1)
        y[0] = self.x     
        z = np.random.normal(size = self.n)
        for i in range(self.n):
            y[i+1] = y[i] + self.drift(i*h, y[i])*h+ self.volat(i*h, y[i])*np.sqrt(h)*z[i]
        return y

    def milstein(self, diffuse_1 = None, h = 0.001):
        """ Simulate a sample-path of an SDE via the Milstein method.
        This requires the derivative of the volatility function, and that both
        are functions of just a single state variable, i.e. this method is
        implemented for autonomous SDEs only.
        """

        if type(self.drift) != types.FunctionType or type(self.volat) != types.FunctionType: 
            raise ValueError("Both drift function 'm' and volatility function 'diffuse' must be a function of (t,x).")
        drift_args = len(signature(self.drift).parameters)
        vol_args = len(signature(self.volat).parameters)
        if drift_args != 1 or vol_args != 1:
            raise ValueError("Both drift function 'drift' and volatility function 'diffuse' must be a function of (x). Use 'euler_maruyama' for functions of time and one state variable.")
        if vp is not None:
            vp_args = len(signature(self.volat).parameters)
            if vp_args != 1:
                raise ValueError("The derivative of the volatility function must be a function of a singel state-variable if it is passed exactly.")
        else:
            def vp(x):
                w = (self.volat(x+h)-self.volat(x-h))/(2*h)
                return w
        k = self.T/self.n
        y = np.zeros(self.n+1)
        y[0] = self.x
        
        for i in range(self.n):
            z = np.random.normal()
            dz = np.sqrt(k)*z
            y[i+1] = y[i] + self.drift(y[i])*k+self.volat(y[i])*dz +0.5*self.volat(y[i])*vp(y[i])*(dz**2-k)
        return y

    
    
    def implicit_scheme(self, g, rate = lambda t,x:0, run_cost = lambda t,x:0, variational = False):
        """ Solve parabolic PDEs with function coefficients 
        and initial condition using an implicit finite-difference scheme.
        The argument 'g' is the terminal condition and must be a function of 
        a single float variable representing the state (x), while the functions 
        'rate', 'run_cost' must be functions of two variables (t,x), time and state.
        The functions 'drift' and 'diffuse' are the infinitesimal drift and volatility coefficient
        functions while the functions 'rate' and 'run_cost' are the discounting function and 
        running cost function.
        The argument 'variational' is a boolean that if false will
        solve the problem PDE = 0, and if true will solve the variational inequality
        PDE <= 0.
        Returned is the solution matrix with rows following time and columns state.
 
        """
        if type(g) != types.FunctionType:
            raise ValueError("Argument 'g' must be a single-variable function")
        k = self.T/self.N
        h = (self.b-self.a)/self.M
        
        x = np.linspace(self.a, self.b, self.M+1)
        
        u = np.zeros((self.N+1, self.M+1))
        # IC
        u[0,:] = g(x)
        # BC
        u[:, 0] = g(x[0])
        u[:, self.M] = g(x[self.M])
        
        # Time-stepping integration
        for i in range(0, self.N, 1):
            tt = self.T-i*k
            alpha = (self.volat(tt, x)**2)/(2*(h**2))-(self.drift(tt, x))/(2*h)
            beta = -rate(tt, x)-(self.volat(tt, x)/h)**2
            delta = (self.volat(tt, x)**2)/(2*(h**2))+(self.drift(tt, x))/(2*h)
            ff = run_cost(tt, x[1:self.M])
            if type(beta) == float:
                beta = np.repeat(beta, self.M)
            if type(alpha) == float:
                alpha = np.repeat(alpha, self.M-1)
                delta = np.repeat(delta, self.M-1)
            if type(ff) == float:
                ff = np.repeat(ff, self.M-1)
            a = -k*alpha[1:self.M]
            b = 1-k*beta[0:self.M]
            c = -k*delta[0:(self.M-1)]
            # Setting up the target vector
            d = np.zeros(self.M-1)
            d[0] = alpha[0]*u[0, 0]
            d[self.M-2] = delta[self.M-2]*u[0, self.M]
            di = u[i, 1:self.M] + k*(d+ff)
            u[i+1, 1:self.M] = self.tridiag(a, b, c, di)
        if variational:
                for j in range(0, self.M+1):
                    u[i+1, j] = np.max(a = np.array([u[i+1, j], g(x[j])]))
        return u

    def path_ensemble(self, numpaths = 100):
        """ Generate a large ensemble of sample paths for Monte-Carlo integration and for computing path-integrals numerically.
        (Parameters)
        int 'numpaths' the number of paths in the ensemble to generate.
        """
        ensemble = np.zeros(shape = (self.n+1, numpaths))
        for i in range(numpaths):
            ensemble[:, i] = self.euler_maruyama()
        return ensemble

    def monte_carlo(self, g, ensemble = None, numpaths = 30):
        """ Compute conditional expectations via Monte-Carlo ensemble averaging.
        Here 'g' is the function to compute in E(g(X_T)|X_t=x)
        """
        if ensemble is None:
            ensemble = np.zeros(shape = (self.n+1, numpaths))
            for i in range(numpaths):
                ensemble[:, i] = self.euler_maruyama()

        return np.mean(g(ensemble[self.n, :]))

    def cond_exp(self, g, numpaths = 30):
        """ Compute a conditional expectation of a function of a process driven by an SDE.
        """
        if type(g) != types.FunctionType:
            raise ValueError("The argument 'g' must be a function of one variable 'x'.")
        if len(signature(g).parameters) > 1:
            raise ValueError("The argument 'g' must be a function of one variable 'x'.")
        x1 = self.implicit_scheme(g)
        x1 = x1[self.N, int(self.M/2)]
        x2 = self.monte_carlo(g, numpaths = numpaths)
        return [x1, x2]

    # Plotting functions for sample paths and PDE solutions
    def plotSamplePath(self, s = None):
        """ Simulate or pass a sample path and plot it."""
        if s is None:
            s = self.euler_maruyama()
        t = np.linspace(0, self.T, num = self.n+1)
        fig = plt.figure()
        plt.plot(t, s)
        plt.show();

    # Plotting functions for sample paths and PDE solutions
    def plotEnsemble(self, ensemble = None, numpaths = 30):
        """ Simulate an ensemble of sample paths for the given dynamics of this sde.
        The argument 'numpaths' specifies the number of paths in the ensemble.
        """
        t = np.linspace(0, self.T, num = self.n+1)
        fig = plt.figure()
        if ensemble is None:
            for i in range(numpaths):
                s = self.euler_maruyama()
                plt.plot(t, s)
        else:
            for i in range(ensemble.shape[1]):
                plt.plot(t, ensemble[:, i])
        plt.show();

    def plotPDE(self, g, rate = lambda t,x:0, run_cost = lambda t,x:0, variational = False):
        """ Plot the PDE solution surface for a given Feynman-Kac problem. This is defined by a 
        terminal cost function 'g', a discounting rate, a running_cost, and a boolean for variational inequalitiy problems instead
        of pure PDEs.
        """
        # Computing solution grid to PDE problem
        u = self.implicit_scheme(g, rate, run_cost, variational)
        print(u[self.N, int(self.M/2)])
        # 3D mesh
        time = self.T-discretize.time_grid(self.T, self.N)
        space = discretize.space_grid(self.a, self.b, self.M)
        # For 3D plotting
        time, space = np.meshgrid(time, space)
        fig = plt.figure()
        # Plotting solution surface
        ax = fig.add_subplot(111, projection = "3d")
        ax.plot_surface(time, space, np.transpose(u))
        plt.show();

# Now we can define custom classes for specific models with ease
# They will inherit the simulation methods, and we can go back 
# and add more that are model independent.
class Heston(sde):
    def __init__(self, x, T, kappa, theta, xi):
        super().__init__(x, T)
        if 2*kappa*theta <= xi**2:
            raise ValueError("Feller condition violated, try smaller vol-of-vol etc.")
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.drift = lambda t,x : self.kappa*(self.theta-x)
        self.volat = lambda t,x : self.xi*np.sqrt(x)

    def __str__(self):
        return super().__str__()+"Heston mean-reverting volatility (kappa, theta, xi) = "+"("+str(self.kappa)+", "+str(self.theta)+", "+str(self.xi)+")"

class Gbm(sde):
    def __init__(self, x, T, mu = 0, sigma = 1):
        super().__init__(x, T)
        self.mu = mu
        self.sigma = sigma
        self.drift = lambda t,x : self.mu*x
        self.volat = lambda t,x : self.sigma*x
        self.kelly = mu/sigma**2
        self.sharpe = 0.5*(mu/sigma)**2

    def __str__(self):
        return super().__str__()+"Geometric brownian motion (mu, sigma) = "+"("+str(self.mu)+", "+str(self.sigma)+")"
    
    def __lt__(self, other):
        """ Compare GBMs via Kelly-growth rates"""
        if self.sharp < other.sharp:
            return True

    def kelly_criterion():
        return self.kelly

    def kelly_growth():
        return self.sharpe

    def fit(self, log_returns, h = 1/252, alpha = 0.05):
        """ Fit the parameters of a GBM to a given time-series of log-returns.
        Returns a vector of (mu, sigma, epsilon) where the last entry is the error on the estimate for the first.
        """
        mu = np.mean(log_returns)/h
        sigma = np.std(log_returns)/np.sqrt(h)
        # Accounting for volatility drag
        mu += 0.5*sigma**2
        T = log_returns.size*h
        epsilon = sigma*norm.ppf(1-alpha/2)/np.sqrt(T)
        return mu, sigma, epsilon

    def updateModel(self, param):
        """ Pass a vector of parameters and update the dynamics of the GBM.
        """
        mu = param[0]
        sigma = param[1]
        self.mu = mu
        self.sigma = sigma
        self.kelly = self.mu/self.sigma**2
        self.sharpe = 0.5*(mu/sigma)**2
        self.setCoef(lambda t,x:x*self.mu, lambda t,x:x*self.sigma)

    def getParameters(self):
        """ Return a vector of the parameters.
        """
        return np.array([self.mu, self.sigma])
