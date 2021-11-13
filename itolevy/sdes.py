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



# Generic base class
class sde:
    
    def __init__(self, x = 0.0, T = 1.0, mu = lambda t,x: 0, sigma = lambda t,x: 1):
        """Initiailize an SDE solver with drift and volatility coefficient functions, initial point, and time-horizion.
        The parameters are 'x' the initial point, 'T', the time horizon to simulate a sample path over. 
        Default dynamics give a Brownian motion. The user must override the drift and volatility function themselves to 
        specify a more general SDE. These can be set using 'setCoef()' and passing two anonymous functions of '(t,x)'.
        A PDE-solver is provided as a method for computing conditional expecations of the process driven by the SDE, as well
        as an equivalent Monte-Carlo routine.
        """
        # Default dynamics are Brownian motion
        self.type = None
        self.mu = None
        self.sigma = None
        self.setSDE(mu, sigma)
        self.x = x
        self.T = T
        self.n = 5000
        # For PDE numerical scheme
        self.N = 100
        self.M = 100
        self.a = -1
        self.b = 1
        



    def __str__(self):
        return self.type+" SDE starting at "+str(self.x)+" simulated over [0, "+str(self.T)+"].\n"

    def indicator(self, event):
        """ The indicator function of an event (bool)

        Returns 1 if event is true and 0 otherwise. The argument 'event' can be an array of bools
        """
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


    def _checkCoefs(self, mu, sigma):
        """ Input handling to check coefficients are functions
        """
        if type(mu) != types.FunctionType or type(sigma) != types.FunctionType: 
            raise ValueError("Both drift function 'mu' and volatility function 'diffuse' must be functions")
        else:
            return True

    def _getCoefArgNum(self, mu, sigma):
        drift_args = len(signature(mu).parameters)
        vol_args = len(signature(sigma).parameters)
        return (drift_args, vol_args)
    
    def _setCoefs(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def setSDE(self, mu, sigma):
        # Check to make sure coefficients are functions
        if self._checkCoefs(mu, sigma):
            # Get count of parameters to classify type as (x) autonomous or time-inhomogeneuous (t, x)
            arg_counts = self._getCoefArgNum(mu, sigma)
            drift_args = arg_counts[0]
            vol_args = arg_counts[1]
            # If at least one coefficient function depends on time, it is inhomogeneous and we need a wrapper
            # to pass to Euler-Maruyama
            if drift_args == 2 or vol_args == 2:
                self.type = "non-autonomous"
                if drift_args == 1:
                    drift_wrapper = lambda t,x: mu(x)
                    self._setCoefs(drift_wrapper, sigma)
                elif vol_args == 1:
                    vol_wrapper = lambda t,x: sigma(x)
                    self._setCoefs(mu, vol_wrapper)
                else:
                    self._setCoefs(mu, sigma)
            elif drift_args == 1 and vol_args == 1:
                self.type = "autonomous"
                self._setCoefs(mu, sigma)

   

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

    def _tridiag(self, a, b, c, d):
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
    def _euler_maruyama(self):
        """ Simulate a sample path of a stochastic process via solving an SDE it
        behaves via Euler-Maruyama stochastic integration.
        The initial point of the process, the time-horizion of the simulation, 
        and the time-grid resolution can all be set with sde.x, etc.
        Returned is a numeric array/vector of the sample-path or a time-series.
        """
        if self.type != "non-autonomous":
            raise ValueError("Use milstein for SDEs with coefficients of only the state-variable and not time.")
        h = self.T/self.n
        y = np.zeros(self.n+1)
        y[0] = self.x     
        z = np.random.normal(size = self.n)
        for i in range(self.n):
            y[i+1] = y[i] + self.mu(i*h, y[i])*h+ self.sigma(i*h, y[i])*np.sqrt(h)*z[i]
        return y

    def _milstein(self, vp = None, h = 10**-5):
        """ Simulate a sample-path of an SDE via the Milstein method.
        This requires the derivative of the volatility function, and that both
        are functions of just a single state variable, i.e. this method is
        implemented for autonomous SDEs only.
        """
        if self.type != "autonomous":
            raise ValueError("Cannot use Milstein integration for non-autonomous SDEs.")
        # Check of derivative of sigma(x) is passed otherwise compute it numerically
        if vp is not None:
            vp_args = len(signature(self.sigma).parameters)
            if vp_args != 1:
                raise ValueError("The derivative of the volatility function must be a function of a single state-variable if it is passed exactly.")
        else:
            def vp(x):
                w = (self.sigma(x+h)-self.sigma(x-h))/(2*h)
                return w
        k = self.T/self.n
        y = np.zeros(self.n+1)
        y[0] = self.x
        
        for i in range(self.n):
            z = np.random.normal()
            dz = np.sqrt(k)*z
            y[i+1] = y[i] + self.mu(y[i])*k+self.sigma(y[i])*dz +0.5*self.sigma(y[i])*vp(y[i])*(dz**2-k)
        return y

    
    def ol_solve(self, x, T, mu, sigma, n = 1000, vp = None):
        
        self.setInitialPoint(x)
        self.setTimeHorizon(T)
        self.setSamplePathRes(n)
        self._setSDE(mu, sigma)
        if self.type == "autonomous":
            return self._milstein(vp)
        elif self.type == "non-autonomous":
            return self._euler_maruyama()
        else:
            raise ValueError("Bad SDE type.")

    def solve(self, vp = None):
        if self.type == "autonomous":
            return self._milstein(vp)
        elif self.type == "non-autonomous":
            return self._euler_maruyama()
        else:
            raise ValueError("Bad SDE type.")
    
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
            if self.type == "non-autonomous":
                alpha = (self.sigma(tt, x)**2)/(2*(h**2))-(self.mu(tt, x))/(2*h)
                beta = -rate(tt, x)-(self.sigma(tt, x)/h)**2
                delta = (self.sigma(tt, x)**2)/(2*(h**2))+(self.mu(tt, x))/(2*h)
            elif self.type == "autonomous":
                alpha = (self.sigma(x)**2)/(2*(h**2))-(self.mu(x))/(2*h)
                beta = -rate(tt, x)-(self.sigma(x)/h)**2
                delta = (self.sigma(x)**2)/(2*(h**2))+(self.mu(x))/(2*h)

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
            u[i+1, 1:self.M] = self._tridiag(a, b, c, di)
        if variational:
                for j in range(0, self.M+1):
                    u[i+1, j] = np.max(a = np.array([u[i+1, j], g(x[j])]))
        return u

    def path_ensemble(self, numpaths = 100, vp = None):
        """ Generate a large ensemble of sample paths for Monte-Carlo integration and for computing path-integrals numerically.
        (Parameters)
        int 'numpaths' the number of paths in the ensemble to generate.
        """
        ensemble = np.zeros(shape = (self.n+1, numpaths))
        for i in range(numpaths):
            ensemble[:, i] = self.solve(vp)
        return ensemble

    def monte_carlo(self, g, numpaths = 30, ensemble = None, vp = None):
        """ Compute conditional expectations via Monte-Carlo ensemble averaging.
        Here 'g' is the function to compute in E(g(X_T)|X_t=x)
        """
        if ensemble is None:
            ensemble = self.path_ensemble(numpaths, vp)
        return np.mean(g(ensemble[self.n, :]))

    def cond_exp(self, g, numpaths = 30, vp = None):
        """ Compute a conditional expectation of a function of a process driven by an SDE.
        """
        if type(g) != types.FunctionType:
            raise ValueError("The argument 'g' must be a function.")
        if len(signature(g).parameters) > 1:
            raise ValueError("The argument 'g' must be a function of one variable 'x'.")
        x1 = self.implicit_scheme(g)
        x1 = x1[self.N, int(self.M/2)]
        x2 = self.monte_carlo(g, numpaths = numpaths, vp = vp)
        return [x1, x2]

    # Plotting functions for sample paths and PDE solutions
    def plotSamplePath(self, s = None, vp = None):
        """ Simulate or pass a sample path and plot it."""
        if s is None:
            s = self.solve(vp)
        t = np.linspace(0, self.T, num = self.n+1)
        fig = plt.figure()
        plt.plot(t, s)
        plt.show();

    # Plotting functions for sample paths and PDE solutions
    def plotEnsemble(self, ensemble = None, numpaths = 30, vp = None):
        """ Simulate an ensemble of sample paths for the given dynamics of this sde.
        The argument 'numpaths' specifies the number of paths in the ensemble.
        """
        t = np.linspace(0, self.T, num = self.n+1)
        fig = plt.figure()
        if ensemble is None:
            for i in range(numpaths):
                s = self.solve(vp)
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
        time = self.T-np.linspace(0, self.T, self.N+1)
        space = np.linspace(self.a, self.b, self.M+1)
        # For 3D plotting
        time, space = np.meshgrid(time, space)
        fig = plt.figure()
        # Plotting solution surface
        ax = fig.add_subplot(111, projection = "3d")
        ax.plot_surface(time, space, np.transpose(u))
        plt.show();


class JumpDiffusion(sde):

    def __init__(self, x = 0, T = 1, mu = lambda t,x:0, sigma = lambda t,x: 1, lam = 1, alpha = 0, beta = 0.1):
        super().__init__(x = 0, T = T)
        # Constant parameters defining the jump component of the model
        self.jump_pars = (lam, alpha, beta)
        self.eta = None
        self._meanJumpSize()
        # Drift of log-returns under Merton jump diffusion with compensated jumps
        drift = lambda t,x : mu(t,x)-0.5*sigma(t,x)**2-lam*self.eta
        volat = lambda t,x : sigma(t,x)
        self.setSDE(drift, volat)


    def _meanJumpSize(self):
        self.eta = np.exp(self.jump_pars[1]+0.5*self.jump_pars[2]**2)-1

    # Overridden method from sde
    def euler_maruyama(self):
        """ Simulate a sample path of a stochastic process via solving an SDE it
        behaves via Euler-Maruyama stochastic integration.
        The initial point of the process, the time-horizion of the simulation, 
        and the time-grid resolution can all be set with sde.x, etc.
        Returned is a numeric array/vector of the sample-path or a time-series.
        """
        lam = self.jump_pars[0]
        alpha = self.jump_pars[1]
        beta = self.jump_pars[2]

        h = self.T/self.n
        if lam*h > 1:
            msg = "Number of time-steps in EM scheme too small for a Taylor approximation to the probability of a jump.\n"
            msg = msg + " You need n = "+ str(np.ceil(self.lam*self.T))
            raise ValueError(msg)
        y = np.zeros(self.n+1)
        y[0] = self.x     
        z = np.random.normal(size = self.n)
        u = np.random.uniform(size = self.n)
        for i in range(self.n):
            logj = 0
            if u[i] <= lam*h:
                logj = np.random.normal(alpha, beta)
            y[i+1] = y[i] + self.mu(i*h, y[i])*h+ self.sigma(i*h, y[i])*np.sqrt(h)*z[i]+logj
        return y

#=======================================================================================================================================================
# Now we can define custom classes for specific models with ease
# They will inherit the simulation methods, and we can go back 
# and add more that are model independent.
#======================================================================================================================================================

# Heston Mean-Reverting model (OU process)
class Heston(sde):
    def __init__(self, x, T, kappa, theta, xi):
        """ Heston mean-reversion: kappa = reversion speed, theta = rev. level, xi = vol of vol
        """
        super().__init__(x, T)
        if 2*kappa*theta <= xi**2:
            raise ValueError("Feller condition violated, try smaller vol-of-vol etc.")
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        mu = lambda x : self.kappa*(self.theta-x)
        sigma = lambda x : self.xi*np.sqrt(x)
        self.setSDE(mu, sigma)
        self.name = "Heston mean-reversion"


    def __str__(self):
        return super().__str__()+"Heston mean-reverting volatility (kappa, theta, xi) = "+"("+str(self.kappa)+", "+str(self.theta)+", "+str(self.xi)+")"

class Gbm(sde):
    def __init__(self, x, T, mu = 0, sigma = 1):
        """ Geometric Brownian motion.
        """
        super().__init__(x, T)
        if sigma <= 0:
            raise ValueError("Volatility 'sigma' must be positive.")
        self.pars = (mu, sigma)
        self.setSDE(lambda x: mu*x, lambda x: sigma*x)
        self.name = "GBM"


    def __str__(self):
        return super().__str__()+"Geometric brownian motion (mu, sigma) = "+str(self.pars)
   


    def fit(self, X, h = 1/252, alpha = 0.05):
        """ Fit the parameters of a GBM to a given time-series of log-returns.
        Returns the standard error on the mean estimate. Parameters for the GBM model are updated on the object itself.
        """
        mu = np.mean(X)/h
        sigma = np.std(X)/np.sqrt(h)
        # Accounting for volatility drag
        mu += 0.5*sigma**2

        # Computing standard error for mean-estimate (the volatility estimate will be pretty accurate,
        # but the drift won't unless T = 3000 years or something absurd as can be seen by a basic CLT
        # derivation of the confidence bounds for this specific Gaussian model.
        T = X.shape[0]*h
        epsilon = sigma*norm.ppf(1-alpha/2)/np.sqrt(T)
        self.pars = (mu, sigma)
        self.setSDE(lambda x: mu*x, lambda x: sigma*x)
        return epsilon

class Merton(JumpDiffusion):

    def __init__(self, x, T, mu = 0.1, sigma = 0.5, lam = 10, jump_mean = 0, jump_std = 0.4):
        self.pars = (mu, sigma, lam, jump_mean, jump_std)
        super().__init__(x, T, mu = lambda t,x:mu, sigma = lambda t,x:sigma, lam = lam, alpha = jump_mean, beta = jump_std)

    def pdf(self, x, t):
        """ Probability density function for log-returns in Merton's jump diffusion. This is a infinite weighted
        sum of Gaussian pdfs whose mean and variance depend on the number of jumps n in the Poisson process. The
        weights are exactly Poisson probabilities.
        (Parameters)
        'x' the log-return level
        't' the time to compute the density at
        (Returns)
        Either a float or np.ndarray 
        """
        mu = self.pars[0]
        sigma = self.pars[1]
        lam = self.pars[2]
        alpha = self.pars[3]
        beta = self.pars[4]
        eta = np.exp(alpha+0.5*beta**2)-1
        # 99% coverage for number of jumps in infinite Poisson weighted sum
        N = int(stats.poisson.ppf(0.99, t*lam))
        # Index to sum over
        n = np.arange(N+1)
        # Poisson probabilities and the conditional mean and variance conditional on number of jumps
        pois = stats.poisson.pmf(n, lam*t)
        m = (mu-0.5*sigma**2-lam*eta)*t+n*alpha
        v = np.sqrt(t*sigma**2+n*beta**2)
        # Return the weighted sum based on a single input or array
        if type(x) == float:
            phi = stats.norm.pdf(x, m, v)
            p = np.sum(pois*phi)
            return p
        elif type(x) is np.ndarray:
            p = np.zeros(x.size)
            for i in range(x.size):
                phi = stats.norm.pdf(x[i], m, v)
                p[i]=np.sum(pois*phi)
            return p
  
