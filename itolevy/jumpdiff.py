from itolevy import sde

class merton(sde):

    def __init__(self, mu = 0, sigma = 1, lam = 1, alpha = 0, beta = 1, T = 1):
        super().__init__(x = 0, T = T)
        # Constant parameters defining the model
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self._meanJumpSize()
        # Drift of log-returns under Merton jump diffusion with compensated jumps
        drift = lambda t,x : self.mu-0.5*self.sigma**2-self.lam*self.eta
        volat = lambda t,x : self.sigma
        self.setCoef(drift, volat)


    def _meanJumpSize(self):
        self.eta = np.exp(self.alpha+0.5*self.beta**2)-1

    # Overridden method from sde
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
        if self.lam*h > 1:
            msg = "Number of time-steps in EM scheme too small for a Taylor approximation to the probability of a jump.\n"
            msg = msg + " You need n = "+ str(np.ceil(self.lam*self.T))
            raise ValueError(msg)
        y = np.zeros(self.n+1)
        y[0] = self.x     
        z = np.random.normal(size = self.n)
        u = np.random.uniform(size = self.n)
        for i in range(self.n):
            logj = 0
            if u[i] <= self.lam*h:
                logj = np.random.normal(self.alpha, self.beta)
            y[i+1] = y[i] + self.drift(i*h, y[i])*h+ self.volat(i*h, y[i])*np.sqrt(h)*z[i]+logj
        return y

    # Overridden method from sde
    def implicit_scheme(self, g, rate=lambda t,x: 0, run_cost=lambda t,x: 0, variational=False):
        """ Solve parabolic PIDEs with function coefficients 
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
        eta = np.exp(self.alpha+0.5*self.beta**2)-1
        # 99% coverage for number of jumps in infinite Poisson weighted sum
        N = int(stats.poisson.ppf(0.99, t*self.lam))
        # Index to sum over
        n = np.arange(N+1)
        # Poisson probabilities and the conditional mean and variance conditional on number of jumps
        pois = stats.poisson.pmf(n, self.lam*t)
        m = (self.mu-0.5*self.sigma**2-self.lam*self.eta)*t+n*self.alpha
        v = np.sqrt(t*self.sigma**2+n*self.beta**2)
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
        
    def stats(self, t = 1):
        m = self.drift(0, 0)*t
        v = self.volat(0, 0)*np.sqrt(t)
        return (m, v)
