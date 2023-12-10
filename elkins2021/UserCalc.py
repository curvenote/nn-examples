import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d, pchip


import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth']=2
mpl.rcParams['font.size']=16

# This is the equilibrium transport model class, identical to the original UserCalc of Spiegelman (2000).

class EquilTransport:
    ''' A class for calculating radioactive decay chains assuming equilibrium transport after Spiegelman (2000). 

    Interface:
    ----------
        model(alpha0,lambdas,D,W0,F,dFdz,phi,rho_f=2800.,rho_s=3300.,method=method,Da=inf)

    Parameters:
    -----------
        alpha0  :  numpy array of initial activities
        lambdas :  numpy array of decay constants scaled by solid transport time
        D       :  Function D(z) -- returns an array of partition coefficents at scaled height z
        W0      :  float -- Solid mantle upwelling rate
        F       :  Function F(z) -- returns the degree of melting F  
        dFdz    :  Function dFdz(z) --  returns the derivative of F  
        phi     :  Function phi(z) --  returns the porosity  
        rho_f   :  float -- melt density
        rho_s   :  float -- solid density
        method  :  string -- ODE time-stepping scheme to be passed to solve_ivp (one of 'RK45', 'Radau', 'BDF')
        Da      :  float -- Dahmköhler Number (defaults to \inf, unused in equilibrium model)

    Required Method:
    ----------------
        model.solve(): returns depth and log concentration numpy arrays  z, Us, Uf
    '''
    def __init__(self,alpha0,lambdas,D,W0,F,dFdz,phi,rho_f=2800.,rho_s=3300.,method='Radau',Da=np.inf):
        self.alpha0 = alpha0
        self.N = len(alpha0)
        self.D = D
        self.D0 = np.array([D[i](0.) for i in range(self.N)])
        self.W0 = W0
        self.lambdas = lambdas
        self.F  = F
        self.dFdz = dFdz
        self.phi = phi
        self.Da = Da
        self.rho_f = rho_f
        self.rho_s = rho_s
        self.method = method

    def rhs(self,z,Ur):
        '''
        Returns right-hand side of the decay chain problem for the log of the concentration,
        split into radiogenic and stable components:

        Uf = U^st + U^r where
        U^st is the log of the stable element concentrations U^s = log(D(0)/Fbar_z)
        U^r is the radiogenic ingrowth component

        (Solid concentration is not necessary here, but would simply be the fluid concentration
        times the partition coefficient for each depth z.)

        The general equation is:

        dU_i^r/dz = h\lambda_i/Weff_i * [ R_i^{i-1} exp(Uf_{i-1} - Uf_i) - 1.)

        lambda, D, D0, lambda_tmp, phi0, W_0, and alpha_0 are set by the UserCalc driver routine.
        '''

        # Determine F_bar(z) and rho_bar(z) once:
        Fb = self.F_bar(z)
        rb = self.rho_bar(z)

        # Identify the initial values for the partition coefficients:
        D0 = self.D0

        # Identify the initial values for densities:
        rho_f = self.rho_f
        rho_s = self.rho_s

        # Define the stable melt concentration:
        Ust = np.log(D0/Fb)

        # Define the total melt concentration:
        Uf = Ust + Ur

        # Calculate the effective velocity and scaled decay rates:
        lambda_prime = self.lambdas*rb/Fb

        # Calculate the ingrowth factor and exponential factor:
        R = np.zeros(len(lambda_prime))
        expU = np.zeros(len(lambda_prime))
        for i in range(1,len(lambda_prime)):
            R[i] = self.alpha0[i]*D0[i]/D0[i-1]*rb[i-1]/rb[i]
            expU[i] = np.exp(Uf[i-1]-Uf[i])

        # Return the full right-hand side of the equation:
        return lambda_prime*(R*expU - 1.)

    def solve(self,z_eval=None):
        '''
        Solves the radioactive decay chain problem as an ODE initial value problem.
        
        Parameters:
        ----------
            z_eval:  array
                array of depths to output solutions.  If None: the output at every defined z_eval depth.
        '''

        # Set initial condition and solve the ODE:
        Ur_0 = np.zeros(len(self.D0))
        sol = solve_ivp(self.rhs,(0.,1.),Ur_0,t_eval=z_eval,method=self.method)
        z = sol.t
        Ur = sol.y

        # Calculate the stable component of the melt concentration:
        Ust = np.log(self.D0/self.F_bar(z)).T

        # Calculate the total melt concentration:
        Uf = Ur + Ust

        # Placeholder for solid concentration (not used here, but cannot be blank):
        Us = Uf

        # Return the log concentrations with depth:
        return z,Us,Uf

    # Additional utility functions for returning the scaled degree of melting and density:

    def F_bar(self,zp):
        ''' returns  numpy array of size (len(zp),len(D)) for:
            Fbar_D = F + (1. - F)*D_i
        '''
        D = self.D
        F = self.F(zp)
        if np.isscalar(zp):
            F_bar_D = np.array([ F + (1. - F)*D[i](zp) for i in range(len(D))])
        else :
            F_bar_D = np.zeros((len(zp),len(D)))
            F_bar_D = np.array([ F + (1. - F)*D[i](zp) for i in range(len(D))]).T
        return F_bar_D

    def rho_bar(self,zp):
        ''' returns numpy array of  size (len(zp),len(D)) for:
                rho_bar_D = rho_f/rho_s*phi + (1. - phi)*D_i
        '''
        rho_s = self.rho_s
        rho_f = self.rho_f

        phi = self.phi(zp)
        D = self.D
        if np.isscalar(zp):
            rho_bar_D = np.array([ rho_f/rho_s*phi + (1. - phi)*D[i](zp) for i in range(len(D))])
        else:
            rho_bar_D = np.zeros((len(zp),len(D)))
            rho_bar_D = np.array([ rho_f/rho_s*phi + (1. - phi)*D[i](zp) for i in range(len(D))]).T

        return rho_bar_D

# This is the disequilibrium transport model described in Elkins and Spiegelman (submitted).

class DisequilTransport:
    ''' A class for calculating radioactive decay chains for scaled chemical reactivity and
    disequilibrium scenarios, as described in Elkins and Spiegelman (submitted).

    Interface:
    ----------
        model(alpha0,lambdas,D,W0,F,dFdz,phi,rho_f=2800.,rho_s=3300.,method=method,Da=inf)

    Parameters:
    -----------
        alpha0  :  numpy array of initial activities
        lambdas :  numpy array of decay constants scaled by solid transport time
        D       :  Function D(z) -- returns an array of partition coefficents at scaled height z
        W0      :  float -- Solid mantle upwelling rate
        F       :  Function F(z) -- returns the degree of melting F  
        dFdz    :  Function dFdz(z) --  returns the derivative of F  
        phi     :  Function phi(z) --  returns the porosity  
        rho_f   :  float -- melt density
        rho_s   :  float --  solid density
        method  :  string -- ODE time-stepping scheme to be passed to solve_ivp (one of 'RK45', 'Radau', 'BDF')
        Da      :  float -- Dahmköhler Number (defaults to \inf, unused in equilibrium model)

    Required Method:
    ----------------
        model.solve(): returns depth and log concentration numpy arrays  z, Us, Uf
    '''
    def __init__(self,alpha0,lambdas,D,W0,F,dFdz,phi,rho_f=2800.,rho_s=3300.,method='Radau',Da=0.):
        self.alpha0 = alpha0
        self.N = len(alpha0)
        self.D = lambda zp: np.array([D[i](zp) for i in range(self.N) ])
        self.D0 = self.D(0.)
        self.W0 = W0
        self.lambdas = lambdas
        self.F  = F
        self.dFdz = dFdz
        self.phi = phi
        self.Da = Da
        self.rho_f = rho_f
        self.rho_s = rho_s
        self.method = method

    def rhs(self,z,U):
        ''' Returns right hand side of decay chain problem for the log of concentration of the solid
        U^s = U[:N]  where N=length of the decay chain
        and
        U^f = U[N:]

        The full equations for dU/dz are given by Eqs 28 and 29 in Spiegelman and Elliott (1993).
        Log concentrations are calculated similar to methods in Spiegelman (2000).

        lambda, D, D0, lambda_tmp, phi0, W_0, and alpha_0 are set by the UserCalc driver routine.
        '''

        # Determine F(z) and D(z):
        F = self.F(z)
        dFdz = self.dFdz(z)
        D = np.array(self.D(z))
        phi = self.phi(z)

        # Identify the initial values for the partition coefficients:
        D0 = self.D0

        # Define the Dahmköhler number:
        Da = self.Da

        # Separate U into solid and melt components:
        N = self.N
        Us = U[:N]
        Uf = U[N:]

        # Calculate the radiogenic ingrowth terms:
        Rs = -np.ones(N)
        Rf = -np.ones(N)
        a0 = self.alpha0
        for i in range(1,N):
            Rs[i] += a0[i-1]/a0[i]*np.exp(Us[i-1]-Us[i])
            Rf[i] += (D0[i]*a0[i-1])/(D0[i-1]*a0[i])*np.exp(Uf[i-1]- Uf[i])

        # Calculate the change in log solid concentrations:
        dUs = ((1. - 1./D)*dFdz + Da/D*(D/D0*np.exp(Uf-Us)-1.) + self.lambdas*(1. - phi)*Rs) / (1. - F)

        # Calculate the change in log liquid concentrations:
        if F == 0.:
            dUf = (dUs*(dFdz + Da)/dFdz + self.lambdas*Rf) / (2. + Da/dFdz)
        else:
            dUf = (D0/D*np.exp(Us-Uf)-1.)*(dFdz + Da)/F + (self.rho_f*phi)/(self.rho_s*F)*self.lambdas*Rf

        # Return the full right-hand side of the equation:
        dU=np.zeros(2*N)
        dU[:N] = dUs
        dU[N:] = dUf

        return dU

    def solve(self,z_eval=None):
        ''' Solves the radioactive decay chain problem as an ODE initial value problem.
        If z_eval = None, every point is saved
        Else will save output at every z_eval depth
        '''
        # Set initial conditions and solve the ODE:
        N = self.N
        U0 = np.zeros(2*N)
        try:
            sol = solve_ivp(self.rhs,(0.,1.),U0,t_eval=z_eval,method=self.method)
            z = sol.t
            U = sol.y
        except ValueError as err:
            print('Warning:  solver did not complete, returning NaN: {}'.format(err))
            z = np.array([1.])
            U = np.NaN*np.ones((2*N,len(z)))

        # Return the log concentrations with depth:
        Us = U[:N,:]
        Uf = U[N:,:]
        return z,Us,Uf

# This is the main UserCalc driver class:
class UserCalc:
    ''' A class for constructing solutions for 1-D, steady-state, open-system  U-series transport calculations
        as in Spiegelman (2000) and Elkins and Spiegelman (submitted).

        Usage:
        ------
        
            us = UserCalc(df,dPdz = 0.32373,n = 2.,tol=1.e-6,phi0 = 0.008,
                      W0 =3.,model=EquilTransport,Da=None,stable=False,method='Radau')

        Parameters:
        -----------
            df     : A pandas dataframe with columns ['P','F', Kr','DU','DTh','DRa','DPa']
            dPdz   : float -- Pressure gradient, to convert pressure P to depth z
            n      : float -- Permeability exponent
            tol    : float -- Tolerance for the ODE solver
            phi0   : float -- Reference melt porosity
            W0     : float -- Upwelling velocity (cm/yr)
            model  : class -- A U-series transport model class (one of EquilTransport or DisequilTransport)
            Da     : float -- Optional Da number for disequilibrium transport model
            stable : bool
                 True:  calculates concentrations for non-radiogenic nuclides with same chemical properties 
                        (i.e. sets lambda=0)
                 False: calculates the full radiogenic problem
            method: string
                ODE time-stepping method to pass to solve_ivp (usually one of 'Radau', 'BDF', or 'RK45')
            
        Principal Methods:
        --------
            phi                   :  returns porosity as a function of column height
            set_column_parameters :  resets principal column parameters phi0, n, W0
            solve_1D              :  1D column solution for a single Decay chain 
                                        with arbitrary D, lambda, alpha_0
            solve_all_1D          : Solves a single column model for both 238U and 235U chains.  
                                        returns a pandas dataframe
            solve_grid            : Solves multiple column models for a grid of porosities and upwelling rates
                                        returns a 3-D array of activity ratios    
    '''

    def __init__(self,df,dPdz = 0.32373,n = 2.,tol = 1.e-6,phi0 = 0.008,W0 = 3.,
                 model=EquilTransport,Da=None,stable=False,method='Radau'):
        self.model = model
        self.Da = Da
        self.method = method

        # Check that the model selection and Da number are compatible:
        if isinstance(model, DisequilTransport):
            if self.Da is None or not isinstance(self.Da, float):
                raise ValueError('Da must be set to  a floating point number for disequilibrium transport')

        self.df = df
        self.dPdz = dPdz
        self.n = n
        self.tol = 1.e-6
        self.phi0 = phi0
        self.W0 = W0/1.e5
        self.stable = stable

        # Set the depth scale h:
        self.zmin = df['P'].min()/dPdz
        self.zmax = df['P'].max()/dPdz
        self.h = self.zmax - self.zmin

        # Lambda function to define scaled column height zprime:
        self.zp = lambda P: (self.zmax - P/dPdz)/self.h

        # Set interpolants for F, Kr, and P:
        self.F = pchip(self.zp(df['P']),df['F'])
        self.dFdz = self.F.derivative(nu=1)
        self.Kr = interp1d(self.zp(df['P']),df['Kr'],kind='cubic')
        self.P = interp1d(self.zp(df['P']),df['P'],kind='cubic')

        # Set maximum degree of melting:
        self.Fmax = self.df['F'].max()

        # Set reference densities (assuming a mantle composition):
        self.rho_s = 3300.
        self.rho_f = 2800.

        # Set  decay constants for [ 238U, 230Th, 226Ra] and [ 235U, 231Pa ]:
        t_half_238 = np.array([4.468e9, 7.54e4, 1600.])
        t_half_235 = np.array([7.03e8, 3.276e4])
        self.lambdas_238 = np.log(2.)/t_half_238
        self.lambdas_235 = np.log(2.)/t_half_235

        # Set interpolation functions for partition coefficients for each decay chain:
        self.D_238 = [ interp1d(self.zp(df['P']),df['DU'],kind='cubic'),
                       interp1d(self.zp(df['P']),df['DTh'],kind='cubic'),
                       interp1d(self.zp(df['P']),df['DRa'],kind='cubic') ]
        self.D_235 = [ interp1d(self.zp(df['P']),df['DU'],kind='cubic'),
                       interp1d(self.zp(df['P']),df['DPa'],kind='cubic')]

        # Lambda function to retrieve partition coefficients at zprime = 0:
        self.get_D0 = lambda D: np.array([ D[i](0) for i in range(len(D))])

        # Initialize reference permeability:
        self.setAd(self.phi0,n=n)

    # Initialize the porosity function:
    def setAd(self,phi0,n):
        '''
        Sets the reference permeability given the maximum porosity
        '''
        Fmax = self.Fmax
        self.phi0 = phi0
        self.n = n
        self.Ad =  (self.rho_s/self.rho_f*Fmax - phi0*(1. - Fmax)/(1. - phi0))/(phi0**n*(1.-phi0))

    def phi(self,zp):
        '''
        Returns porosity as function of dimensionless column height zp
        '''
        # Effective permeability:
        K = self.Kr(zp)*self.Ad

        # Degree of melting:
        F = self.F(zp)

        # Density ratio:
        rs_rf = self.rho_s/self.rho_f

        # Rootfinding function to define porosity such that f(phi) = 0
        # Check if scalar else loop:
        if np.isscalar(zp):
            f = lambda phi: K*phi**self.n*(1. - phi)**2 + phi*(1. + F*(rs_rf - 1.)) - F*rs_rf
            upper_bracket= 1.05*self.phi0
            try:
                phi = brentq(f,0.,upper_bracket)
            except ValueError:
                phi_test = np.linspace(0,upper_bracket)
                print('Error in brentq: brackets={}, {}'.format(f(0.),f(upper_bracket)))
                print('zp={},F={}, K={}'.format(zp,F,K))
        else: # loop over length of zp
            phi = np.zeros(zp.shape)
            for i,z in enumerate(zp):
                f = lambda phi: K[i]*phi**self.n*(1. - phi)**2 + phi*(1. + F[i]*(rs_rf - 1.)) - F[i]*rs_rf
                phi[i] = brentq(f,0.,1.)
        return phi

    def set_column_params(self, phi0, n, W0):
        '''Set porosity/permeability and upwelling rate parameters for a single column

        Parameters:
        -----------
            phi0: float
                reference porosity at Fmax
            n   :  float
                permeability exponent
            W0  :  float
                upwelling rate (cm/yr)
        '''

        self.setAd(phi0,n)
        self.W0 = W0/1.e5  # upwelling in km/yr.

    def solve_1D(self,D,lambdas,alpha0 = None, z_eval = None):
        ''' Solves 1-D decay problem (assumes column parameters have been set)

        Usage:  z, a, Us, Uf = solve_1D(D,lambdas,alpha0,z_eval)

        Parameters:
        -----------
            D      :  Function that returns bulk partition coefficients for each nuclide
            lambdas:  Decay constants for each nuclide
            alpha0 :  Initial activities of nuclides in the unmelted solid (defaults to 1)
            z_eval :  Dimensionless column heights where solution is returned

        Returns:
        ---------
            z:   numpy array -- Coordinates where evaluated
            a:   numpy array -- Activities of each nuclide
            Us:  numpy array -- Solid nuclide log concentration
            Uf:  numpy array -- Fluid nuclide log concentration
        '''

        # If z_eval is not set, use initial input values:
        if z_eval is None:
            z_eval = self.zp(self.df['P'])
        elif np.isscalar(z_eval):
            z_eval = np.array([z_eval])

        # If alpha is not set, use 1:
        if alpha0 is None:
            alpha0 = np.ones(len(lambdas))

        # Scaled decay constants and initial partition coefficients:
        lambdap = self.h*lambdas/self.W0
        if self.stable:
            lambdap *= 0.

        us = self.model(alpha0,lambdap,D,self.W0,self.F,self.dFdz,self.phi,self.rho_f,self.rho_s,method=self.method,Da=self.Da)

        D0 = self.get_D0(D)
        z, Us, Uf = us.solve(z_eval)

        # Calculate activities:
        act =  [ alpha0[i]/D0[i]*np.exp(Uf[i]) for i in range(len(D0)) ]
        return z, act, Us, Uf

    def solve_all_1D(self,phi0,n,W0,alphas = np.ones(4),z_eval = None):
        ''' Sets up and solves a  1-D column model for both the U-238 and the U-235 decay chains.
        
        Parameters:
        -----------
            phi0: float
                reference maximum porosity for column
            n: float
                permeability exponent
            W0: float
                upwelling rate (cm/yr)
                

        Returns:
        --------
            df:  pandas dataframe containing physical and chemical parameters
        '''
        self.set_column_params(phi0,n,W0)

        # Evaluate at input depths if not specified:
        if z_eval is None:
            z_eval = self.zp(self.df['P'])

        # Set initial alpha values for [ 238U, 230Th, 226Ra] and [ 235U, 231Pa ]:
        self.alphas_238 = alphas[0:3]
        self.alphas_235 = alphas[3:5]

        # Solve for the U-238 decay series:
        z238, a238, Us238, Uf238 = self.solve_1D(self.D_238,self.lambdas_238,self.alphas_238,z_eval = z_eval)

        # Solve for the U-235 decay series:
        z235, a235, Us235, Uf235 = self.solve_1D(self.D_235,self.lambdas_235,self.alphas_235,z_eval = z_eval)

        # Start building output dataframe:
        z = z_eval

        df = pd.DataFrame()
        df['P'] = self.P(z)
        df['depth'] = self.zmax - self.h*z
        df['F'] = self.F(z)
        df['phi'] = self.phi(z)
        names = ['(230Th/238U)','(226Ra/230Th)']
        for i,name in enumerate(names):
            df[name] = a238[i+1]/a238[i]

        df['(231Pa/235U)'] = a235[1]/a235[0]

        names = ['Uf_238U','Uf_230Th', 'Uf_226Ra']
        for i,name in enumerate(names):
            df[name] = Uf238[i]

        names = ['Us_238U','Us_230Th', 'Us_226Ra']
        for i,name in enumerate(names):
            df[name] = Us238[i]

        names = ['Uf_235U','Uf_231Pa']
        for i,name in enumerate(names):
            df[name] = Uf235[i]

        names = ['Us_235U','Us_231Pa']
        for i,name in enumerate(names):
            df[name] = Us235[i]

        return df

    def solve_grid(self,phi0,n,W0,D,lambdas,alpha0 = None,z = 1.):
        ''' Solves for concentrations and activity ratios at the height z in the column for a 
            grid range of porosities phi0 and upwelling velocities W0.
            
        Parameters:
        -----------
            phi0: array
                1-d array of reference porosities
            n: float
                permeability exponent
            W0: array
                1-d array of solid upwelling velocities (cm/yr)
            D: calleable Function
                D(z) returns array of partition coefficients for all nuclides as a function of height
            lambdas: array
                decay constants for each nuclide ( 1/yrs )
            alpha0: array
                initial activities for each nuclide
                if None:  alpha0 = numpy.ones(N)
            z: float
                height in the column to report (defaults to the top, z=1)
                
        Returns:
            act:  ndarray(N-1,len(W0),len(phi0))
                3-D numpy array of activity ratios for a N nuclide chain
                

        '''
        # Number of nuclides in decay chain:
        Nchain = len(lambdas)

        # If alpha is not set, use 1
        if alpha0 is None:
            alpha0 = np.ones(Nchain)

        act = np.zeros((Nchain - 1,len(W0),len(phi0)))

        for j, W in enumerate(W0):
            print('\nW = {}'.format(W), end=" ")
            for i, phi in enumerate(phi0):
                print('.', end=" ")
                self.set_column_params(phi,n,W)
                z, a, Us, Uf = self.solve_1D(D,lambdas,alpha0,z_eval = z)
                for k in range(1,Nchain):
                    act[k-1,j,i] = a[k]/a[k-1]

        return act


### Utility plotting functions for plotting output of models:
def plot_inputs(df,figsize=(8,6),show_grids=True):
    ''' pretty plots input data from pandas dataframe df
    
    Parameters:
    -----------
        df: pandas dataframe
            dataframe of input parameters ([P, K_r, DU, DTh, DRa, DPa])
        figsize: tuple
            size of figure
        show_grids: bool
            if true, plot grids on all figures  
    
    Returns:
    --------
        fig:  matplotlib figure object
        axes: tuple of matplotlib axes objects for replotting 
    
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True,figsize=figsize)
    ax1.plot(df['F']*100,df['P'])
    ax1.invert_yaxis()
    ax1.set_xlabel('F (%)')
    ax1.set_ylabel('Pressure (kbar)')
    xticks = np.linspace(0,max(df['F']),10)
    if show_grids:
        ax1.grid()
    ax1.set_xticks(xticks,minor=True)
    ax2.plot(df['Kr'],df['P'])
    ax2.set_xlabel(r'$K_r$')
    if show_grids:
        ax2.grid()
    for s in ['DU','DTh','DRa','DPa']:
        ax3.semilogx(df[s],df['P'],label=s)
    ax3.set_xlabel(r'$D_i$')
    if show_grids:
        ax3.grid()
    ax3.legend(loc='best',bbox_to_anchor=(1.1,1))
    return fig, (ax1, ax2, ax3)

def plot_1Dcolumn(df,figsize=(8,6),show_grids=False):
    ''' pretty plots output data from dataframe of output
        
    Parameters:
    -----------
        df: pandas dataframe
            dataframe of model output
        figsize: tuple
            size of figure
        show_grids: bool
            if true, plot grids on all figures
            if false, just plot grid on activities plot
    
    Returns:
    --------
        fig:  matplotlib figure object
        axes: tuple of matplotlib axes objects for replotting 
    '''

    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=figsize)
    ax1.plot(df['phi']*100,df['P'],'r',label='$\phi$')
    ax1.set_xlabel('Porosity (%)',color='r')
    ax1.set_ylabel('Pressure (kbar)')
    ax1.invert_yaxis()
    if show_grids:
        ax1.grid(color='r')

    ax1a = ax1.twiny()
    ax1a.plot(df['F']*100,df['P'],'b',label='$F$')
    ax1a.set_xlabel('Degree of melting (%)',color='b')
    if show_grids:
        ax1a.grid(color='b')

    for s in ['(230Th/238U)','(226Ra/230Th)','(231Pa/235U)']:
        ax2.plot(df[s],df['P'],label=s)
    ax2.set_xlabel('Activity Ratios')
    ax2.set_xlim(0,5)
    ax2.set_xticks(range(5))
    ax2.grid()
    ax2.legend(loc='best',bbox_to_anchor=(1.1,1))
    return fig,(ax1,ax1a,ax2)

def plot_contours(phi0,W0,act,figsize=(8,10)):
    '''
    pretty plot activity contour plots
    '''

    Nplots = act.shape[0]
    if Nplots == 2:
        labels = ['$(^{230}Th/^{238}U)$', '$(^{226}Ra/^{230}Th)$']
    else:
        labels = ['$(^{231}Pa/^{235}U)$']

    if Nplots == 1:
        fig =  plt.figure(figsize=figsize)
        ax =  fig.add_subplot(1,1,1)
        cf = plt.contourf(phi0*100,W0,act[0])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Porosity (%)')
        ax.set_ylabel('Upwelling Rate (cm/yr)')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(labels[0])
        fig.colorbar(cf, ax=ax, orientation='horizontal',shrink=1.)
    else:
        fig, axes = plt.subplots(1,Nplots,sharey=True,figsize=(2*figsize[0],figsize[1]))
        for i,ax in enumerate(axes):
            cf = ax.contourf(phi0*100,W0,act[i])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_aspect('auto')
            ax.set_xlabel('Porosity (%)')
            ax.set_ylabel('Upwelling Rate (cm/yr)')
            ax.xaxis.set_major_locator(mpl.ticker.LogLocator())
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_title(labels[i])
            fig.colorbar(cf, ax=ax, orientation='horizontal',shrink=1.)
            
    

def plot_mesh_Ra(Th,Ra,W0,phi0,figsize=(10,12)):
    '''
    activity mesh plot for Ra vs. Th, from gridded data
    '''
    mW,nphi=Th.shape
    plt.figure()
    for m in range(mW):
        plt.plot(Th[m],Ra[m],label='W = {} cm/yr'.format(W0[m]))
    for n in range(nphi):
        plt.plot(Th.T[n],Ra.T[n],'--',label='$\phi$ = {}'.format(phi0[n]))

    plt.legend(loc='upper left',bbox_to_anchor=(0,-0.3))
    plt.axis([0.9,2.0,0.7,10.0])
    plt.grid()
    plt.xlabel('($^{230}$Th/$^{238}$U)')
    plt.ylabel('($^{226}$Ra/$^{230}$Th)')
#    plt.axes().set_aspect('equal',)

def plot_mesh_Pa(Th,Pa,W0,phi0,figsize=(12,12)):
    '''
    activity mesh plot for Pa vs. Th, from gridded data
    '''
    mW,nphi = Th.shape
    plt.figure()
    for m in range(mW):
        plt.plot(Th[m],Pa[m],label='W = {} cm/yr'.format(W0[m]))
    for n in range(nphi):
        plt.plot(Th.T[n],Pa.T[n],'--',label='$\phi$ = {}'.format(phi0[n]))

    plt.legend(loc='upper left',bbox_to_anchor=(0,-0.3))
    plt.axis([0.9,2.0,0.7,10.0])
    plt.grid()
    plt.xlabel('($^{230}$Th/$^{238}$U)')
    plt.ylabel('($^{231}$Pa/$^{230}$Th)')

    #plt.show()

#---------------------------------------
# Utility functions for calculating smoothed steps and ramps and the two-layer input generator

from collections import OrderedDict

def _step(x, x_0, x_lambda, f):
    """ function to return smoothed step function between values f_min and f_max
    
        f_min + (f_max - f_min)*H((x-x0)/x_lambda)
        
        where H(x) is a smoothed Heaviside function 
        
        Parameters:
        -----------
        
        x: array
            numpy array of locations x
        x_0: float
            location of step
        x_lambda: float
            smoothing lengthscale
        f:  array [f_min f_max]
            critical values in step function
            f_min value of step at x[0]
            f_max value of step at x[1]
            
    """
    return f[0] + (f[1] - f[0])*0.5*(np.tanh((x - x_0)/x_lambda) + 1)

def _ramp(x, x_0, x_lambda, f ):
    """ function to return smoothed broken ramp function  between values f_min, f_0 and f_max
    
        This function is just the integral of step(x, df1, df2, x_0, x_lambda) with values given by the slopes between 
        
            df1 = (f_0 - f_min)/(x_0 - x_min)
            df2 = (f_max - f_0)/(x_max - x_0)
        
        Parameters:
        -----------
        
        x: array
            numpy array of locations x
        x_0: float
            location of step
        x_lambda: float
            smoothing lengthscale
        f: array [f_min f_0 f_max ]
            critical values in ramp function
            f_min value of ramp at x[0]
            f_0 value ramp at x_0
            f_max value of ramp at x[-1]
        
            
    """
    # set slopes of the two segments of the ramp far from the inflection point
    df1 = (f[1] - f[0])/(x_0 - x[0])
    df2 = (f[-1] - f[1])/(x[-1] - x_0)
    
    # calculate the scaled difference from the inflection point
    x_scaled = (x-x_0)/x_lambda
    
    # find all points > xscaled = -10
    indx = x_scaled >= -10
    istart =  np.where(indx)[0][0]
    
    # smoothed heaviside function
    H = (np.tanh(x_scaled) + 1.)/2.
    
    # initialized bottom part of ramp
    ramp = f[0] + df1*x
    # calculate integral of step function for x_scaled > -10
    intdf = df1*x[indx] + (df2 - df1)*(x[indx] - 0.5*x_lambda*np.log(H[indx])) 
    # patch up ramp function
    ramp[indx]  = ramp[istart] - intdf[0] + intdf
    return ramp 

def twolayermodel(P, F, Kr, D_lower, D_upper,  N=100, P_lambda = 1,):
    """ create a pandas dataframe for the UserCalc sample two-layer model defined by a column that spans pressures from
        P_bottom to P_top with a possible change in layer properties at pressure P_boundary that is smoothed over a pressure range given by P_lambda
        
        Each layer can have its own bulk partition coefficient D, relative permeability Kr. 
        
        The degree of melting is given by a smoothed ramp function defined by three values of F, F(P_bottom), F(P_boundary) and F(P_top)
        
        Parameters:
        -----------
        
        P: array [ P_bottom, P_boundary, P_top ]
            pressure bounds in the column (assumed kb) 
            P[0] = P_bottom is the pressure at the bottom of the column
            P[1] = P_boundary is the pressure at the layer boundaries
            P[2] = P_top is the pressure at the top of the column
        F: array [ F_bottom, F_boundary, F_top]
            Degree of melting at the bottom, layer boundary and top of the column
        D_lower:  array [ D_U, D_Th, D_Ra, D_Pa ]
            bulk partition coefficients for U-series nuclides in the bottom layer
        D_upper:  array [ D_U, D_Th, D_Ra, D_Pa ]
            bulk partition coefficients for U-series nuclides in the upper layer
        Kr: array [ Kr_lower,  Kr_upper]
            relative permeability of lower and upper layers (controls step function)
        N: int
            Number of rows in dataframe
        P_lambda: float
            Pressure smoothing parameter.  Controls the width of  Smooth steps and ramps between layers (defaults to 1kb)
            
        Returns:
        --------
        
        df: pandas dataframe
            with columns P, F, Kr, DU, DTh, DRa, DPa
        
    """
    # basic check on array lengths
    assert(len(P) == 3)
    assert(len(F) == 3)
    assert(len(Kr) == 2)
    assert(len(D_lower) == 4)
    assert(len(D_lower) == len(D_upper))    
    
    # construct pressure array and reordered pressure array
    P_array = np.linspace(P[0], P[-1], N)
    x = P_array[0] - P_array
    x_0 = P_array[0] - P[1]
    
    Ds = list(zip(D_lower, D_upper))
    D_labels = [ 'DU', 'DTh', 'DRa', 'DPa' ]
        
    _dict = OrderedDict()
    _dict['P'] = P_array
    _dict['F'] = _ramp(x, x_0, P_lambda, F)
    _dict['Kr'] = _step(x, x_0, P_lambda, Kr)
    for i,D in enumerate(Ds):
        _dict[D_labels[i]] = _step(x, x_0, P_lambda, D)
        
    return pd.DataFrame.from_dict(_dict)
        