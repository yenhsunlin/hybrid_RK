# Created by Yen-Hsun Lin (Academia Sinica) in 12/2024.
# Copyright (c) 2024 Yen-Hsun Lin.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.



from numpy import array,asarray,abs,sqrt,finfo,minimum,maximum,concatenate
from numpy import max as np_max
from numpy import min as np_min
from numpy.linalg import eigvals,norm
from scipy.optimize import approx_fprime,root


#------------------------------------------------------------#
#                                                            #
#      Solver Classes for Explicit/Implicit RK45 Method      #
#                                                            #
#------------------------------------------------------------#

# * --- Explicit class --- * #
class SolverExplicit45:
    """
    This class implements explicit/implicit RK45 with adaptive step size
    method for solving ODEs.

    Usage
    ------
    Given a system ODEs 

         dy1/dt = f1(t,y1,y2,...,yn)
         dy2/dt = f2(t,y1,y2,...,yn)
               ...
         dyn/dt = fn(t,y1,y2,...,yn)

    One can compile them into a single function that takes 2 arguments where
    the first argument is time and the second is a list of [y1,y2,...,yn].
    For instance

    >>> def f(t,y):
            y1,y2,...,yn = y
            dy1 = ...
            dy2 = ...
            ...
            dyn = ...
            return [dy1,dy2,...,dyn]

    with initial condition t0 and y0 = [y10,y20,...yn0]. Once the relative
    tolerance and absolute tolerance are specified, we can initialize the
    solver instance

    >>> solver = SolverExplicit45(f,t0,y0,rtol,atol)

    One can determine the initial step size by

    >>> dt0 = solver.dt0

    To evaluate the solution for the next step given (t,y,dt) where they are
    time for the next step, solution from the last step and step size for the
    next step. One excute

    >>> y_new,dt_new,success_flag = solver.RK45_solver(t,y,dt)

    `y_new` is the solution, `dt_new` is the new step size that can be applied to
    the next iteration and `success_flag` is a bool that characterizes the solution
    in this step can be accepted or not.

    If `success_flag` is True, then the solution `y_new` can be recorded and apply
    it and `dt_new` to the next round of iteration. If otherwise, the solution has
    error larger than tolerance and must be depricated. One can use the adjusted
    `dt_new` with `y` from last step to re-iterate again until `success_flag`
    becomes True.
    """

    def __init__(self,f,t0,y0,rtol,atol):
        self.f_raw = f
        self.t0 = t0
        self.y0 = asarray(y0)
        self.rtol = rtol
        self.atol = atol

    def local_error(self,y4,y5):
        """
        Evaluate the error from 4th and 5th-order solutions

        u4: the 4th-order solution
        u5: the 5th-order solution

        The local_error will be truncated at machine precision if it is
        too small
        """
        # Define machine precision
        epsilon = sqrt(finfo(float).eps)
        # If f is coupled ODEs, y5 & y4 are arrays. We thus take Euclidean norm
        error = norm(y5 - y4,ord=2)
        # If error is smaller than machine precision, using epsilon instead
        # to avoid divergence when update step size
        effective_error = maximum(error,epsilon)
        return effective_error

    def tolerance(self,y):
        """
        The tolerance follows solve_ivp in scipy
        """
        # If f is coupled ODEs, y is an array. We thus take Euclidean norm
        norm_y = norm(y,ord=2)
        return self.atol + self.rtol*abs(norm_y)
    
    def update_step_size(self,dt,tol,err,p):
        """
        Update the step size
        """
        # Pick up the minimum new dt if it contains multiple values because tol and err are arrays
        dt_new = dt*(tol/err)**(1/(p+1))*0.8
        # The adjustment avoids the new step size grows/shrinks too fast. We follow solve_ivp
        dt_new = minimum(2 * dt, maximum(0.1 * dt, dt_new))
        return dt_new

    def is_divergent(self,dt,tol):
        """
        If continues shrinking step size at a particular point, it
        generally implies the solution cannot be found. The problem
        could be divergent at that point.
        """
        # machine precision
        epsilon = finfo(float).eps
        # minimum step size that can be tolerated
        dt_minimum = 10*epsilon
        # check if dt is too small
        if abs(dt) < dt_minimum:
            return True
        else:
            return False
    
    def f(self,t,y):
        """
        This is a warapper that forces the callable input f to have an
        numpy.array output.
        """
        return array(self.f_raw(t, y))
    
    def explicit_y4y5(self,t,y,dt):
        """
        High-order solutions y4 and y5
        """
        y = asarray(y)
        # k coefficients
        k1 = self.f(t, y)
        k2 = self.f(t + dt/4, (y + dt/4*k1))
        k3 = self.f(t + 3*dt/8, (y + 3*dt/32*k1 + 9*dt/32*k2))
        k4 = self.f(t + 12*dt/13, (y + 1932*dt/2197*k1 - 7200*dt/2197*k2 + 7296*dt/2197*k3))
        k5 = self.f(t + dt, (y + 439*dt/216*k1 - 8*dt*k2 + 3680*dt/513*k3 - 845*dt/4104*k4))
        k6 = self.f(t + dt/2, (y - 8*dt/27*k1 + 2*dt*k2 - 3544*dt/2565*k3 + 1859*dt/4104*k4 - 11*dt/40*k5))

        # 4th and 5th -order solutions
        y4 = y + dt*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
        y5 = y + dt* (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
        return y4,y5
    
    def explicit_RK45_solver(self,t,y,dt):
        """
        Evaluate the next solution given the current (t,y) and dt
        """
        # Get high-order solutions
        y4,y5 = self.explicit_y4y5(t,y,dt)
        
        # Evaluate local error and tolerance
        error = self.local_error(y4,y5)
        tol = self.tolerance(y)
        
        # Adjust step size
        dt_new = self.update_step_size(dt,tol,error,p=4)
        
        # Check divergence
        is_divergent = self.is_divergent(dt_new,tol)
        
        if error <= tol:
            # Accept solution and update step size
            return y5, dt_new, True, is_divergent # solution, new step size, successfully found y5, is divergent
        else:
            # Reject solution and update step size 
            return y, dt_new, False, is_divergent # solution, new step size, successfully found y5, is divergent

    @property
    def dt0(self):
        """
        Initial guess of the beginning step size
        """
        tol = self.tolerance(self.y0)
        norm_f0 = norm(self.f(self.t0,self.y0),ord=2) 
        epsilon = sqrt(finfo(float).eps) # machine precision
        dt0 = 0.1*tol**2/maximum(norm_f0,epsilon)
        return minimum(1.0, sqrt(dt0))


# * --- Implicit class --- * #
class SolverImplicit45(SolverExplicit45):
    
    def __init__(self,f,t0,y0,rtol,atol):
        # Initialize Implicit Solver by its parent class SolverExplicit45
        super().__init__(f,t0,y0,rtol,atol)
    
    def k_initial_guess(self,t,y,dt):
        """
        Initial guess on coefficient k by explicit method
        """
        k1 = self.f(t, y)
        k2 = self.f(t + dt / 2, y + dt / 2 * k1)
        k3 = self.f(t + dt / 2, y + dt / 2 * k2)
        k4 = self.f(t + dt, y + dt * k3)
        return k1, k2, k3, k4

    def solve_k(self,t,y,dt):
        """
        Solve k coefficients with "hybr" method in scipy root
        """
        # initial guess k
        _,k2_ini,k3_ini,k4_ini = self.k_initial_guess(t,y,dt)
        # concatenate as a 1D array
        k_ini = concatenate([k2_ini,k3_ini,k4_ini])
        
        # solve ki
        k1 = self.f(t+1/4*dt,y)

        def k_eqs(k):
            n = len(k)
            # extract k
            k2,k3,k4 = k.reshape(3,n//3)
            
            # get different ki
            k2eq = self.f(t + 3/8*dt, y+dt*3/32*k1) - k2
            k3eq = self.f(t + 12/13*dt, y+dt*(1932/2197*k1-7200/2197*k2)) - k3
            k4eq = self.f(t+dt,y+dt*(439/216*k1-8*k2+3680/513*k3)) - k4
            
            # concatenate them and output as a (3n,) array
            return concatenate([k2eq, k3eq, k4eq])
        
        k_sol = root(k_eqs, k_ini, method='hybr')
        n = len(k_sol.x)
        k2,k3,k4 = k_sol.x.reshape(3,n//3)
        return k1,k2,k3,k4
        
    def implicit_y4y5(self,t,y,dt):
        # Solve for k coefficients
        k1,_,k3,k4 = self.solve_k(t, y, dt)
        # higher-order solutions
        y5 = y + dt*(16/135*k1+6656/12825*k3+28561/56430*k4)
        y4 = y + dt*(25/216*k1+1408/2565*k3+2197/4104*k4)
        return y4,y5

    def implicit_RK45_solver(self,t,y,dt):
        """
        Evaluate the next solution given the current (t,y) and dt
        """
        y = asarray(y)
        # Get high-order solutions
        y4,y5 = self.implicit_y4y5(t,y,dt)
        
        # Evaluate local error and tolerance
        error = self.local_error(y4,y5)
        tol = self.tolerance(y)

        # Adjust step size
        dt_new = self.update_step_size(dt,tol,error,p=4)
        
        # Check divergence
        is_divergent = self.is_divergent(dt_new,tol)
        
        if error <= tol:
            # Accept solution and update step size
            return y5, dt_new, True, is_divergent # solution, new step size, successfully found y5, is divergent
        else:
            # Reject solution and update step size 
            return y, dt_new, False, is_divergent # solution, new step size, successfully found y5, is divergent


#------------------------------------------------------------#
#                                                            #
#                     Utility Functions                      #
#                                                            #
#------------------------------------------------------------#

def stiffness_ratio(f,t,y,dt=None,show_jac=False):
    """
    Estimating the stiffness ratio R

    In
    ---
    f: Callable, f(t,y)
    t: The point where system Jacobian to be derived
    y: Value(s) of the ODE at t
    dt: Time step, default None
    show_jac: Bool, return Jacobian matrix
    
    Out
    ---
    R: Stiffness ratio
    """
    # Check the numbers of ODEs
    n = len(y)
    
    # The infinitesimal difference to calculate the derivative
    if dt is None:
        # Apply machine precision
        dt = sqrt(finfo(float).eps)
    elif isinstance(var, (int, float)):
        # User-defined dt
        pass 
    else:
        # Illegat dt input
        raise ValueError("\'dt\' must be a number.")
        
    # Evaluate the Jacobian for the system ODE
    jac = approx_fprime(y, lambda y: f(t, y), dt)
    # Evaluate eigenvalues and stiffness ratio
    if n == 1:
        # jac is a single value.
        # Suffix .item() converts a (1,) or (1,1) array into a scalar
        R = abs(jac).item()
    else:
        # Eigenvalues can be evaluated if jac is a squared matrix
        abs_eigvals = abs(eigvals(jac))
        R = np_max(abs_eigvals)/np_min(abs_eigvals)

    if show_jac is True:
        return R, jac
    elif show_jac is False:
        return R
    else:
        raise ValueError("\'show_jac\' must be a bool.")


def _tidy_solution(t,y):
    """
    Tidy up solutions into numpy array

    In
    ---
    t: array/list like, the corresponding time stamp
    y: array/list like, the corresponding solutions to ODE
    
    Out
    ---
    t: numpy array, shape of (1,)
    y: numpy array, shape of (m,n). `m` is the number of steps
        applied in this solution set until the algorithm reaches
        termination. `n` is the number of ODEs in this system.
        If n =1, then the output array will have shape (m,). 
    """
    t,y = asarray(t),asarray(y)
    # Check shapes of y
    _,n = y.shape
    # Tidy up solution, particular for n = 1
    if n == 1:
        y = y.ravel()
    else:
        y = y.T
    return t,y


# Main function
def solve_ode(f,trange,y0,rtol=1e-3,atol=1e-6,method="explicit",R_threshold=1000,args=None):
    """
    Solving initial value problem with given system ODEs by implicit RK45

    In
    ---
    f: Callable, f(t,y)
    trange: Range of steps, 1D array of shape (m,) where m specifies
        all time steps that the solutions will be evaluated
    y0: Initial value(s), 1D array
        Even there's only one ODE, y0 should be an array of shape (1,)
        eg. y0 = [1]
    rtol: Relative tolerance. Default is 1e-3.
    atol: Absolute tolerance. Default is 1e-6.
    method: String, "explicit" (default), "implicit" or "hybrid"
        "hybrid" will make the solver shifts between "explicit" and
        "implicit" methods given the stiffness ratio
    R_threshold: Float, the threshold that the solver should switch
        from explicit method to implicit method.
    args: Tuple, it will pass to the callable function by f(t,y,*args)
    
    Out
    ---
    sol_t: arr-like, solution t
    sol_y: arr-like, Solution y
    """
    # Identify the beginning and the end of time steps
    t0,tmax = trange[0],trange[-1]
    if tmax < t0:
        sign = -1
    else:
        sign = 1

    # Deal with extra arguments
    if args is None:
        f_rhs = lambda t,y: f(t,y)
    else:
        # pass extra args to f_rhs 
        f_rhs = lambda t,y: f(t,y,*args)
    
    # Initializing the solver instance
    # Note that implicit solver already inherits all methods in explicti solver
    solver = SolverImplicit45(f_rhs,t0=t0,y0=y0,rtol=rtol,atol=atol)
    
    # Determine the initial setup and setup the placeholders for storing solutions
    # and time steps
    t,y = t0,y0
    solution_t,solution_y = [t0], [y0]
    dt = sign*solver.dt0
    R = stiffness_ratio(f_rhs,t0,y0)
    
    # Starting loop until reaching tmax
    while t < tmax:
        # make sure that dt will not exceed tmax
        dt = min(dt, tmax - solution_t[-1])
        
        # determine which method should be used
        if method == "explicit":
            y_new, dt_new, is_successful, is_divergent = solver.explicit_RK45_solver(t, y, dt)
        elif method == "implicit":
            y_new, dt_new, is_successful, is_divergent = solver.implicit_RK45_solver(t, y, dt)
        elif method == "hybrid":
            # the solver will switch between explicit and implicit methods judging by stiffness ratio
            if R < R_threshold:
                # non-stiff
                y_new, dt_new, is_successful, is_divergent = solver.explicit_RK45_solver(t, y, dt)
            else:
                # stiff
                y_new, dt_new, is_successful, is_divergent = solver.implicit_RK45_solver(t, y, dt)
        else:
            raise ValueError("Argument \'method\' must be either \'explicit\', \'implicit\' or \'hybrid\'.")    
        
        if is_successful:
            # if the solver successfully converges
            t += dt
            y = y_new
            solution_y.append(y)
            solution_t.append(t)

            # Does evaluate R required? Only hybrid or implicit methods applied
            if method == "hybrid" or method == "implicit":
                R = stiffness_ratio(f_rhs,t,y)
                print("\rProgress: {:.2f}% and stiffness ratio R = {:.3e}.".format(t/tmax * 100,R), end="")
            else:
                print("\rProgress: {:.2f}%".format(t/tmax * 100), end="")
        if is_divergent:
            # if dt is too small to find the solution, the solution could be divergent
            # at this point. One should terminate the loop before error occur
            print(f"\rThe solver has terminated at t = {t} due to too small step size. Suspect a divergence.",end="")
            break
        # update step size
        dt = sign*dt_new
    
    # Tidy up the solution and make them more consise and transparent
    solution_t,solution_y = _tidy_solution(solution_t,solution_y)
    return solution_t,solution_y