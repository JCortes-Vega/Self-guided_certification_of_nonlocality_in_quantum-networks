import numpy as np

class Results:
    def __init__(self) -> None:
        pass

class CSPSA:

    def __init__(self, gains, minimize=True ) -> None:
        self.gains = gains 
        self._sign = minimize

    def minimize( self, 
                    obj_fun, 
                    x_in, 
                    max_iter, 
                    args = (),
                    postprocessing = None, 
                    output_function = None ):
        """
        minimizing f(x)
        obj_fun  --> f
        x_in     --> initial condition
        max_iter --> maximum number of iterations
        args     --> tuple with other arguments directly passed to the objective function
        """
        
        if not isinstance(args, tuple):
            args = (args,)

        results = Results()
        results.x = []
        results.fun = []

        self.x_in = x_in
        dims = x_in.shape 

        for i in range( max_iter ):
            # k: iteraciones. psi: estado
            # a = G[0], A = G[1], b = G[2], s = G[3], t = G[4]

            alpha = self.gains[0]/(i+1+self.gains[1])**self.gains[3]
            beta  = self.gains[2]/(i+1)**self.gains[4]

            delta = (1j)**(np.random.randint(1, 5, dims))
            x_plus  = x_in + (beta*delta)
            x_minus = x_in - (beta*delta)
            
            f_plus  = np.real( obj_fun( x_plus, *args ) )
            f_minus = np.real( obj_fun( x_minus, *args ) )
            
            grad = np.divide( f_plus - f_minus, 2*beta*delta.conj() ) 
            
            x_in = x_in + (-1)**self._sign * alpha*grad

            results.x.append( x_in )

            if postprocessing is not None:
                x_in = postprocessing( x_in )
            
            if output_function is None:
                results.fun.append( obj_fun( x_in, *args ) )
            else:
                results.fun.append( output_function( x_in ) )

        return results 


    def _minimizev2( self, 
                 st,
                 obj_fun, 
                 x_in, 
                 max_iter, 
                 postprocessing=None, 
                 output_function=None ):
        """
        minimizing f(x)
        obj_fun  --> f
        x_in     --> initial condition
        max_iter --> maximum number of iterations
        """

        results = Results()
        results.x = []
        results.fun = []

        self.x_in = x_in
        dims = x_in.shape 

        for i in range( max_iter ):
            # k: iteraciones. psi: estado
            # a = G[0], A = G[1], b = G[2], s = G[3], t = G[4]

            alpha = self.gains[0]/(i+1+self.gains[1])**self.gains[3]
            beta  = self.gains[2]/(i+1)**self.gains[4]

            delta = (1j)**(np.random.randint(1, 5, dims))
            x_plus  = x_in + (beta*delta)
            x_minus = x_in - (beta*delta)
            
            f_plus  = np.real( obj_fun( x_plus , st ) )
            f_minus = np.real( obj_fun( x_minus , st ) )
            
            grad = np.divide( f_plus - f_minus, 2*beta*delta.conj() ) 
            
            x_in = x_in + (-1)**self._sign * alpha*grad

            results.x.append( x_in )

            if postprocessing is not None:
                x_in = postprocessing( x_in )
            
            if output_function is None:
                results.fun.append( obj_fun( x_in , st) )
            else:
                results.fun.append( output_function( x_in , st) )

        return results