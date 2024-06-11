import torch

from torchdiffeq import odeint_adjoint as odeint

from torch.autograd import Function
from torch.autograd.function import once_differentiable

import matplotlib.pyplot as plt


class RHS_geodesic_isotropic2D(torch.nn.Module):

    def __repr__(self):
        return 'Class inherinting from torch.nn.Module, forward function outputs the RHS f of the ODE dy/dt = f(t,y), y(0) = y0 in the case of the geodesic equation in 2D with metric potential(x)*torch.eye(2).'

    def __init__(self, potential):
        super().__init__()
        self.potential = potential # pass the potential function as attribute
     
    def forward(self,t,y):
        v = torch.stack([y[1] , -(y[1].unsqueeze(0).unsqueeze(0)*self.christoffels(y[0])*(y[1].unsqueeze(0).unsqueeze(2))).sum(dim=[1,2])]) #returns the right-hand side of the ODE for the geodesic equation defined by the potential
        return v
    
    def christoffels(self,x):
        Gamma = torch.zeros([2,2,2]) # Initialize Tensor of christoffel symbols

        with torch.enable_grad(): # Need grad to compute derivatives
            x_in = x.clone()
            if not x_in.requires_grad:
                x_in.requires_grad = True
            
            jacobian = torch.autograd.grad(
                outputs = .5*torch.log(self.potential(x_in)),
                inputs = x_in,
                create_graph = True,
                retain_graph = True,
                only_inputs = True,
                allow_unused = True)[0] # Christoffel symbols are expressed in terms of the derivatives of the log of the potential
            
            Gamma[0,0,0] = jacobian[0] # Build the tensor from the computed derivatives
            Gamma[0,0,1] = jacobian[1]
            Gamma[0,1,0] = jacobian[1]
            Gamma[0,1,1] = -jacobian[0]
            Gamma[1,0,0] = -jacobian[1]
            Gamma[1,0,1] = jacobian[0]
            Gamma[1,1,0] = jacobian[0]
            Gamma[1,1,1] = jacobian[1]
        return Gamma


# class ExponentialCurveModule(torch.nn.Module):
#     '''
#         Module wrapper for Exponential curves
    
#         Args:
#             control_points: Points defining t
        
#         Note : 
#     '''
    
#     def __repr__(self):
#         return "Polygonal curves"
    
#     def __init__(self, control_points, potential, discretization_times = None):
#         super().__init__()
#         self.first_point = control_points[0] #
#         self.tangent_vectors = control_points[1:]
#         if discretization_times==None:
#             self.discretization_times = torch.linspace(0,1,control_points.shape[0])
#         else:
#             self.discretization_times =  discretization_times

#         self.order = control_points.shape[0]
#         self.dim =  control_points.shape[1]

#         self.potential = potential
#         self.RHS = RHS_geodesic_isotropic2D(self.potential)

#         assert (self.order == self.discretization_times.shape[0])
        
#     def forward(self, timestamps):
#         '''
#         Evaluation of

#         Parameters
#         ----------
#         timestamps : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         TYPE
#             DESCRIPTION.

#         # '''
#         # assert sorted(list(timestamps)) == list(timestamps)
#         # for t in sorted(list(timestamps)):

        
#         # indices = torch.floor(timestamps*(self.control_points.shape[0]-1)).to(int)
#         # return ((1-(timestamps*(self.control_points.shape[0]-1)-indices)).squeeze(0).T*self.control_points[indices].squeeze() + (timestamps*(self.control_points.shape[0]-1)-indices).squeeze(0).T*self.control_points[indices+1].squeeze()).T

#         if timestamps==self.discretization_times:
#             list = [self.first_point]
#             for k in range(self.tangent_vectors.shape[0]):
#                 print(list)
#                 y0 = torch.stack([list[-1], self.tangent_vectors[k]])
#                 list.append(odeint(self.RHS,y0,t=torch.linspace(0,1,2))[-1][0])

#         else:
#             raise ValueError('not implemented yet')
        
#         return torch.stack(list)
    
#     def derivative(self,timestamps):
#         '''
#         Evaluate time derivative

#         Parameters
#         ----------
#         self.timestamps : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         '''
#         if timestamps == self.discretization_times:
#             return self.tangent_vectors/(self.discretization_times[1:] - self.discretization_times[:-1])
#         else:
#             raise ValueError('Not implemented yet')
        
#     def curves_for_plots(self, n_samples=16):
#         list = [self.first_point.unsqueeze(0)]
#         for k in range(self.tangent_vectors.shape[0]):
#             print(k)
#             y0 = torch.stack([list[-1][-1], self.tangent_vectors[k]])
#             list.append(odeint(self.RHS,y0,t=torch.linspace(0,1,n_samples))[:,0])

#         return torch.stack(list[1:])


class RHS_geodesic_RS_2D(torch.nn.Module):
    """
    Class inheriting from torch.nn.Module, representing the right-hand side (RHS) of the ODE dy/dt = f(t,y),
    where y(0) = y0, in the case of the geodesic equation in 2D with metric potential(x)*torch.eye(2).
    """

    def __init__(self, epsilon=1., xi=1.):
        super().__init__()
        self.epsilon = epsilon
        self.xi = xi
     
    def forward(self, t, y):
        """
        Computes the forward pass of the module.

        Args:
            t (torch.Tensor): The time variable.
            y (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing the RHS of the ODE for the geodesic equation.
        """
        v = torch.stack([y[:,1],
                          -(y[:,1].unsqueeze(1).unsqueeze(1)*self.christoffels(y[:,0])*(y[:,1].unsqueeze(1).unsqueeze(3))).sum(dim=[2,3])], dim=1)
        return v
    

    def christoffels(self, x):
        """
        Computes the Christoffel symbols for the geodesic equation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor representing the Christoffel symbols.
        """
        Gamma = torch.zeros([x.shape[0],3,3,3])

        eps = self.epsilon
        xi = self.xi
        z = x[:,2]

        Gamma[:,0,0,2] = -.5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,0,1,2] = -.5*(eps**4 - (eps**4 - 1)*torch.cos(z)**2 - eps**2)/eps**2 
        Gamma[:,1,0,2] = .5*((eps**4 - 1)*torch.cos(z)**2 - eps**2 + 1)/eps**2 
        Gamma[:,1,1,2] = .5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,2,0,0] = .5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 
        Gamma[:,2,0,1] = -.5*(2*(eps**2 - 1)*torch.cos(z)**2 - eps**2 + 1)/((eps**2)*(xi**2)) 
        Gamma[:,2,1,1] = -.5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 

        Gamma_sym = Gamma + Gamma.permute([0,1,3,2])
        return Gamma_sym
    
    def metric_matrix(self, x):
        """
        Computes the metric matrix for the geodesic equation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor representing the metric matrix.
        """
        epsilon = self.epsilon
        xi = self.xi

        g = torch.stack([torch.stack([torch.cos(x[...,2])**2+(1/epsilon**2)*torch.sin(x[...,2])**2,(1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.zeros(x.shape[0])]),
                        torch.stack([(1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.sin(x[...,2])**2+(1/epsilon**2)*torch.cos(x[...,2])**2,torch.zeros(x.shape[0])]),
                        torch.tensor([0,0,(xi)**2]).expand([x.shape[0],3]).T]) 
        
        return g.permute([2,0,1])
    
    def metric_inverse_matrix(self, x):
        """
        Computes the inverse of the metric matrix for the geodesic equation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor representing the inverse of the metric matrix.
        """
        epsilon = self.epsilon
        xi = self.xi
        g_inv = torch.stack([torch.stack([torch.cos(x[...,2])**2+(epsilon**2)*torch.sin(x[...,2])**2,(1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.zeros(x.shape[0])]),
                        torch.stack([(1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.sin(x[...,2])**2+(epsilon**2)*torch.cos(x[...,2])**2,torch.zeros(x.shape[0])]),
                        torch.tensor([0,0,(1/xi)**2]).expand([x.shape[0],3]).T]) 
        
        return g_inv.permute([2,0,1])

# class ExponentialCurveModule_RS(torch.nn.Module):
#     '''
#         Module wrapper for Exponential curves in Reeds-Shepp space
    
#         Args:
#             control_points: Points defining t
        
#         Note : 
#     '''
    
#     def __repr__(self):
#         return "Piecewise exponential curves for relaxed Reeds-Shepp space"
    
#     def __init__(self, n_points, epsilon = 1., xi = 1., discretization_times = None):
#         super().__init__()
#         self.first_point = torch.zeros([3], requires_grad=True) #
#         self.tangent_vectors = torch.randn([n_points-1,3], requires_grad=True)
#         if discretization_times==None:
#             self.discretization_times = torch.linspace(0,1,n_points)
#         else:
#             self.discretization_times =  discretization_times

#         self.order = n_points
#         self.dim =  3

#         self.epsilon = epsilon
#         self.xi = xi
#         self.RHS = RHS_geodesic_RS_2D(epsilon = self.epsilon, xi = self.xi)

#         assert (self.order == self.discretization_times.shape[0])
        
#     def forward(self, timestamps):
#         '''
#         Evaluation of

#         Parameters
#         ----------
#         timestamps : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         TYPE
#             DESCRIPTION.

#         # '''
#         # assert sorted(list(timestamps)) == list(timestamps)
#         # for t in sorted(list(timestamps)):

        
#         # indices = torch.floor(timestamps*(self.control_points.shape[0]-1)).to(int)
#         # return ((1-(timestamps*(self.control_points.shape[0]-1)-indices)).squeeze(0).T*self.control_points[indices].squeeze() + (timestamps*(self.control_points.shape[0]-1)-indices).squeeze(0).T*self.control_points[indices+1].squeeze()).T

#         if (timestamps==self.discretization_times).all():
#             list = [self.first_point]
#             for k in range(self.tangent_vectors.shape[0]):
#                 y0 = torch.stack([list[-1], self.tangent_vectors[k]])
#                 list.append(odeint(self.RHS,y0,t=torch.linspace(0,1,2))[-1][0])

#         else:
#             raise ValueError('not implemented yet')
        
#         return torch.stack(list)
    
#     def derivative(self,timestamps):
#         '''
#         Evaluate time derivative

#         Parameters
#         ----------
#         self.timestamps : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         '''
#         if timestamps == self.discretization_times:
#             return self.tangent_vectors/(self.discretization_times[1:] - self.discretization_times[:-1])
#         else:
#             raise ValueError('Not implemented yet')
        
#     def curves_for_plots(self, n_samples=16):
#         list = [self.first_point.unsqueeze(0)]
#         for k in range(self.tangent_vectors.shape[0]):
#             y0 = torch.stack([list[-1][-1], self.tangent_vectors[k]])
#             list.append(odeint(self.RHS,y0,t=torch.linspace(0,1,n_samples))[:,0])

#         return torch.stack(list[1:])

#     def Action(self):

#         points = self(self.discretization_times).detach()

#         R = torch.stack([torch.stack([torch.cos(points[...,2]), torch.sin(points[...,2]), torch.zeros(points.shape[0])],dim=1),
#                             torch.stack([-torch.sin(points[...,2]), torch.cos(points[...,2]), torch.zeros(points.shape[0])], dim=1),
#                             torch.stack([torch.zeros(points.shape[0]), torch.zeros(points.shape[0]), torch.ones(points.shape[0])], dim=1)], dim=1)
            
#         L = torch.diag(1/torch.tensor([1., 1/(self.epsilon**2), self.xi**2])).unsqueeze(0)

#         v = torch.matmul(R[1:],(self.tangent_vectors.unsqueeze(2)))

#         return (v.permute([0,2,1])@(L@v)).squeeze().sum()
    
#     def metric_inverse_matrix(self,x):

#         R = torch.stack([torch.stack([torch.cos(x[...,2]), torch.sin(x[...,2]), torch.tensor(0.)]),
#                             torch.stack([-torch.sin(x[...,2]), torch.cos(x[...,2]), torch.tensor(0.)]),
#                             torch.tensor([0., 0., 1.])])
            
#         L = torch.diag(1/torch.tensor([1., (self.epsilon**2), 1/(self.xi**2)]))

#         g_inv = (R.T)@(L@R)

#         return g_inv
    
#     def fit(self, epochs=200):
#         plt.figure(1)

#         dt=.01

#         if self.order!=2:
#             raise ValueError('Not implemented yet')
#         list_loss = []
#         for iter in range(epochs):
#             # self.first_point.zero_grad()
#             # self.tangent_vectors.zero_grad()

#             loss = 0.*self.Action() + 1*((self(self.discretization_times)[[0,-1]]-torch.tensor([[0,0,0],[1.,1.,torch.pi/2]]))**2).sum()
#             # print(self(self.discretization_times))
#             # print(self.tangent_vectors)
#             list_loss.append(loss.data)
#             if loss.data<1e-8:
#                 break

#             loss.backward()

#             eucl_grad_first_point = self.first_point.grad
#             eucl_grad_tangent_vectors = self.tangent_vectors.grad

#             with torch.no_grad():
#                 A = self.metric_inverse_matrix(self.first_point)

#                 # print(A)

#                 riem_grad_first_point = (A@eucl_grad_first_point.T).T
#                 riem_grad_tangent_vectors = (A@eucl_grad_tangent_vectors.permute([1,0])).permute([1,0])
                
#                 # print(riem_grad_first_point, riem_grad_tangent_vectors)
#                 y0 = torch.stack([self.first_point.detach(), -dt*riem_grad_first_point])

#                 self.first_point.data = odeint(self.RHS,y0,t=torch.linspace(0,1,2))[-1][0]
#                 self.tangent_vectors.data -= dt*riem_grad_tangent_vectors

#                 if (iter+1)%20==0:
#                     plt.clf()
#                     q = self.curves_for_plots(n_samples = 32)
#                     for k in range(q.shape[0]):
#                         plt.scatter(q[k,:,0].detach(), q[k,:,1].detach())
#                     # plt.axis('equal')
#                     plt.xlim(-1,1)
#                     plt.ylim(-1,1)

#                     plt.show(block=False)
#                 plt.pause(0.01)

#             self.first_point.grad.zero_()
#             self.tangent_vectors.grad.zero_()

#         plt.figure(10)
#         plt.plot(list_loss)

class RHS_geodesic_RS_parallel_2D(torch.nn.Module):
    """
    Class inheriting from torch.nn.Module, representing the right-hand side (RHS) of the geodesic equation in 2D with metric relaxed Reeds-Shepp and parallel transport along the geodesic.

    Args:
        epsilon (float): The value of epsilon parameter. Default is 1.
        xi (float): The value of xi parameter. Default is 1.
    """

    def __repr__(self):
        return 'Class inheriting from torch.nn.Module, forward function outputs the RHS f of the ODE dy/dt = f(t,y), y(0) = y0 in the case of the geodesic equation in 2D with metric potential(x)*torch.eye(2).'

    def __init__(self, epsilon=1., xi=1.):
        super().__init__()
        self.epsilon = epsilon
        self.xi = xi
     
    def forward(self, t, y):
        """
        Computes the RHS of the ODE for the geodesic equation and parallel transport along that geodesic.
        y[:,0] are the base positions
        y[:,1] are the base velocities
        y[:,2] the vectors to be parallelly transported

        Args:
            t (torch.Tensor): The time tensor.
            y (torch.Tensor): The state tensor. Shape [N,3,3]

        Returns:
            torch.Tensor: The computed RHS of the ODE.
        """
        v = torch.stack([y[...,1,:] , 
                         -(y[...,1,:].unsqueeze(1).unsqueeze(1)*self.christoffels(y[...,0,:])*(y[...,1,:].unsqueeze(1).unsqueeze(3))).sum(dim=[-1,-2]),
                         -(y[...,2,:].unsqueeze(1).unsqueeze(1)*self.christoffels(y[...,0,:])*(y[...,1,:].unsqueeze(1).unsqueeze(3))).sum(dim=[-1,-2])], dim=1) #returns the right-hand side of the ODE for the geodesic equation defined by the potential
        return v
    

    def christoffels(self, x):
        """
        Computes the Christoffel symbols for the given input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed Christoffel symbols.
        """
        Gamma = torch.zeros([x.shape[0],3,3,3])

        eps = self.epsilon
        xi = self.xi

        z = x[...,2]

        Gamma[:,0,0,2] = -.5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,0,1,2] = -.5*(eps**4 - (eps**4 - 1)*torch.cos(z)**2 - eps**2)/eps**2 
        Gamma[:,1,0,2] = .5*((eps**4 - 1)*torch.cos(z)**2 - eps**2 + 1)/eps**2 
        Gamma[:,1,1,2] = .5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,2,0,0] = .5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 
        Gamma[:,2,0,1] = -.5*(2*(eps**2 - 1)*torch.cos(z)**2 - eps**2 + 1)/((eps**2)*(xi**2)) 
        Gamma[:,2,1,1] = -.5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 

        Gamma_sym = Gamma + Gamma.permute([0,1,3,2])
        return Gamma_sym
    
    def metric_matrix(self, x):
        """
        Computes the metric matrix for the given input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed metric matrix.
        """
        epsilon = self.epsilon
        xi = self.xi

        g = torch.stack([torch.stack([torch.cos(x[...,2])**2+(1/epsilon**2)*torch.sin(x[...,2])**2,(1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.zeros(x.shape[0])]),
                        torch.stack([(1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.sin(x[...,2])**2+(1/epsilon**2)*torch.cos(x[...,2])**2,torch.zeros(x.shape[0])]),
                        torch.tensor([0,0,(xi)**2]).expand([x.shape[0],3]).T]) 
        
        return g.permute([2,0,1])
    
    def metric_inverse_matrix(self, x):
        """
        Computes the inverse of the metric matrix for the given input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed inverse of the metric matrix.
        """
        epsilon = self.epsilon
        xi = self.xi
        g_inv = torch.stack([torch.stack([torch.cos(x[...,2])**2+(epsilon**2)*torch.sin(x[...,2])**2,(1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.zeros(x.shape[0])]),
                        torch.stack([(1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.sin(x[...,2])**2+(epsilon**2)*torch.cos(x[...,2])**2,torch.zeros(x.shape[0])]),
                        torch.tensor([0,0,(1/xi)**2]).expand([x.shape[0],3]).T]) 
        
        return g_inv.permute([2,0,1])
    

class PiecewiseGeodesic_RS(torch.nn.Module):
    '''
        Module wrapper for Exponential curves in Reeds-Shepp space
    
        Args:
            control_points: Points defining t
        
        Note : 
    '''
    
    def __repr__(self):
        return "Piecewise exponential curves for relaxed Reeds-Shepp space"
    
    def __init__(self, n_points, epsilon = 1., xi = 1., discretization_times = None):
        super().__init__()
        self.points = torch.randn([n_points-1,3], requires_grad=True) #
        self.tangent_vectors = torch.randn([n_points-1,3], requires_grad=True)
        
        if discretization_times==None:
            self.discretization_times = torch.linspace(0,1,n_points)
        else:
            self.discretization_times =  discretization_times

        self.order = n_points
        self.dim =  3

        self.epsilon = epsilon
        self.xi = xi

        self.RHS = torch.compile(RHS_geodesic_RS_2D(epsilon = self.epsilon, xi = self.xi))
        self.RHS_optim = torch.compile(RHS_geodesic_RS_parallel_2D(epsilon = self.epsilon, xi = self.xi))

        assert (self.order == self.discretization_times.shape[0])
        
    def forward(self, timestamps):
        '''
        Evaluation of

        Parameters
        ----------
        timestamps : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        # '''
        list=[]

        for k in range(self.order-1):
            mask_k = torch.logical_or(((timestamps>=self.discretization_times[k])*(timestamps<self.discretization_times[k+1])),(timestamps==1.)*(self.discretization_times[k+1]==1.))
            if mask_k.sum()>0:
                output = odeint(self.RHS, torch.stack([self.points[k].unsqueeze(0), self.tangent_vectors[k].unsqueeze(0)], dim=1), t = (timestamps[mask_k] - self.discretization_times[k])/(self.discretization_times[k+1]- self.discretization_times[k]))
                list.append(output[:,:,0])
        
        return torch.cat(list, dim=0)
    
    def derivative(self,timestamps):
        '''
        Evaluate time derivative

        Parameters
        ----------
        self.timestamps : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        timestamps_to_int = torch.minimum(torch.floor(timestamps*(self.order)).long(), torch.tensor(self.order-1))

        list=[]

        for k in range(self.order-1):
            mask_k = (timestamps_to_int==k)
            if mask_k.sum()>0:
                output = odeint(self.RHS, torch.stack([self.points[k].unsqueeze(0), self.tangent_vectors[k].unsqueeze(0)], dim=1), t = self.order*timestamps[mask_k] - k)
                list.append(output[:,:,1])
        
        return torch.stack(list)
        
    def curves_for_plots(self, n_samples=16):
        y0 = torch.stack([self.points, self.tangent_vectors], dim=1)
        return odeint(self.RHS,y0,t=torch.linspace(0,1,n_samples))[:,:,0].permute([1,0,2])

    def Action(self):
        A = self.RHS.metric_matrix(self.points)
        v = self.tangent_vectors.unsqueeze(2)
        return (v.permute([0,2,1])@(A@v)).squeeze().sum()
    
    def fit(self, epochs=200):
        plt.figure(1)

        dt = .01

        list_loss = []
        for iter in range(epochs):

            loss = 0.*self.Action() + 1*((self(self.discretization_times)[[0,-1]]-torch.tensor([[0,0,0],[1.,1.,torch.pi/2]]))**2).sum()

            list_loss.append(loss.data)
            if loss.data<1e-6:
                break

            loss.backward()

            x_eucl_grad = self.points.grad
            v_eucl_grad = self.tangent_vectors.grad

            with torch.no_grad():
                A = self.RHS.metric_inverse_matrix(x)
                if x_eucl_grad==None:                    
                    y0 = torch.stack([self.points, 0*self.tangent_vectors, self.tangent_vectors - dt*v_eucl_grad],dim=1)
                else:
                    riem_grad = (A@x_eucl_grad.unsqueeze(2)).squeeze()
                    
                    y0 = torch.stack([self.points, -dt*riem_grad, self.tangent_vectors - dt*v_eucl_grad],dim=1)

                result_ode = odeint(self.RHS_optim,y0,t=torch.linspace(0,1,2))[-1]
                
                self.points.data = result_ode[:,0]
                self.tangent_vectors.data = result_ode[:,2]

                if (iter+1)%20==0:
                    plt.clf()
                    q = self.curves_for_plots(n_samples = 32)
                    for k in range(q.shape[0]):
                        plt.scatter(q[k,:,0].detach(), q[k,:,1].detach())
                    # plt.axis('equal')
                    plt.xlim(-1,1)
                    plt.ylim(-1,1)

                    plt.show(block=False)
                plt.pause(0.01)

            self.first_point.grad.zero_()
            self.tangent_vectors.grad.zero_()

        plt.figure(10)
        plt.plot(list_loss)

if __name__ == '__main__':



    # RHS = RHS_geodesic_RS_2D(epsilon=.5, xi=.9)

    # trajectory = odeint(RHS,torch.tensor([[0.,0.,0.],
    #                          [1.,0.,1.]]),t=torch.linspace(0,1,128), method='rk4')
    
    # # print(trajectory.shape)
    # print(trajectory[:,0,2])
    
    # plt.figure(0)
    # plt.scatter(trajectory[:,0,0], trajectory[:,0,1], c=torch.remainder(trajectory[:,0,2], torch.pi))
    # plt.axis('equal')

    # C = torch.cat([torch.tensor([[0.,0.,0.]]),torch.randn([1,3])])
    # C.requires_grad = True

    # curve_mod = ExponentialCurveModule_RS(n_points=2, epsilon=.5)

    # epochs = 400
    # curve_mod.fit(epochs)

    #%%
    from tqdm import tqdm

    epochs = 1

    n_points = 10
    x = torch.zeros([n_points,3], requires_grad=True)
    v = torch.zeros([n_points,3], requires_grad=True)


    t = torch.linspace(0,1,n_points)

    # x1 = torch.tensor([[1,1,0],
    #                    [-1,1,torch.pi/2]])

    x1 = torch.stack([torch.cos(t*torch.pi), torch.sin(t*torch.pi), torch.pi/2 -t*torch.pi], dim=1)

    dt = .02

    eps = .5
    xi = 2.

    constr = 5.

    RHS_optim = torch.compile(RHS_geodesic_RS_parallel_2D(epsilon = eps, xi = xi))

    RHS = torch.compile(RHS_geodesic_RS_2D(epsilon = eps, xi = xi))

    list_pos = []
    list_loss = []

    for iter in tqdm(range(epochs)):         
            x_v = odeint(RHS,torch.stack([x,v],dim=1),t=torch.linspace(0,1,2))[-1,:,0]

            loss = ((x[...,:2]-x1[...,:2])**2).sum() + .0001*(((RHS.metric_matrix(x)@v.unsqueeze(-1)).squeeze())*v).sum() + constr*((x[1:]-x_v[:-1])**2).sum()

            list_loss.append(loss.data)

            loss.backward()

            x_eucl_grad = x.grad
            v_eucl_grad = v.grad

            with torch.no_grad():
                A = RHS_optim.metric_inverse_matrix(x)
                if x_eucl_grad==None:                    
                    y0 = torch.stack([x, 0*v, v - dt*v_eucl_grad],dim=1)
                else:
                    riem_grad = (A@x_eucl_grad.unsqueeze(2)).squeeze()
                    
                    y0 = torch.stack([x, -dt*riem_grad, v - dt*v_eucl_grad],dim=1)

                result_ode = odeint(RHS_optim,y0,t=torch.linspace(0,1,2))[-1]
                
                x.data = result_ode[:,0]
                v.data = result_ode[:,2]
            
            list_pos.append(x_v.data)
            
            if x_eucl_grad!=None:
                x.grad.zero_()
            v.grad.zero_()

    stack_pos = torch.stack(list_pos)
    #%%
    plt.figure(0)
    for k in range(x.shape[0]):
        plt.scatter(stack_pos[:,k,0], stack_pos[:,k,1], s=5)


    plt.figure(1)
    plt.plot(torch.log(torch.stack(list_loss))/torch.log(torch.tensor(10)))
    
    plt.figure(2)
    x_v = odeint(RHS,torch.stack([x,v],dim=1),t=torch.linspace(0,1,20))[:,:,0].detach()
    for k in range(x.shape[0]):
        plt.plot(x_v[:,k,0], x_v[:,k,1])
    # sigma = torch.tensor(0.05)

    # def gaussian(x):
    #     return torch.exp(-((x-torch.tensor([-0.5,0.]))**2).sum(dim=-1)/sigma)/torch.sqrt(sigma)+ torch.exp(-((x-torch.tensor([0.5,0.]))**2).sum(dim=-1)/sigma)/torch.sqrt(sigma) + 1.0
    
    # curve_mod = ExponentialCurveModule(control_points = torch.cat([torch.tensor([[0.,0.]]),torch.randn([3,2])]), potential = gaussian)

    # q = curve_mod.curves_for_plots(n_samples = 64)

    # plt.figure(0)
    # X,Y = torch.meshgrid(torch.linspace(-1,1,128), torch.linspace(-1,1,128))
    # plt.imshow(gaussian(torch.stack([X,Y]).permute([1,2,0])).T, origin = 'lower', extent=[-1,1,-1,1])
    # for k in range(q.shape[0]):
    #     plt.scatter(q[k,:,0], q[k,:,1])
    # plt.show()
    
    # RHS_gaussian_geom = RHS_geodesic_isotropic2D(potential = gaussian)

    # log_func = geodesic_distance_via_log_map()

    # a = torch.tensor([.9,1.], requires_grad=True)

    # optimizer = torch.optim.Adam(params = [a], lr=.01)
    # for i in range(50):
    #     optimizer.zero_grad()
    #     loss = log_func.apply(a, torch.tensor([1.,1.]), RHS_gaussian_geom, gaussian)
    #     print(loss.data, a.data)

    #     loss.backward()
    #     optimizer.step()
# %%
