#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:27:10 2023

@author: tbertrand
"""

import torch

import scipy
from scipy.linalg import hankel
import numpy as np

import matplotlib.pyplot as plt

from torchdiffeq import odeint_adjoint as odeint

from tqdm import tqdm
import time


# if necessary we define the Reeds-Shepp norm in a separate function
def RS_relaxed_norm(x, v, eps=.05, xi=.1):
    assert(v.ndim == 2 and v.shape[1] == 3)
    # assert v.shape[0] == 3
    theta = torch.remainder(x[:,2], torch.pi)
    return((torch.cos(theta)*v[:,0] + torch.sin(theta)*v[:,1])**2
           + (1/(eps**2))*(-torch.sin(theta)*v[:,0] + torch.cos(theta)*v[:,1])**2
           + (xi**2)*v[:,2]**2)

# Module for Bezier curves
class BezierCurveModule(torch.nn.Module):
    '''
        Module wrapper for Bezier curves
    
        Args:
            control_points: torch.Tensor containing the sequence of points that define the Bezier Curve
    '''
    
    def __repr__(self):
        return "Bezier curve defined by the input control"
    
    def __init__(self, nc=8, n_start=1, w=None, timestamps=None, h=1.):
        super().__init__()
        if w==None:
            # Initiate n_start sets of control points at random
            self.control_points = torch.randn([n_start, nc,2], requires_grad=True)
        else:
            x=torch.linspace(-1,1,w.shape[0])
            X = torch.stack(torch.meshgrid([x,x], indexing='ij'), dim=0)
            
            timestamps_control_points = torch.linspace(0,1,nc)
            indices = torch.floor(timestamps_control_points*(w.shape[-1]-2)).long()

            w_interp = ((1-(timestamps_control_points*(w.shape[-1]-1)-indices)).unsqueeze(0).unsqueeze(0)*w[...,indices] + (timestamps_control_points*(w.shape[-1]-1)-indices).unsqueeze(0).unsqueeze(0)*w[...,indices+1].squeeze())
            P = (torch.exp(h*w_interp)/torch.exp(h*w_interp).sum(dim=[0,1], keepdim=True)).numpy()
            
            # Initiate n_start sets of control points at random by sampling from a distribution depending on the residual
            random_choice = np.stack([np.random.choice(P.shape[0]*P.shape[1],n_start,p=P[:,:,i].reshape([-1])) for i in range(P.shape[-1])])
            self.control_points = X.reshape([2,-1])[:,random_choice].permute([2,1,0])
            self.control_points.requires_grad=True

        self.order = nc

        self.chosen_control_points = None
        
        self.n_start = n_start

        n = self.order-1

        #Define the matrix for the computation of the Action in the Eucledian case
        self.M =(1/(2*n-1)) * (scipy.special.comb(n-1,np.arange(0,n))[:,None]*scipy.special.comb(n-1,np.arange(0,n))[None,:])/hankel(scipy.special.comb(2*n-2,np.arange(0,n)), scipy.special.comb(2*n-2,np.arange(n-1,2*n-1)))
        
    def get_control_points(self):
        # function to access control points
        return self.control_points

    def forward(self, timestamps):
        '''
        Evaluation of Bezier curve via naive implementation of de Casteljau algorithm

        Parameters
        ----------
        timestamps : torch.Tensor
            1D torch.Tensor containign the timestamps at which we want to compute the position of the curve via de Casteljau's algorithm.

        Returns
        -------
        TYPE
            torch.Tensor containing the computed points.

        '''
        
        list_points = [self.control_points.unsqueeze(1)]
        while list_points[-1].shape[-2] > 1:
            list_points.append(
                (1 - timestamps).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * list_points[-1][...,:-1,:] + timestamps.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * list_points[-1][...,1:,:])
            
        return list_points[-1].squeeze()
    
    def derivative(self,timestamps):
        '''
        Evaluate time derivative of Bezier curve by naive implementation of de Casteljau algorithm (run on the successive differences of control points)

        Parameters
        ----------
        timestamps : torch.Tensor
            1D torch.Tensor containign the timestamps at which we want to compute the position of the curve via de Casteljau's algorithm.

        Returns
        -------
        TYPE
            torch.Tensor containing the computed points.

        '''
        
        list_points = [(self.control_points[:-1]-self.control_points[1:]).unsqueeze(2)]
        while list_points[-1].shape[-2] > 1:
            list_points.append(
                (1 - timestamps).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * list_points[-1][:-1] + timestamps.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * list_points[-1][1:])
            
        return self.order*list_points[-1].squeeze()
    
    def Action(self, geometry='euclidean', n_evaluation=128, epsilon=1., xi=1.):
        '''
        Evaluate the action of the curve in the specified geometry (default is euclidean).

        Parameters
        ----------
        geometry : str
            Specifies geometry, only euclidean available for now.
        
        n_evaluation : int
            Number of points considered for the evaluation of the Action in the RS case.

        epsilon : float

        xi : float

        

        Returns
        -------
        Value of the Action.

        '''
        assert geometry=='euclidean'
        if geometry=='euclidean':
            dcontrol = (self.control_points[...,1:,:]-self.control_points[...,:-1,:])
            return (dcontrol.double().permute([0,2,1])@(torch.tensor(self.M).double().unsqueeze(0)@dcontrol.double())).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        elif geometry=='RS' and self.control_points.shape[1]==3:
            curve = self(torch.linspace(0,1,n_evaluation)).squeeze()
            dcurve = self.derivative(torch.linspace(0,1,n_evaluation)).squeeze()

            return RS_relaxed_norm(curve, dcurve, eps = epsilon, xi=xi).mean()
        
        else:
            raise ValueError('bad shape/geometry')
    
    def curves_for_plots(self, n_samples=16):
        return self(torch.linspace(0,1,n_samples*self.order)).detach()

    def fit(self, y, operator, timestamps, n_epoch=1, lr=1e-2, regul=1e-2, n_sample=8):
        """
        Fits the model to the given data using the specified parameters.

        Args:
            y (torch.Tensor): The target values.
            operator (callable): The operator function to apply to the model's output.
            timestamps (torch.Tensor): The timestamps for the data.
            n_epoch (int, optional): The number of epochs to train the model. Defaults to 1.
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-2.
            regul (float, optional): The regularization parameter. Defaults to 1e-2.
            n_sample (int, optional): The number of samples to use for plotting. Defaults to 8.

        Returns:
            tuple: A tuple containing the following elements:
                - phi (torch.Tensor): The phi values for the best control points.
                - nrj (torch.Tensor): The energy values for each epoch.
                - phi_vec (torch.Tensor): The phi values for each epoch.
                - plot_vec (torch.Tensor): The plot values for each epoch.
                - points_vec (torch.Tensor): The control points for each epoch.
                - result_vec (torch.Tensor): The result values for each epoch.
        """
        optimizer = torch.optim.Adam([self.control_points], lr=lr)
        
        nrj = torch.zeros(self.n_start, n_epoch)
        plot_vec = torch.zeros(self.n_start, n_epoch, 2, n_sample*timestamps.shape[0])
        result_vec = torch.zeros(self.n_start, n_epoch, 2, timestamps.shape[0])
        points_vec = torch.zeros(self.n_start, n_epoch, self.order, 2)
        phi_vec = torch.zeros(self.n_start, n_epoch, y.shape[0], y.shape[0])

        for epoch in tqdm(range(n_epoch), desc=f'Computing UFW step',
                    position=0,
                    leave=False):
            
            optimizer.zero_grad()
            result = self(timestamps).squeeze()

            phi = operator(result)
            
            a = ((-y)*phi).mean(dim=[-1,-2,-3]) + regul*self.Action(geometry='euclidean')

            a.mean().backward()
            optimizer.step()

            nrj[:, epoch] = a.data
            phi_vec[:, epoch, :, :] = phi.mean(dim=-1).detach()
            plot_vec[:, epoch, :, :] = self(torch.linspace(0,1,n_sample*timestamps.shape[0])).squeeze().detach().permute([0,2,1])
            result_vec[:, epoch, :, :] = result.detach().permute([0,2,1])
            points_vec[:, epoch, :, :] = self.get_control_points().detach()
        
        idx_best = torch.argmin(nrj[:,-1])
        self.chosen_control_points = self.control_points[idx_best]

        return  phi[idx_best], nrj[idx_best], phi_vec[idx_best], plot_vec[idx_best], points_vec[idx_best], result_vec[idx_best]


# Module for Polygonial curves
class PolygonalCurveModule(torch.nn.Module):
    '''
        Module wrapper for Polygonal curves
    
        Args:
            control_points: torch.Tensor
                Sequence of points defining the polygonal curve
        
        Note : 
    '''
    
    def __repr__(self):
        return "Polygonal curves"
    
    def __init__(self, nc=8, n_start=1, w=None, timestamps=None):
        super().__init__()
        if w==None:
            # Initiate n_start sets of control points at random
            self.control_points = torch.randn([n_start,nc,2], requires_grad=True)
        else:
            x=torch.linspace(-1,1,w.shape[0])
            X = torch.stack(torch.meshgrid([x,x], indexing='ij'), dim=0)

            timestamps_control_points = torch.linspace(0,1,nc)
            indices = torch.floor(timestamps_control_points*(w.shape[-1]-2)).long()

            w_interp = ((1-(timestamps_control_points*(w.shape[-1]-1)-indices)).unsqueeze(0).unsqueeze(0)*w[...,indices] + (timestamps_control_points*(w.shape[-1]-1)-indices).unsqueeze(0).unsqueeze(0)*w[...,indices+1].squeeze())
            P = (torch.exp(w_interp)/torch.exp(w_interp).sum(dim=[0,1], keepdim=True)).numpy()

            # Initiate n_start sets of control points at random by sampling from a distribution depending on the residual
            random_choice = np.stack([np.random.choice(P.shape[0]*P.shape[1],n_start,p=P[:,:,i].reshape([-1])) for i in range(P.shape[-1])])
            self.control_points = X.reshape([2,-1])[:,random_choice].permute([2,1,0])
            self.control_points.requires_grad=True
        self.order = nc

        if timestamps==None:
            self.timestamps = torch.linspace(0,1,w.shape[-1])
        else:
            self.timestamps = timestamps

        self.chosen_control_points = None
        self.n_start = n_start
        
    def get_control_points(self):
        # method to access control points
        return self.control_points
        
    def forward(self, timestamps):
        '''
        Evaluation of Bezier curve via naive implementation of de Casteljau algorithm

        Parameters
        ----------
        timestamps : torch.Tensor
            1D torch.Tensor containign the timestamps at which we want to compute the position of the curve via de Casteljau's algorithm.

        Returns
        -------
        torch.Tensor
            torch.Tensor containing the computed points.

        '''
        
        indices = torch.floor(timestamps*(self.control_points.shape[1]-2)).long()
        return ((1-(timestamps*(self.control_points.shape[1]-1)-indices)).unsqueeze(1)*self.control_points[...,indices,:] + (timestamps*(self.control_points.shape[1]-1)-indices).unsqueeze(1)*self.control_points[...,indices+1,:].squeeze())
    
    def derivative(self,timestamps):
        indices = torch.floor(timestamps*(self.control_points.shape[0]-2)).long()
        return (self.control_points[...,indices+1,:].squeeze()-self.control_points[indices].squeeze())*(self.control_points.shape[0]-1)
    
    def curves_for_plots(self, n_samples=16):
        return self(torch.linspace(0,1,n_samples*self.order)).detach()
    
    def Action(self, geometry='euclidean', n_evaluation=128, epsilon=1., xi=1.):
        # compute the action of the curve
        assert geometry=='euclidean'
        if geometry=='euclidean':
            # return (self.derivative(torch.linspace(0,1,n_evaluation)).squeeze()**2).mean()
            return ((self.control_points[...,1:,:]-self.control_points[...,:-1,:])**2).mean(dim=[-1,-2])
        elif geometry=='RS' and self.control_points.shape[1]==3:
            curve = self(torch.linspace(0,1,n_evaluation)).squeeze()
            dcurve = self.derivative(torch.linspace(0,1,n_evaluation)).squeeze()

            return RS_relaxed_norm(curve, dcurve, eps = epsilon, xi=xi).mean()
        else:
            raise ValueError('bad shape/geometry')
        
    def fit(self, y, operator, timestamps, n_epoch=1, lr=1e-2, regul=1e-2, n_sample=8):
        """
        Fits the model to the given data by optimizing the control points.

        Args:
            y (torch.Tensor): The target values.
            operator (callable): The operator function to apply to the model's output.
            timestamps (torch.Tensor): The timestamps for the data.
            n_epoch (int, optional): The number of epochs to train for. Defaults to 1.
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-2.
            regul (float, optional): The regularization parameter. Defaults to 1e-2.
            n_sample (int, optional): The number of samples to use for plotting. Defaults to 8.

        Returns:
            tuple: A tuple containing the following elements:
                - phi (torch.Tensor): The optimized phi values.
                - nrj (torch.Tensor): The energy values.
                - phi_vec (torch.Tensor): The phi values for each epoch.
                - plot_vec (torch.Tensor): The plot values for each epoch.
                - points_vec (torch.Tensor): The control points for each epoch.
                - result_vec (torch.Tensor): The result values for each epoch.
        """
        optimizer = torch.optim.Adam([self.control_points], lr=lr)
        
        nrj = torch.zeros(self.n_start, n_epoch)
        plot_vec = torch.zeros(self.n_start, n_epoch, 2, n_sample*timestamps.shape[0])
        result_vec = torch.zeros(self.n_start, n_epoch, 2, timestamps.shape[0])
        points_vec = torch.zeros(self.n_start, n_epoch, self.order, 2)
        phi_vec = torch.zeros(self.n_start, n_epoch, y.shape[0], y.shape[0])

        for epoch in tqdm(range(n_epoch), desc=f'Computing UFW step',
                    position=0,
                    leave=False):
            
            optimizer.zero_grad()
            result = self(timestamps).squeeze()

            phi = operator(result)
            
            a = ((-y.unsqueeze(0))*phi).mean(dim=[-1,-2,-3]) + regul*self.Action(geometry='euclidean')

            a.mean().backward()
            optimizer.step()

            nrj[:,epoch] = a.data
            phi_vec[:,epoch, :, :] = phi.mean(dim=-1).detach()
            plot_vec[:,epoch, :, :] = self(torch.linspace(0,1,n_sample*timestamps.shape[0])).squeeze().detach().permute([0,2,1])
            result_vec[:,epoch, :, :] = result.detach().permute([0,2,1])
            points_vec[:,epoch, :, :] = self.get_control_points().detach()
        idx_best = torch.argmin(nrj[:,-1])
        self.chosen_control_points = self.control_points[idx_best]

        return  phi[idx_best], nrj[idx_best], phi_vec[idx_best], plot_vec[idx_best], points_vec[idx_best], result_vec[idx_best]


# Define RHS for solving the geodesic equation in the Reeds-Shepp geometry
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
        v = torch.stack([y[...,1,:],
                          -(y[...,1,:].unsqueeze(2).unsqueeze(2)*self.christoffels(y[...,0,:])*(y[...,1,:].unsqueeze(2).unsqueeze(4))).sum(dim=[3,4])], dim=2)
        return v
    

    def christoffels(self, x):
        """
        Computes the Christoffel symbols for the geodesic equation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor representing the Christoffel symbols.
        """
        Gamma = torch.zeros([x.shape[0],x.shape[1],3,3,3])

        eps = self.epsilon
        xi = self.xi
        z = x[...,2]

        Gamma[:,:,0,0,2] = -.5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,:,0,1,2] = -.5*(eps**4 - (eps**4 - 1)*torch.cos(z)**2 - eps**2)/eps**2 
        Gamma[:,:,1,0,2] = .5*((eps**4 - 1)*torch.cos(z)**2 - eps**2 + 1)/eps**2 
        Gamma[:,:,1,1,2] = .5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,:,2,0,0] = .5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 
        Gamma[:,:,2,0,1] = -.5*(2*(eps**2 - 1)*torch.cos(z)**2 - eps**2 + 1)/((eps**2)*(xi**2)) 
        Gamma[:,:,2,1,1] = -.5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 

        Gamma_sym = Gamma + Gamma.permute([0,1,2,4,3])
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

        # g = torch.stack([torch.stack([torch.cos(x[...,2])**2+(1/epsilon**2)*torch.sin(x[...,2])**2,(1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.zeros(x.shape[0])]),
        #                 torch.stack([(1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2]),torch.sin(x[...,2])**2+(1/epsilon**2)*torch.cos(x[...,2])**2,torch.zeros(x.shape[0])]),
        #                 torch.tensor([0,0,(xi)**2]).expand([x.shape[1],3]).T]) 
        
        g = torch.zeros([x.shape[0],x.shape[1],3,3])
        g[...,0,0] = torch.cos(x[...,2])**2+(1/epsilon**2)*torch.sin(x[...,2])**2
        g[...,0,1] = (1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g[...,1,0] = (1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g[...,1,1] = torch.sin(x[...,2])**2+(1/epsilon**2)*torch.cos(x[...,2])**2
        g[...,2,2] = (xi)**2
        return g
    
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
        g_inv = torch.zeros([x.shape[0],x.shape[1],3,3])
        g_inv[...,0,0] = torch.cos(x[...,2])**2+(epsilon**2)*torch.sin(x[...,2])**2
        g_inv[...,0,1] = (1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g_inv[...,1,0] = (1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g_inv[...,1,1] = torch.sin(x[...,2])**2+(epsilon**2)*torch.cos(x[...,2])**2
        g_inv[...,2,2] = (1/xi)**2
        return g_inv

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
                         -(y[...,1,:].unsqueeze(2).unsqueeze(2)*self.christoffels(y[...,0,:])*(y[...,1,:].unsqueeze(2).unsqueeze(4))).sum(dim=[-1,-2]),
                         -(y[...,2,:].unsqueeze(2).unsqueeze(2)*self.christoffels(y[...,0,:])*(y[...,1,:].unsqueeze(2).unsqueeze(4))).sum(dim=[-1,-2])], dim=2) #returns the right-hand side of the ODE for the geodesic equation defined by the potential
        return v
    

    def christoffels(self, x):
        """
        Computes the Christoffel symbols for the geodesic equation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor representing the Christoffel symbols.
        """
        # print(x.shape)
        Gamma = torch.zeros([x.shape[0],x.shape[1],3,3,3])

        eps = self.epsilon
        xi = self.xi
        z = x[...,2]

        Gamma[:,:,0,0,2] = -.5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,:,0,1,2] = -.5*(eps**4 - (eps**4 - 1)*torch.cos(z)**2 - eps**2)/eps**2 
        Gamma[:,:,1,0,2] = .5*((eps**4 - 1)*torch.cos(z)**2 - eps**2 + 1)/eps**2 
        Gamma[:,:,1,1,2] = .5*(eps**4 - 1)*torch.cos(z)*torch.sin(z)/eps**2 
        Gamma[:,:,2,0,0] = .5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 
        Gamma[:,:,2,0,1] = -.5*(2*(eps**2 - 1)*torch.cos(z)**2 - eps**2 + 1)/((eps**2)*(xi**2)) 
        Gamma[:,:,2,1,1] = -.5*(eps**2 - 1)*torch.cos(z)*torch.sin(z)/((eps**2)*(xi**2)) 

        Gamma_sym = Gamma + Gamma.permute([0,1,2,4,3])
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

        g = torch.zeros([x.shape[0],x.shape[1],3,3])
        g[...,0,0] = torch.cos(x[...,2])**2+(1/epsilon**2)*torch.sin(x[...,2])**2
        g[...,0,1] = (1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g[...,1,0] = (1-(1/epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g[...,1,1] = torch.sin(x[...,2])**2+(1/epsilon**2)*torch.cos(x[...,2])**2
        g[...,2,2] = (xi)**2
        
        return g
    
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
        g_inv = torch.zeros([x.shape[0],x.shape[1],3,3])
        g_inv[...,0,0] = torch.cos(x[...,2])**2+(epsilon**2)*torch.sin(x[...,2])**2
        g_inv[...,0,1] = (1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g_inv[...,1,0] = (1-(epsilon**2))*torch.cos(x[...,2])*torch.sin(x[...,2])
        g_inv[...,1,1] = torch.sin(x[...,2])**2+(epsilon**2)*torch.cos(x[...,2])**2
        g_inv[...,2,2] = (1/xi)**2

        return g_inv
    

# Define RHS for solving the paralell transport along the geodesic equation in the Reeds-Shepp geometry for the update step in the Riemannian gradient descent
class PiecewiseGeodesic_RS(torch.nn.Module):
    '''
    Module wrapper for Exponential curves in Reeds-Shepp space

    Args:
        n_points (int): Number of control points.
        epsilon (float, optional): Epsilon parameter. Defaults to 1.0.
        xi (float, optional): Xi parameter. Defaults to 1.0.
        discretization_times (torch.Tensor, optional): Discretization times. If None, it will be generated using torch.linspace(0, 1, n_points). Defaults to None.

    Attributes:
        points (torch.Tensor): Control points.
        tangent_vectors (torch.Tensor): Tangent vectors.
        discretization_times (torch.Tensor): Discretization times.
        order (int): Order of the curves.
        dim (int): Dimension of the curves.
        epsilon (float): Epsilon parameter.
        xi (float): Xi parameter.
        RHS (torch.Tensor): Right-hand side of the geodesic equations.
        RHS_optim (torch.Tensor): Right-hand side of the geodesic equations for optimization.
    '''

    def __repr__(self):
        return "Piecewise exponential curves for relaxed Reeds-Shepp space"

    def __init__(self, n_points, n_start=1, epsilon=1., xi=1., discretization_times=None, w=None, RHS=None, RHS_optim=None, timestamps=None):
        super().__init__()

        if w==None:
            self.points = torch.randn([n_start, n_points-1, 3], requires_grad=True)
        else:
            x=torch.linspace(-1,1,w.shape[0])
            X = torch.stack(torch.meshgrid([x,x], indexing='ij'), dim=0)

            timestamps_control_points = torch.linspace(0,1,n_points)[:-1]
            indices = torch.floor(timestamps_control_points*(w.shape[-1]-2)).long()

            w_interp = ((1-(timestamps_control_points*(w.shape[-1]-1)-indices)).unsqueeze(0).unsqueeze(0)*w[...,indices] + (timestamps_control_points*(w.shape[-1]-1)-indices).unsqueeze(0).unsqueeze(0)*w[...,indices+1].squeeze())
            P = (torch.exp(w_interp)/torch.exp(w_interp).sum(dim=[0,1], keepdim=True)).numpy()
            random_choice = np.stack([np.random.choice(P.shape[0]*P.shape[1],n_start,p=P[:,:,i].reshape([-1])) for i in range(P.shape[-1])])
            self.points = torch.cat([X.reshape([2,-1])[:,random_choice].permute([2,1,0]),torch.randn([n_start, n_points-1, 1])], dim=-1)
            self.points.requires_grad=True

        self.tangent_vectors = torch.zeros([n_start, n_points-1, 3], requires_grad=True)
        # print('init', self.points.shape, self.tangent_vectors.shape)
        if discretization_times is None:
            self.discretization_times = torch.linspace(0, 1, n_points)
        else:
            self.discretization_times = discretization_times

        self.order = n_points
        self.dim = 3

        self.epsilon = epsilon
        self.xi = xi

        if RHS==None:
            self.RHS = torch.compile(RHS_geodesic_RS_2D(epsilon=self.epsilon, xi=self.xi))
        else:
            self.RHS = RHS
        
        if RHS_optim==None:
            self.RHS_optim = torch.compile(RHS_geodesic_RS_parallel_2D(epsilon=self.epsilon, xi=self.xi))
        else:
            self.RHS_optim = RHS_optim

        self.n_start = n_start

        assert (self.order == self.discretization_times.shape[0])

    # @torch.compile
    def forward(self, timestamps):
        '''
        Evaluate the piecewise geodesic curves at the given timestamps.

        Parameters:
            timestamps (torch.Tensor): Timestamps at which to evaluate the curves.

        Returns:
            torch.Tensor: Evaluated curves at the given timestamps.
        '''
        list = []

        for k in range(self.order-1):
            mask_k = torch.logical_or(((timestamps >= self.discretization_times[k]) * (timestamps < self.discretization_times[k+1])), (timestamps == 1.) * (self.discretization_times[k+1] == 1.))
            if mask_k.sum() > 0:
                output = odeint(self.RHS, torch.stack([self.points[:,k].unsqueeze(1), self.tangent_vectors[:,k].unsqueeze(1)], dim=2), t=(timestamps[mask_k] - self.discretization_times[k]) / (self.discretization_times[k+1] - self.discretization_times[k]))
                list.append(output[:, :, 0, 0])
        return torch.cat(list, dim=0).permute([1, 0, 2])

    def Action(self):
        '''
        Compute the action of the curves.

        Returns:
            torch.Tensor: Action of the curves.
        '''
        A = self.RHS.metric_matrix(self.points)
        v = self.tangent_vectors.unsqueeze(3)
        # print(A.shape, v.shape, (v.permute([0, 1, 3, 2]) @ (A @ v)).shape, (v.permute([0, 1, 3, 2]) @ (A @ v)).squeeze().mean(dim=[-1]).shape)
        return (v.permute([0, 1, 3, 2]) @ (A @ v)).squeeze().mean(dim=[-1])

    def fit(self, y, operator, timestamps, n_epoch=1, lr=1e-2, regul=1e-2, constr=1., n_sample=8):
        '''
        Fit the curves to the given data.

        Parameters:
            y (torch.Tensor): Target data.
            operator (callable): Operator to apply to the curves.
            timestamps (torch.Tensor): Timestamps of the data.
            n_epoch (int, optional): Number of epochs. Defaults to 1.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            regul (float, optional): Regularization parameter. Defaults to 1e-2.
            constr (float, optional): Constraint parameter. Defaults to 5.0.
        '''
        nrj = torch.zeros(self.n_start, n_epoch)
        plot_vec = torch.zeros(self.n_start, n_epoch, 3, n_sample*timestamps.shape[0])
        result_vec = torch.zeros(self.n_start, n_epoch, 3, timestamps.shape[0])
        print(self.order)
        points_vec = torch.zeros(self.n_start, n_epoch, self.order, 3)
        phi_vec = torch.zeros(self.n_start, n_epoch, y.shape[0], y.shape[0])

        for epoch in tqdm(range(n_epoch)):
            result = self(timestamps)
            phi = operator(result)

            constraint_loss = ((odeint(self.RHS, torch.stack([self.points, self.tangent_vectors], dim=2), t=torch.tensor([0., 1.]))[-1, :, :-1, 0] - self.points[:,1:]) ** 2).mean(dim=[-1,-2])
            loss = (phi * (-y)).mean(dim=[-1,-2,-3]) + regul * self.Action() + constr * constraint_loss
            loss.mean().backward()

            # Get euclidean gradients for the points and tangent vectors for the Riemannian gradient descent
            x_eucl_grad = self.points.grad
            x_eucl_grad = x_eucl_grad/(x_eucl_grad.norm(dim=2, keepdim=True) + 1e-16)
            v_eucl_grad = self.tangent_vectors.grad
            v_eucl_grad = v_eucl_grad/(v_eucl_grad.norm(dim=2, keepdim=True) + 1e-16)
            
            with torch.no_grad():
                A = self.RHS.metric_inverse_matrix(self.points)
                if x_eucl_grad is None:
                    y0 = torch.stack([self.points, 0 * self.tangent_vectors, self.tangent_vectors - lr * v_eucl_grad], dim=2)
                else:
                    riem_grad = (A @ x_eucl_grad.unsqueeze(3)).squeeze()

                    y0 = torch.stack([self.points, -lr * riem_grad, self.tangent_vectors - lr * v_eucl_grad], dim=2)

                # Perform exponential map and parallel transport for the Riemannian Gradient Descent
                result_ode = odeint(self.RHS_optim, y0, t=torch.linspace(0, 1, 2))[-1]

                # print(result_ode.shape)
                self.points.data = result_ode[..., 0,:]
                self.tangent_vectors.data = result_ode[..., 2,:]

            nrj[:,epoch] = loss.data
            phi_vec[:,epoch, :, :] = phi.mean(dim=3).detach()
            plot_vec[:,epoch, :, :] = self(torch.linspace(0,1,n_sample*timestamps.shape[0])).detach().permute([0,2,1])
            result_vec[:,epoch, :, :] = result.detach().permute([0,2,1])
            # print(torch.cat([self.points, odeint(self.RHS, torch.stack([self.points[:,-1], self.tangent_vectors[:,-1]], dim=1).unsqueeze(1), t=torch.tensor([0., 1.]))[-1,:,-1, 0].unsqueeze(1)], dim=1).detach().shape)
            # print(self.points.shape)
            points_vec[:,epoch, :, :] = torch.cat([self.points, odeint(self.RHS, torch.stack([self.points[:,-1], self.tangent_vectors[:,-1]], dim=1).unsqueeze(1), t=torch.tensor([0., 1.]))[-1,:,-1, 0].unsqueeze(1)], dim=1).detach()

            self.points.grad.zero_()
            self.tangent_vectors.grad.zero_()
        idx_best = torch.argmin(nrj[:,-1])
        print(constraint_loss.data)
        return  phi[idx_best], nrj[idx_best], phi_vec[idx_best], plot_vec[idx_best], points_vec[idx_best], result_vec[idx_best]
