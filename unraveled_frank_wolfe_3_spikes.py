# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:09:35 2023

@author: blaville
"""

import argparse
import wandb 

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Discretization_curves import BezierCurveModule, PolygonalCurveModule, PiecewiseGeodesic_RS, RHS_geodesic_RS_2D, RHS_geodesic_RS_parallel_2D

torch.manual_seed(10)

def RS_relaxed_norm(x, v, eps=.05, xi=.1):
    assert(v.ndim == 2 and v.shape[0] == 3)
    # assert v.shape[0] == 3
    theta = torch.remainder(x[2], torch.pi)
    return((torch.cos(theta)*v[0] + torch.sin(theta)*v[1])**2
           + (1/(eps**2))*(-torch.sin(theta)*v[0] + torch.cos(theta)*v[1])**2
           + (xi**2)*v[2]**2)

def build_3_spikes_acquisition(nt,sigma, dom_size, noise_level=0.6):
    timestamps = torch.linspace(0, 1, nt)
    reverse_time = torch.arange(0, 1, 1/(nt-10))
    corner_time = torch.arange(0, 1, 1/10)
    y_1 = .6 * torch.stack([-1+2*timestamps, -(1-2*timestamps)])
    y_21 = .5 * torch.stack([torch.cos(1.2 * torch.pi * reverse_time)+0.5,
                            torch.sin(1.2 * torch.pi * reverse_time) - 0.9])
    y_22 = 1.*torch.stack([0.48+0.3*corner_time, (-0.61+0.15*corner_time)])
    y_2 = torch.cat([y_22, y_21], dim=1).flip(1)
    y_3 = .5 * torch.stack([torch.cos(0.8 * torch.pi * timestamps-2)-1.0,
                            torch.sin(0.8 * torch.pi * timestamps-2)+1.0])
    y_0 = y_1
    y_k = [y_1, y_2, y_3]

    x = torch.linspace(-1, 1, dom_size)
    X, Y = torch.meshgrid(x, x, indexing='ij')

    phi_1 = torch.exp(-((y_1[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                        ** 2 + (y_1[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)
    phi_2 = torch.exp(-((y_2[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                        ** 2 + (y_2[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)
    phi_3 = torch.exp(-((y_3[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                        ** 2 + (y_3[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

    ponder = 1.
    phi0 = phi_1 + ponder * phi_2 + ponder * phi_3

    noise = noise_level*phi0.max()
    phi0 += torch.normal(0, noise, size=phi0.shape)

    return phi0, y_k

def build_2_spikes_acquisition(nt,sigma, dom_size, noise_level=0.6):
    timestamps = torch.arange(0, 1, 1/nt)
    y_1 =  torch.stack([-0.8+1.6*timestamps, -(0.8-1.6*timestamps)])
    y_2 =  torch.stack([-0.8+1.6*timestamps, 0.8-1.6*timestamps])
    y_0 = y_1
    y_k = [y_1, y_2]

    x = torch.linspace(-1, 1, dom_size)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    phi_1 = torch.exp(-((y_1[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                     ** 2 + (y_1[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)
    phi_2 = torch.exp(-((y_2[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                     ** 2 + (y_2[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

    ponder = 1.
    phi0 = phi_1 + ponder * phi_2

    noise = noise_level*phi0.max()
    phi0 += torch.normal(0, noise, size=phi0.shape)

    return phi0, y_k

def figures_UFW(result_vec, plot_vec, points_vec, phi_vec, nrj, y_k, phi0):
    for k in range(plot_vec.shape[0]):
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.imshow(phi_vec[k, -1, :, :], cmap='bone',
                extent=(-1, 1, -1, 1), origin='lower')
        plt.plot(points_vec[k, -1, :, 0], points_vec[k, -1, :, 1], 'x', c='b',
                label='control pts')
        # plt.plot(phi_vec[k, -1, 0, :], phi_vec[k, -1, 1, :], c='orange',
        #         label='ground-truth')
        plt.plot(plot_vec[k, -1, 0, :], plot_vec[k, -1, 1, :], '--', c='red',
                label='reconstruction')
        plt.legend()
        plt.subplot(122)
        plt.plot(torch.linspace(0, len(nrj[k, :])-1, len(nrj[k, :])),
                nrj[k, :], '.--')
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Energy', fontsize=20)
        plt.grid()
        plt.show(block=False)

    plt.figure()
    plt.imshow(phi0.mean(dim=2), cmap='bone', origin='lower',
            extent=(-1, 1, -1, 1))
    for k in range(plot_vec.shape[0]):
        # plt.plot(points_vec[k, -1, :, 0], points_vec[k, -1, :, 1], 'x', c='b',
        #           label='control pts')
        
        if k == 0:
            plt.plot(y_k[k][0, :], y_k[k][1, :], 
                 c='orange',
                label='ground-truth')
        else:
            plt.plot(y_k[k][0, :], y_k[k][1, :], 
                 c='orange')

    for k in range(plot_vec.shape[0]):
        plt.plot(plot_vec[k, -1, 0, :], plot_vec[k, -1, 1, :], '--', c='red',
                label='reconstruction')
        # plt.scatter(plot_vec[k, -1, 0, :], plot_vec[k, -1, 1, :], c='red',
        #         label='reconstruction', s=1)
        if k == 0:
            plt.legend()
    # plt.legend()
    plt.show(block=True)
    # plt.pause(0.1)

    wandb.log({'reconstruction': wandb.Image(plt)})

    return None

def UFW(acquis_0, iteration, nc, sigma, n_epoch=150, regul=.00001, geom='euclidean', method='bezier', lr=1e-2, epsilon=1., xi=1., n_start=1, n_sample=8):
    n = acquis_0.shape[-1]

    if method == 'exponential_RS':
        d = 3
    else:
        d = 2
    
    dom_size = acquis_0.shape[0]

    y = acquis_0

    timestamps = torch.linspace(0, 1, n)

    nrj = torch.zeros(iteration, n_epoch)
    phi = torch.zeros(size=acquis_0.shape)
    result_vec = torch.zeros(iteration, n_epoch, d, n)
    points_vec = torch.zeros(iteration, n_epoch, nc, d)
    phi_vec = torch.zeros(iteration, n_epoch, dom_size, dom_size)
    plot_vec = torch.zeros(iteration, n_epoch, d, n_sample*n)

    x = torch.linspace(-1, 1, dom_size)
    X, Y = torch.meshgrid(x, x, indexing='ij')

    if method=='exponential_RS': #Feed the same compiled RHS function to all diracs to avoid recompiling it at each iteration
        RHS = torch.compile(RHS_geodesic_RS_2D(epsilon=epsilon, xi=xi))
        RHS_optim = torch.compile(RHS_geodesic_RS_parallel_2D(epsilon=epsilon, xi=xi))

    for k in tqdm(range(iteration),
                  desc='Computing UFW',
                  position=0, leave=True):

        if method=='bezier':
            Curve = BezierCurveModule(nc, n_start, w=y, timestamps=timestamps)
        elif method=='polygonal':
            Curve = PolygonalCurveModule(nc,n_start, w=y, timestamps=timestamps)
        elif method=='exponential_RS':
            Curve = PiecewiseGeodesic_RS(n_points=nc, n_start=n_start, epsilon=epsilon, xi=xi, RHS=RHS, RHS_optim=RHS_optim, w=y, timestamps=timestamps)
        else:
            raise ValueError('Unknown method')

        operator = lambda x: torch.exp(-((x[...,1].unsqueeze(1).unsqueeze(1)-X.unsqueeze(2).unsqueeze(0))**2 
                                         + (x[...,0].unsqueeze(1).unsqueeze(1)-Y.unsqueeze(2).unsqueeze(0))**2)/sigma)


        phi, nrj[k], phi_vec[k], plot_vec[k], points_vec[k], result_vec[k] = Curve.fit(y, operator, timestamps, n_epoch, lr=lr, regul=regul)
        # optimizer.zero_grad()
        # result = Curve(timestamps).squeeze()

        # phi = torch.exp(-((result[:,1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))**2 + (
        #     result[:,0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)
        
        # a = ((-y)*phi).mean() + regul*Curve.Action(geometry=geom, epsilon=epsilon, xi=xi)
        

        # a.backward()
        # optimizer.step()

        # nrj[k, epoch] = a.data
        # phi_vec[k, epoch, :, :] = phi.mean(dim=2).detach()
        # result_vec[k, epoch, :, :] = result.detach().T
        # points_vec[k, epoch, :, :] = Curve.get_control_points().detach()

        # del a, points
        y -= phi.detach()
        
        # plt.imshow(phi.max(dim=2).values.detach(), cmap='bone', origin='lower')
        # plt.pause(0.1)
    # wandb.log({'energy 1': nrj[0, :]})
    # wandb.log({'energy 2': nrj[1, :]})
    # wandb.log({'energy 3': nrj[2, :]})
    return(points_vec, phi_vec, nrj, plot_vec, result_vec)

def evaluate_solution(estimated, y_k):

    return ((estimated[:,None,:2,:] - torch.stack(y_k)[None,:,:,:])**2).mean(dim=(2,3)).min(dim=0).values.sum()

def main(args):
    wandb.login()
    wandb.init(name = 'fine_tuning_' + str(args.nb_pics) + '_spikes', config = args)

    n = args.n
    nc = args.nc
    dom_size = 16

    nb_pics = args.nb_pics

    sigma = (0.2)**2

    if nb_pics==3:
        phi0, y_k = build_3_spikes_acquisition(n, sigma, dom_size, args.noise)
    elif nb_pics==2:
        phi0, y_k = build_2_spikes_acquisition(n, sigma, dom_size, args.noise)
    else:
        raise ValueError('Unknown number of spikes')

    epochs = args.epochs
    regularization = args.regularization

    method = args.method

    (points_vec, phi_vec, nrj, plot_vec, result_vec) = UFW(phi0.detach().clone(), nb_pics, nc, sigma,
                                                n_epoch=epochs, regul=regularization, geom=args.geometry, method=method,lr=args.lr, epsilon=args.epsilon, xi=args.xi, n_start=args.n_start)
    figures_UFW(result_vec, plot_vec, points_vec, phi_vec, nrj, y_k, phi0)

    wandb.log({'evaluation': evaluate_solution(result_vec[:,-1,:], y_k)})

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Training U-Net model for segmentation of brain MRI")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train (default: 100)")
    parser.add_argument("--regularization", type=float, default=0.1, help="regularization parameter")
    parser.add_argument("--n", type=int, default=21, help="number of points in the acquisition")
    parser.add_argument("--nc", type=int, default=4, help="number of control points")
    parser.add_argument("--geometry", type=str, default='euclidean', help="geometry of the space")
    parser.add_argument("--method", type=str, default='exponential_RS', help="parametrisation de la courbe")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--nb_pics", type=int, default=2, help="number of spikes in the acquisition")
    parser.add_argument("--epsilon", type=float, default=.5, help="epsilon parameter for the RS geometry")
    parser.add_argument("--xi", type=float, default=1., help="xi parameter for the RS geometry")
    parser.add_argument("--n_start", type=int, default=64, help="Number of curves for multistart")
    parser.add_argument("--noise", type=float, default=.4, help="Number of curves for multistart")

    args, unknown = parser.parse_known_args()

    main(args)


#%% Generate the animation of the acquisition stack

# def gif_pile(number, ph, y, video='mp4', title=None):
#     # ax, fig = plt.subplots(figsize=(10,10))
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111)
#     ax.imshow(ph[:, :, 0], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')

#     # divider = make_axes_locatable(ax)
#     # cax = divider.append_axes("right", size="5%", pad=0.15)
#     # fig.colorbar(cont_pile, cax=cax)
#     # plt.tight_layout()

#     def animate(k):
#         if k >= number:
#             # On fige l'animation pour faire une pause à la fin
#             return
#         ax.clear()
#         ax.set_aspect('equal', adjustable='box')
#         ax.imshow(ph[:, :, k], cmap='bone',
#                   extent=(-1, 1, -1, 1), origin='lower')
#         for i in range(len(y_k)):
#             y_i = y_k[i]
#             if i==0:
#                 ax.plot(y_i[0, :k], y_i[1, :k], '--', c='orange', linewidth=4.0,
#                         label='Ground-truth')
#             else:
#                 ax.plot(y_i[0, :k], y_i[1, :k], '--', c='orange', 
#                         linewidth=4.0)
#             ax.plot(y_i[0, k], y_i[1, k], 'x', c='orange', linewidth=10.0)
#         ax.set_xlabel('X', fontsize=25)
#         ax.set_ylabel('Y', fontsize=25)
#         ax.set_title(f'Iteration {k}', fontsize=30)
#         ax.legend()
#         # ax.legend(loc=1, fontsize=20)
#         # ax.tight_layout()

#     anim = FuncAnimation(fig, animate, interval=50,
#                          frames=number+30,
#                          blit=False)
#     plt.draw()

#     if title is None:
#         title = 'anim-acquis-2d'
#     elif isinstance(title, str):
#         title = '' + title
#     else:
#         raise TypeError("You ought to give a str type name for the video file")

#     if video == "mp4":
#         anim.save(title + '.mp4')
#     elif video == "gif":
#         anim.save(title + '.gif')
#     else:
#         raise ValueError('Unknown video format')
#     return fig


#     gif_pile(n, phi0, y_0, title='cross_acquis')


