# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:09:41 2023

@author: blaville
"""

import torch
# import bezier as bz
from tqdm import tqdm
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

torch.manual_seed(10)


def de_casteljau(points, t):
    """
    Computes the De Casteljau algorithm for a set of control points.
    """
    # points = torch.tensor(points, dtype=torch.float32)
    list_points = [points.unsqueeze(2)]
    while list_points[-1].shape[0] > 1:
        # print(list_points[-1].shape)
        list_points.append(
            (1 - t) * list_points[-1][:-1] + t * list_points[-1][1:])

    return list_points[-1]


def derivative_de_casteljau(points, t):
    # print((points[1:]-points[:-1]).shape)
    return points.shape[1] * de_casteljau(points[1:]-points[:-1], t)


def RS_relaxed_norm(x, v, eps=.05, xi=.1):
    assert(v.ndim == 2 and v.shape[0] == 3)
    # assert v.shape[0] == 3
    theta = torch.remainder(x[2], np.pi)
    return((torch.cos(theta)*v[0] + torch.sin(theta)*v[1])**2
           + (1/(eps**2))*(-torch.sin(theta)*v[0] + torch.cos(theta)*v[1])**2
           + (xi**2)*v[2]**2)


if __name__ == '__main__':

    n = 128
    d = 3
    nc = 16

    sigma = 0.01
    dom_size = 64 
    x = torch.linspace(-1, 1, dom_size)
    X, Y = torch.meshgrid(x, x)

    # control points
    points = torch.zeros([nc, d])
    points.requires_grad = True

    timestamps = torch.arange(0, 1, 1/n)
    y_1 = .5 * torch.stack([-0.8+2*timestamps, -(0.8-2*timestamps)])
    y_2 = .5 * torch.stack([-0.8+2*timestamps, 0.8-2*timestamps])
    y_0 = y_1
    y_k = [y_1, y_2]
    
    phi_1 = torch.exp(-((y_1[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                     ** 2 + (y_1[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)
    phi_2 = torch.exp(-((y_2[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                     ** 2 + (y_2[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

    ponder = 0.9
    phi0 = phi_1 + ponder * phi_2

    noise = 1e-1
    phi0 += torch.normal(0, noise, size=phi0.shape)

    plt.figure(2)
    plt.clf()
    plt.imshow(phi0.mean(dim=2).detach(), cmap='bone', origin='lower')
    plt.show()

    n_epoch = 120
    nrj = torch.zeros(n_epoch)
    result_vec = torch.zeros(n_epoch, 3, n)
    points_vec = torch.zeros(n_epoch, nc, 3)
    phi_vec = torch.zeros(n_epoch, dom_size, dom_size)

    regul = .0001
    optimizer = torch.optim.Adam([points], lr=1e-1)
    for i in tqdm(range(n_epoch),
                  desc=f'Computing UFW for {dom_size} x {dom_size}'):

        optimizer.zero_grad()
        # Evaluate the Bézier curve at t=0.5
        result = de_casteljau(
            points, t=timestamps.unsqueeze(0).unsqueeze(0)).squeeze()
        result_d = derivative_de_casteljau(points, t=torch.arange(
            0, 1, 1/(n)).unsqueeze(0).unsqueeze(0)).squeeze()
        phi = torch.exp(-((result[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))**2 + (
            result[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

        a = ((phi-phi0)**2).mean() + regul * \
            (RS_relaxed_norm(result, result_d, eps=0.05, xi=1)).mean()

        nrj[i] = a.data

        a.backward()
        optimizer.step()

        phi_vec[i, :] = phi.mean(dim=2).detach()
        result_vec[i, :] = result.detach()
        points_vec[i, :] = points.detach()

        # plt.figure(0)
        # plt.clf()
        # plt.scatter(result[0,:].detach(), result[1,:].detach(), c='r')
        # plt.scatter(points[:,0].detach(), points[:,1].detach(), c='b')
        # plt.ylim((-1.5,1.5))
        # plt.xlim((-1.5,1.5))
        # plt.axis('equal')
        # plt.show()

        # plt.figure(1)
        # plt.clf()
        # plt.imshow(phi.mean(dim=2).detach(), extent=(-1,1,-1,1), origin='lower')
        # plt.scatter(y_0[0,:].detach(), y_0[1,:].detach(), c=timestamps, cmap='bone')
        # plt.scatter(result[0,:].detach(), result[1,:].detach(), c='r')
        # plt.scatter(points[:,0].detach(), points[:,1].detach(), c='b')

    # Energy
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.plot(points[:, 0].detach(), points[:, 1].detach(), 'x', c='b',
             label='control pts')
    plt.plot(y_0[0, :].detach(), y_0[1, :].detach(), c='purple',
             label='ground-truth')
    plt.plot(result[0, :].detach(), result[1, :].detach(), '--', c='green',
             label='reconstruction')
    plt.legend()
    plt.subplot(122)
    plt.plot(torch.linspace(0, n_epoch-1, n_epoch), nrj, '.--')
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Energy', fontsize=20)
    plt.grid()
    plt.show()



def UFW(acquis_0, iteration, n_epoch=120, regul=.0001):
    n = 128
    d = 3
    dom_size = 64
    nc = 16

    y = acquis_0

    nrj = torch.zeros(iteration, n_epoch)
    phi = torch.zeros(size=acquis_0.shape)
    result_vec = torch.zeros(iteration, n_epoch, 3, n)
    points_vec = torch.zeros(iteration, n_epoch, nc, 3)
    phi_vec = torch.zeros(iteration, n_epoch, dom_size, dom_size)

    for k in tqdm(range(iteration),
                  desc='Computing UFW',
                  position=0, leave=True):
    
        # control points
        points = torch.zeros([nc, d])
        points.requires_grad = True
    
        optimizer = torch.optim.Adam([points], lr=1e-1)
        for epoch in tqdm(range(n_epoch),desc=f'Computing UFW step {k}',
                      position=0,
                      leave=False):

            optimizer.zero_grad()
            # Evaluate the Bézier curve at t=0.5
            result = de_casteljau(
                points, t=timestamps.unsqueeze(0).unsqueeze(0)).squeeze()
            result_d = derivative_de_casteljau(points, t=torch.arange(
                0, 1, 1/(n)).unsqueeze(0).unsqueeze(0)).squeeze()
            phi = torch.exp(-((result[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))**2 + (
                result[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

            a = ((phi-y)**2).mean() + regul * \
                (RS_relaxed_norm(result, result_d, eps=0.05, xi=1)).mean()

            nrj[k, epoch] = a.data

            a.backward()
            optimizer.step()

            phi_vec[k, epoch, :, :] = phi.mean(dim=2).detach()
            result_vec[k, epoch, :, :] = result.detach()
            points_vec[k, epoch, :, :] = points.detach()
          
        del a
        y -= phi.detach()

    return(result_vec, points_vec, phi_vec, nrj)


plt.figure()
plt.imshow(phi0.mean(dim=2), cmap='bone', origin='lower', 
           extent=(-1,1,-1,1))

(result_vec, points_vec, phi_vec, nrj) = UFW(phi0.detach().clone(), 2)


for k in range(result_vec.shape[0]):
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.imshow(phi_vec[k, -1 ,: ,:], cmap='bone', 
               extent=(-1, 1, -1, 1), origin='lower')
    plt.plot(points_vec[k, -1, :, 0], points_vec[k, -1, :, 1], 'x', c='b',
              label='control pts')
    plt.plot(phi_vec[k, :, 0, :], phi_vec[k, :, 1, :], c='orange',
             label='ground-truth')
    plt.plot(result_vec[k, -1, 0, :], result_vec[k, -1, 1, :], '--', c='red',
             label='reconstruction')
    # plt.legend()
    plt.subplot(122)
    plt.plot(torch.linspace(0, n_epoch-1, n_epoch), nrj[k,:], '.--')
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Energy', fontsize=20)
    plt.grid()
    plt.show()


plt.figure()
plt.imshow(phi0.mean(dim=2), cmap='bone', origin='lower', 
           extent=(-1,1,-1,1))
for k in range(result_vec.shape[0]):
    # plt.plot(points_vec[k, -1, :, 0], points_vec[k, -1, :, 1], 'x', c='b',
    #           label='control pts')
    plt.plot(y_k[k][0, :], y_k[k][1, :], c='orange',
             label='ground-truth')
    plt.plot(result_vec[k, -1, 0, :], result_vec[k, -1, 1, :], '--', c='red',
             label='reconstruction')
    if k==0: plt.legend()
# plt.legend()
plt.show()


#%%

def gif_pile(number, ph, y, video='mp4', title=None):
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(ph[:,:,0], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')
    
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # fig.colorbar(cont_pile, cax=cax)
    # plt.tight_layout()

    def animate(k):
        if k >= number:
            # On fige l'animation pour faire une pause à la fin
            return
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(ph[:,:,k], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')
        ax.plot(y_1[0, :k], y_1[1, :k], '--', c='orange', linewidth=4.0, 
                label='Ground-truth')
        ax.plot(y_2[0, :k], y_2[1, :k], '--', c='orange', linewidth=4.0)
        ax.plot(y_1[0, k], y_1[1, k], 'x', c='orange', linewidth=5.0)
        ax.plot(y_2[0, k], y_2[1, k], 'x', c='orange', linewidth=5.0)
        ax.set_xlabel('X', fontsize=25)
        ax.set_ylabel('Y', fontsize=25)
        ax.set_title(f'Iteration {k}', fontsize=30)
        ax.legend()
        # ax.legend(loc=1, fontsize=20)
        # ax.tight_layout()

    anim = FuncAnimation(fig, animate, interval=50,
                         frames=number+30,
                         blit=False)
    plt.draw()

    if title is None:
        title = 'anim-acquis-2d'
    elif isinstance(title, str):
        title = '' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


gif_pile(n, phi0, y_0, title='cross_acquis')

#%% Gif reconstruction stack animation

it = 1

def gif_reconstruction(n_epoch, ph, res, pts, y, video='mp4', title=None):
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    # ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.imshow(ph[it,0,:,:], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # fig.colorbar(cont_pile, cax=cax)
    # plt.tight_layout()

    def animate(k):
        if k >= n_epoch:
            # On fige l'animation pour faire une pause à la fin
            return
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(ph[it, k, :, :], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')
        ax.plot(pts[it,k,:, 0], pts[0,k, :, 1], 'x', c='b', label='control pts')
        ax.plot(y_k[it][0,:], y_k[it][1,:], '--', c='orange', linewidth=2.0, 
                label='ground truth')
        ax.plot(res[it, k, 0, :], res[0, k, 1, :], c='red', linewidth=2.0, 
                label='reconstruction')
        ax.set_xlabel('X', fontsize=25)
        ax.set_ylabel('Y', fontsize=25)
        ax.set_title(f'Iteration {k}', fontsize=30)
        ax.legend(loc=2)
        # ax.legend(loc=1, fontsize=20)
        # ax.tight_layout()

    anim = FuncAnimation(fig, animate, interval=70,
                         frames=n_epoch+15,
                         blit=False)
    plt.draw()

    if title is None:
        title = 'cross_recons' + '_' + str(it)
    elif isinstance(title, str):
        title = '' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


gif_reconstruction(n_epoch, phi_vec, result_vec, points_vec, y_0)