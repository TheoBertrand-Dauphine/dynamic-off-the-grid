
import torch
import torch.nn as nn
import scipy as sc
import numpy as np
# import bezier as bz
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# class BezierCurve(nn.Module):

#     def __init__(self, C):
#         # super(UNet,self).__init__()
#         n,d = C.shape
#         t = np.linspace(0,1,n)
#         # sc_spline  = sc.interpolate.BSpline(t, C, n)
#         self.bz_curve = bz.Curve.from_nodes(C)

#         def __repr__(self):
#             return 'Bezier curve defined from control points, nn.Module object'


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
    return points.shape[0]*de_casteljau(points[1:]-points[:-1], t)


if __name__ == '__main__':

    n = 128
    d = 2
    # C = np.random.randn(d,n)

    x = torch.linspace(0, 1, 64)

    # C = np.stack([x, np.sin(2*np.pi*x)])

    # Define control points
    # points = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], requires_grad=True)
    points = torch.stack([0.*x, 0.*torch.sin(2*np.pi*x)]
                         ).T.requires_grad_(True)

    optimizer = torch.optim.Adam([points], lr=1e-1)

    timestamps = torch.arange(0, 1, 1/(n))
    y_0 = .5*torch.stack([torch.cos(2*np.pi*timestamps),
                         torch.sin(2*np.pi*timestamps)])

    sigma = 0.01

    x = torch.linspace(-1, 1, 64)
    X, Y = torch.meshgrid(x, x)
    phi0 = torch.exp(-((y_0[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                     ** 2 + (y_0[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

    plt.figure(2)
    plt.clf()
    plt.imshow(phi0.mean(dim=2).detach())
    plt.show()

    n_epoch = 100
    nrj = torch.zeros(n_epoch)
    result_vec = torch.zeros(n_epoch, 2, n)
    points_vec = torch.zeros(n_epoch, 64, 2)
    y_0_vec = torch.zeros(n_epoch, 2, n)
    phi_vec = torch.zeros(n_epoch, 64, 64)
    for i in tqdm(range(n_epoch)):
        optimizer.zero_grad()

        # Evaluate the Bézier curve at t=0.5
        result = de_casteljau(points, 
                              t=timestamps.unsqueeze(0).unsqueeze(0)).squeeze()
        result_d = derivative_de_casteljau(points, t=torch.arange(
            0, 1, 1/(n)).unsqueeze(0).unsqueeze(0)).squeeze()
        phi = torch.exp(-((result[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))**2 + (
            result[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)

        a = ((phi-phi0)**2).mean()  # + .0001*(result_d**2).mean()

        nrj[i] = a.data

        a.backward()
        optimizer.step()

        phi_vec[i,:] = phi.mean(dim=2).detach()
        result_vec[i,:] = result.detach()
        points_vec[i,:] = points.detach()
        y_0_vec[i,:] = y_0.detach()

        # plt.figure(1)
        # plt.clf()
        # plt.imshow(phi.mean(dim=2).detach(), extent=(-1, 1, -1, 1), cmap='hot')
        # plt.plot(result[0, :].detach(), result[1, :].detach(), c='green')
        # plt.plot(points[:, 0].detach(), points[:, 1].detach(), 'x', c='b')
        # plt.plot(y_0[0, :].detach(), y_0[1, :].detach(), c='pink')

        # plt.show()

        # plt.pause(0.05)
        # plt.close('all')


#%%

plt.figure(figsize=(14, 5))
plt.subplot(121)
# plt.imshow(phi.mean(dim=2).detach(), extent=(-1,1,-1,1), cmap='hot')
plt.plot(points[:, 0].detach(), points[:, 1].detach(), 'x', c='b',
         label='control pts')
plt.plot(y_0[0, :].detach(), y_0[1, :].detach(), c='orange',
         label='ground-truth')
plt.plot(result[0, :].detach(), result[1, :].detach(), '--', c='red',
         label='reconstruction')
plt.legend()
plt.subplot(122)
plt.plot(torch.linspace(0, n_epoch-1, n_epoch), nrj)
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Energy', fontsize=20)
plt.grid()
plt.show()


def gif_pile(n_epoch, ph, res, pts, y, video='mp4', title=None):
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    # ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.imshow(ph[0,:], cmap='bone', extent=(-1, 1, -1, 1))
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
        ax.imshow(ph[k,:], cmap='bone', extent=(-1, 1, -1, 1))
        ax.plot(y[k, 0, :], y[k, 1, :], c='orange', label='ground-truth', 
                linewidth=2)
        ax.plot(res[k, 0, :], res[k, 1, :], c='red',label='reconstruction', 
                linewidth=2)
        ax.plot(pts[k,:, 0], pts[k, :, 1], 'x', c='b', label='control points', 
                linewidth=2)
        ax.set_xlabel('X', fontsize=25)
        ax.set_ylabel('Y', fontsize=25)
        ax.set_title(f'Iteration {k}', fontsize=30)
        ax.legend()
        # ax.legend(loc=1, fontsize=20)
        # ax.tight_layout()

    anim = FuncAnimation(fig, animate, interval=70,
                         frames=n_epoch+3,
                         blit=False)
    plt.draw()

    if title is None:
        title = 'anim-pile-2d'
    elif isinstance(title, str):
        title = 'fig/anim/' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


gif_pile(n_epoch, phi_vec, result_vec, points_vec, y_0_vec)


#%%


phi = torch.exp(-((result[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                ** 2 + (result[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/0.01)
