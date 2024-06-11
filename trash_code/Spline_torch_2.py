
import torch
import torch.nn as nn
import scipy as sc
import numpy as np
# import bezier as bz
from tqdm import tqdm


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
    return (torch.cos(theta)*v[0]+torch.sin(theta)*v[1])**2 + (1/(eps**2))*(-torch.sin(theta)*v[0]+torch.cos(theta)*v[1])**2 + (xi**2)*v[2]**2


if __name__ == '__main__':

    n = 128
    d = 3
    dom_size = 64
    # C = np.random.randn(d,n)

    nc = 16

    # control points
    points = torch.zeros([nc, d])
    points.requires_grad = True

    optimizer = torch.optim.Adam([points], lr=1e-1)

    timestamps = torch.arange(0, 1, 1/n)
    y_0 = .5*torch.stack([torch.cos(np.pi*timestamps),
                         torch.sin(np.pi*timestamps)])

    sigma = 0.01

    x = torch.linspace(-1, 1, dom_size)
    X, Y = torch.meshgrid(x, x)
    phi0 = torch.exp(-((y_0[1].unsqueeze(0).unsqueeze(0)-X.unsqueeze(2))
                     ** 2 + (y_0[0].unsqueeze(0).unsqueeze(0)-Y.unsqueeze(2))**2)/sigma)
    
    noise = 5e-1
    phi0 += torch.normal(0, noise, size=phi0.shape)

    plt.figure(2)
    plt.clf()
    plt.imshow(phi0.mean(dim=2).detach(), cmap='bone', origin='lower')
    plt.show()

    n_epoch = 150
    nrj = torch.zeros(n_epoch)
    result_vec = torch.zeros(n_epoch, 3, n)
    points_vec = torch.zeros(n_epoch, nc, 3)
    phi_vec = torch.zeros(n_epoch, dom_size, dom_size)
    
    regul = .0001
    for i in tqdm(range(n_epoch), 
                  desc=f'Computing CSFW for {dom_size} x {dom_size}'):

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


#%% Gif acquisition stackanimation

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
        ax.plot(y[0, :k], y[1, :k], '--', c='purple', linewidth=2.0, 
                label='Ground-truth')
        ax.plot(y[0, k], y[1, k], 'x', c='purple', linewidth=3.0)
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


gif_pile(n, phi0, y_0, title='acquis-half-circle')

#%% Gif reconstruction stack animation

def gif_reconstruction(n_epoch, ph, res, pts, y, video='mp4', title=None):
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    # ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.imshow(ph[0,:], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')
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
        ax.imshow(ph[k,:], cmap='bone', extent=(-1, 1, -1, 1), origin='lower')
        ax.plot(pts[k,:, 0], pts[k, :, 1], 'x', c='b', label='control pts')
        ax.plot(y[0,:], y[1,:], '--', c='orange', linewidth=3.0, 
                label='ground truth')
        ax.plot(res[k, 0, :], res[k, 1, :], c='red', linewidth=3.0, 
                label='reconstruction')
        ax.set_xlabel('X', fontsize=25)
        ax.set_ylabel('Y', fontsize=25)
        ax.set_title(f'Iteration {k}', fontsize=30)
        ax.legend()
        # ax.legend(loc=1, fontsize=20)
        # ax.tight_layout()

    anim = FuncAnimation(fig, animate, interval=70,
                         frames=n_epoch+15,
                         blit=False)
    plt.draw()

    if title is None:
        title = 'anim-pile-2d'
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


gif_reconstruction(n_epoch, phi_vec, result_vec, points_vec, y_0,
                   title='result-half-circle')


#%%
fig = plt.figure(3)
ax = fig.add_subplot(projection='3d')
ax.scatter(result[0].detach(), result[1].detach(), torch.remainder(
    result[2].detach(), np.pi), c='r', marker='o', s=50)
ax.scatter(points[:, 0].detach(), points[:, 1].detach(), torch.remainder(
    points[:, 2].detach(), np.pi), c='b', marker='o', s=50)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, np.pi)

#%%

plt.figure(4)
plt.clf()
plt.plot(timestamps, torch.angle(
    result_d[0]+1.j*result_d[1]).detach().remainder(np.pi), c='g')
plt.plot(timestamps, result[2].detach().remainder(np.pi), c='r')
plt.title('Angle on $\mathbb{S}\,^{d-1}$', fontsize=20)
plt.show()

#%%

v_0 = np.pi*.5 * \
    torch.stack([-torch.sin(np.pi*timestamps),
                torch.cos(np.pi*timestamps)])

y_0t = torch.cat([y_0, torch.remainder(
    (np.pi*timestamps + np.pi/2).unsqueeze(0), np.pi)])
# v_0t = torch.cat([v_0, torch.remainder((np.pi*timestamps+np.pi/2).unsqueeze(0),np.pi)])
v_0t = torch.cat([v_0, torch.angle(v_0[0]+1.j*v_0[1]).unsqueeze(0)])

print(RS_relaxed_norm(y_0t, v_0t, eps=.1, xi=0))


# to check with a planar curve if RS model is OK
resultt = torch.cat([result[:2], torch.angle(
    result_d[0]+1.j*result_d[1]).unsqueeze(0)])
print(RS_relaxed_norm(resultt, result_d, eps=1e-4, xi=0).sum())
print(RS_relaxed_norm(resultt, result_d, eps=1e8, xi=0).sum())
