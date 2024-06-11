import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp

n = 64
x = np.linspace(-1,1,n)

[X,Y] = np.meshgrid(x,x)

sigma = 0.5
# k_pot = (1*np.exp(-(X**2+Y**2)/(sigma**2)) + 1)
k_pot = 0.*((X**2+Y**2)<.5**2)+1.

# k_pot = (X>0) + 1


# assert (k_pot == k_pot_vec.reshape([n,n])).all()




def proj_pot(p, pot):

    out = np.copy(p)
    norm_p = np.sqrt((out**2).sum(axis=0))

    # print(norm_p.mean())

    out[:, norm_p[:,0]>pot[:,0],:] = pot[norm_p[:,0]>pot[:,0],:]*out[:,norm_p[:,0]>pot[:,0],:]/norm_p[norm_p[:,0]>pot[:,0],:]
    # print((norm_p[:,0]>pot[:,0]).sum())
    # print((out**2).sum(axis=0).max())

    return out



def primal_dual_iterations(k_pot,seed):
    derivx = (-sp.diags(np.concatenate([np.ones(n-1),np.zeros(1)]), offsets=0) + sp.diags(np.ones(n-1), offsets=1))/(2/n)

    Dx = sp.kron(np.eye(n),derivx)

    Dy = sp.kron(derivx, np.eye(n))

    Div1 = -Dx.T

    Div2 = -Dy.T

    u = np.zeros([n,n])
    u_vec = u.reshape([-1,1])
    u_vec_bar = u.reshape([-1,1])

    phi = np.stack([Dx.dot(u_vec), Dy.dot(u_vec)])

    eta = 0.2*(2/n)
    tau = 0.2*(2/n)

    k_pot_vec = k_pot.reshape([-1,1])

    epochs = 100000
    plt.figure(0)
    for i in range(epochs):
        u_vec_old = (u_vec)

        # primal step
        psi = phi + eta*np.stack([Dx.dot(u_vec_bar), Dy.dot(u_vec_bar)])
        phi = psi - eta*proj_pot(psi/eta,k_pot_vec)

        # dual step
        v = (u_vec + tau*(Div1.dot(phi[0]) + Div2.dot(phi[1])))
        # partie prox
        u_vec = (v + tau)
        u_vec[seed,0] = 0

        # extragradient
        u_vec_bar =(2*u_vec - u_vec_old)

        if i%10000==0:
            plt.clf()
            plt.imshow(np.ma.masked_where(k_pot.reshape([n,n])>10,u_vec_bar.reshape([n,n])), cmap='cool', extent=[-1,1,-1,1], origin='lower')
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.001)

        if np.sqrt(((u_vec-u_vec_old)**2).mean())/np.sqrt((u_vec**2).mean())<1e-6 and i>10:
            print('break at iteration',i)
            break
    return u_vec_bar


phi1 = np.ones([n,n])
mask1 = np.zeros([n,n])
seed1 = np.zeros([1,2])

u_vec_bar1 = primal_dual_iterations(phi1, seed = [(n//2)*n+(n//2),(n//4)*n+(n//4)])

D1 = u_vec_bar1.reshape([n,n])

plt.figure(110)
plt.imshow((1-1.*mask1)*D1,extent=[-1,1,-1,1], cmap='cool', origin='lower')
plt.colorbar()
plt.scatter(seed1[:,1],seed1[:,0], c='r')
plt.savefig('../FMECNN/fig_fm/distance_pm1')

plt.figure(111)
plt.imshow((1-1.*mask1)*np.sin(32*D1), cmap='cool', extent=[-1,1,-1,1], origin='lower')
plt.colorbar()
plt.scatter(seed1[:,1],seed1[:,0], c='r')
plt.savefig('../FMECNN/fig_fm/distance_pm_sin1')


plt.figure(112)
plt.imshow(phi1, extent=[-1,1,-1,1], origin='lower')
plt.colorbar()
plt.scatter(seed1[:,1],seed1[:,0], c='r')
plt.savefig('../FMECNN/fig_fm/potential_pm1')

plt.show()

#%%
# from PIL import Image

# mask2 = np.array(Image.open('../FMECNN/utils/maze.png').resize([n,n]))[...,0]<255
# phi2 = 1e16*(mask2) +1.
# seed2 = np.array([[.99,0.1]])

# u_vec_bar2 = primal_dual_iterations(phi2, seed = int(.5*(1+seed2[0,0])*n)*n+int(.5*(seed2[0,1]+1)*n))

# D2 = u_vec_bar2.reshape([n,n])


# #%%
# plt.figure(210)
# plt.imshow(np.ma.masked_where(mask2==1,D2),extent=[-1,1,-1,1], cmap='cool', origin='lower')
# plt.colorbar()
# plt.scatter(seed2[:,1],seed2[:,0], c='r')
# plt.savefig('../FMECNN/fig_fm/distance_pm2')

# plt.figure(211)
# plt.imshow(np.ma.masked_where(mask2==1,np.sin(32*D2)), cmap='cool', extent=[-1,1,-1,1], origin='lower')
# plt.colorbar()
# plt.scatter(seed2[:,1],seed2[:,0], c='r')
# plt.savefig('../FMECNN/fig_fm/distance_pm_sin2')


# plt.figure(212)
# plt.imshow(phi2, extent=[-1,1,-1,1], origin='lower')
# plt.colorbar()
# plt.scatter(seed2[:,1],seed2[:,0], c='r')
# plt.savefig('../FMECNN/fig_fm/potential_pm2')

# %%
