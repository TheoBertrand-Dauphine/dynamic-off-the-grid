import time
import matplotlib.pyplot as plt
import torch


x1 = torch.tensor([[1,0,0]])

x2 = torch.tensor([[0,1,1]])

x = torch.tensor([(-x1[:,0] + x2[:,0])*torch.cos(x1[:,2]) - (-x1[:,1] + x2[:,1])*torch.sin(x1[:,2]), 
     (x2[:,0] - x1[:,0])*torch.sin(x1[:,2]) + (x2[:,1]-x1[:,1])*torch.cos(x1[:,2]),
      x2[:,2]-x1[:,2] ]).unsqueeze(0) #g_1^-1g_2

c1 = (x[:,0]*torch.cos(x[:,2]/2) + x[:,1]*torch.sin(x[:,2]/2))/(torch.sinc(x[:,2]/2))
c2 = (-x[:,0]*torch.sin(x[:,2]/2) + x[:,1]*torch.cos(x[:,2]/2))/(torch.sinc(x[:,2]/2))
c3 = x[:,2]

w1 = 1
w2 = 3
w3 = .5


t = torch.linspace(0,1,100)

c1_t = t*c1
c2_t = t*c2
c3_t = t*c3

expx = (c1_t*torch.cos(c3_t/2)-c2_t*torch.sin(c3_t/2))*torch.sinc(c3_t/2)+1
expy = (c1_t*torch.sin(c3_t/2)+c2_t*torch.cos(c3_t/2))*torch.sinc(c3_t/2)
exptheta = torch.remainder(c3_t,2*torch.pi)

print(torch.angle(expx +1.j*expy))
print(exptheta)

plt.figure(0)
plt.scatter(expx,expy)