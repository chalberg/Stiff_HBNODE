import torch
import torch.nn as nn
import torchdiffeq.adjoint_odeint as odeint
import pandas as pd

#ROBER system
class ROBER(nn.Module):
    def __init__(self):
        super(ROBER, self).__init__()
        self.k1 = nn.Parameter(torch.as_tensor([0.04]))
        self.k2 = nn.Parameter(torch.as_tensor([3e7]))
        self.k3 = nn.Parameter(torch.as_tensor([1e4]))

    def forward(self, t, y):
        y1, y2, y3 = y
        du1 = (-self.k1[0] * y1) + (self.k3[0] * y2 * y3)
        du2 = (self.k1[0] * y1) - (self.k3[0] * y2 * y3) - (self.k2[0] * (y2 ** 2))
        du3 = self.k2[0] * (y2 ** 2)
        return torch.stack([du1, du2, du3])

#POLLU system
class POLLU(nn.Module):
    def __init__(self):
        super(POLLU, self).__init__()
        self.k1 = nn.Parameter(torch.as_tensor([.35e0]))
        self.k2 = nn.Parameter(torch.as_tensor([.266e2]))
        self.k3 = nn.Parameter(torch.as_tensor([.123e5]))
        self.k4 = nn.Parameter(torch.as_tensor([.86e-3]))
        self.k5 = nn.Parameter(torch.as_tensor([.82e-3]))
        self.k6 = nn.Parameter(torch.as_tensor([.15e5]))
        self.k7 = nn.Parameter(torch.as_tensor([.13e-3]))
        self.k8 = nn.Parameter(torch.as_tensor([.24e5]))
        self.k9 = nn.Parameter(torch.as_tensor([.165e5]))
        self.k10 = nn.Parameter(torch.as_tensor([.9e4]))
        self.k11 = nn.Parameter(torch.as_tensor([.22e-1]))
        self.k12 = nn.Parameter(torch.as_tensor([.12e5]))
        self.k13 = nn.Parameter(torch.as_tensor([.188e1]))
        self.k14 = nn.Parameter(torch.as_tensor([.163e5]))
        self.k15 = nn.Parameter(torch.as_tensor([.48e7]))
        self.k16 = nn.Parameter(torch.as_tensor([.35e-3]))
        self.k17 = nn.Parameter(torch.as_tensor([.175e-1]))
        self.k18 = nn.Parameter(torch.as_tensor([.1e9]))
        self.k19 = nn.Parameter(torch.as_tensor([.444e12]))
        self.k20 = nn.Parameter(torch.as_tensor([.124e4]))
        self.k21 = nn.Parameter(torch.as_tensor([.21e1]))
        self.k22 = nn.Parameter(torch.as_tensor([.578e1]))
        self.k23 = nn.Parameter(torch.as_tensor([.474e-1]))
        self.k24 = nn.Parameter(torch.as_tensor([.178e4]))
        self.k25 = nn.Parameter(torch.as_tensor([.312e1]))

    def forward(self, t, y):
        r1 = self.k1 * y[0]
        r2 = self.k2 * y[1] * y[3]
        r3 = self.k3 * y[4] * y[1]
        r4 = self.k4 * y[6]
        r5 = self.k5 * y[6]
        r6 = self.k6 * y[6] * y[5]
        r7 = self.k7 * y[8]
        r8 = self.k8 * y[8] * y[7]
        r9 = self.k9 * y[10] * y[1]
        r10 = self.k10 * y[10] * y[0]
        r11 = self.k11 * y[12]
        r12 = self.k12 * y[9] * y[1]
        r13 = self.k13 * y[13]
        r14 = self.k14 * y[0] * y[5]
        r15 = self.k15 * y[2]
        r16 = self.k16 * y[3]
        r17 = self.k17 * y[3]
        r18 = self.k18 * y[15]
        r19 = self.k19 * y[15]
        r20 = self.k20 * y[16] * y[5]
        r21 = self.k21 * y[18]
        r22 = self.k22 * y[18]
        r23 = self.k23 * y[0] * y[3]
        r24 = self.k24 * y[18] * y[0]
        r25 = self.k25 * y[19]

        dy = torch.zeros(20)
        dy[0] = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25
        dy[1] = -r2 - r3 - r9 - r12 + r1 + r21
        dy[2] = -r15 + r1 + r17 + r19 + r22
        dy[3] = -r2 - r16 - r17 - r23 + r15
        dy[4] = -r3 + r4 + r4 + r6 + r7 + r13 + r20
        dy[5] = -r6 - r8 - r14 - r20 + r3 + r18 + r18
        dy[6] = -r4 - r5 - r6 + r13
        dy[7] = r4 + r5 + r6 + r7
        dy[8] = -r7 - r8
        dy[9] = -r12 + r7 + r9
        dy[10] = -r9 - r10 + r8 + r11
        dy[11] = r9
        dy[12] = -r11 + r10
        dy[13] = -r13 + r12
        dy[14] = r14
        dy[15] = -r18 - r19 + r16
        dy[16] = -r20
        dy[17] = r20
        dy[18] = -r21 - r22 - r24 + r23 + r25
        dy[19] = -r25 + r24
        return dy

#initial conditions and time scales
r0 = torch.as_tensor([1.0, 0.0, 0.0])
rt = torch.linspace(0, 1e5, 50)

p0 = torch.zeros(20)
p0[1]  = 0.2
p0[3]  = 0.04
p0[6]  = 0.1
p0[7]  = 0.3
p0[8]  = 0.01
p0[16] = 0.007
pt = torch.linspace(0, 60, 50)

#forward solve using LSODA
rbr = ROBER()
pol = POLLU()

def get_data():
    sol_rober = odeint(rbr, r0, rt, method="scipy_solver", options={"solver": 'LSODA'}, atol= 1e-7)
    sol_pollu = odeint(pol, p0, pt, method="scipy_solver", options={'solver': 'LSODA'}, atol= 1e-7)
    robdat = torch.as_tensor(sol_rober)
    poldat = torch.as_tensor(sol_pollu)
    return robdat, poldat

def save_dat():
    r, p = get_data()
    r_df = pd.DataFrame(r.numpy())
    p_df = pd.DataFrame(p.numpy())
    #save data to file
    p_df.to_csv("POLLU_data",index=False)
    r_df.to_csv("ROBER_data",index=False)

save_dat()
