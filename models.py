#imports
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

#imports from https://github.com/hedixia/HeavyBallNODE
from HeavyBallNODE import basehelper, misc
import HeavyBallNODE.base as hb

#load data
p_df = pd.read_csv("POLLU_data")
r_df = pd.read_csv("ROBER_data")
poldat = torch.as_tensor(p_df.to_numpy())
robdat = torch.as_tensor(r_df.to_numpy())

rob_slow = [0, 1, 2]
pol_slow = np.arange(20)

#ROBER base model
nslow = len(rob_slow)
node = 5

class RobNet(nn.Module):
    def __init__(self):
        super(RobNet, self).__init__()
        self.fc1 = nn.Linear(nslow, node)
        self.fc2 = nn.Linear(node, node)
        self.fc3 = nn.Linear(node, node)
        self.fc4 = nn.Linear(node, node)
        self.fc5 = nn.Linear(node, node)
        self.fc6 = nn.Linear(node, node)
        self.fc7 = nn.Linear(node, nslow)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        x = F.gelu(x)
        x = self.fc5(x)
        x = F.gelu(x)
        x = self.fc6(x)
        x = F.gelu(x)
        output = x
        return output

tmp = RobNet()
rob = hb.NODEintegrate(hb.NODE(tmp))

#POLLU base model
nslow = len(pol_slow)
node = 10

class PolNet(nn.Module):
    def __init__(self):
        super(PolNet, self).__init__()
        self.fc1 = nn.Linear(nslow, node)
        self.fc2 = nn.Linear(node, node)
        self.fc3 = nn.Linear(node, node)
        self.fc4 = nn.Linear(node, nslow)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        output = x
        return output

tmp = PolNet()
pol = hb.NODEintegrate(hb.NODE(tmp))

#scaled models
#robscale = torch.amax(robdat, axis=1)[rob_slow] - torch.amin(robdat, axis=1)[rob_slow]
#polscale = torch.amax(poldat, axis=1)[pol_slow] - torch.amin(poldat, axis=1)[pol_slow]
#
#

#heavy ball NODEs
tmp = RobNet()
hbRob = hb.NODEintegrate(hb.HBNODE(tmp))

tmp = PolNet()
hbPol = hb.NODEintegrate(hb.HBNODE(tmp))

#training
n_epoch = 5000
ntotal = 20
batch_size = ntotal

rob_train_times = torch.linspace(0, 1e5, 50)
pol_train_times = torch.linspace(0, 60, 50)

def train(model, data, prob, fname, mname, niter=500, lr_dict=None, gradrec=None, pre_shrink = 0.01):
    lr_dict = {0: 0.001, 50: 0.0001} if lr_dict is None else lr_dict
    recorder = Recorder()
    torch.manual_seed(0)
    model = shrink_parameters(model, pre_shrink)
    criteria = nn.L1loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    #print('Number of Parameters: {}'.format(count_parameters(model)))

    for epoch in range(niter):

        recorder['epoch'] = epoch

        # Train
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])

        if prob == "pollu":
            batchsize = 20
        else:
            batchsize = 50

        for b_n in range(0, list(data.size)[1], batchsize):
            model.cell.nfe = 0
            batch_start_time = time.time()
            model.zero_grad()

            # forward pass
            if prob == "pollu":
                train_times = pol_train_times
            else:
                train_times = rob_train_times

            init, predict = model(evaluation_times = train_times[:b_n+batchsize])

            # loss
            loss = criteria(predict, data[:, b_n:b_n + batchsize])
            recorder['forward_time'] = time.time() - batch_start_time
            recorder['forward_nfe'] = model.cell.nfe
            recorder['loss'] = loss

            # Gradient backprop computation
            if gradrec is not None:
                loss.backward(retain_graph=True)
                vals = model.ode_rnn.h_ode #help
                for i in range(len(vals)):
                    recorder['grad_{}'.format(i)] = torch.norm(vals[i].grad)
                model.zero_grad()

            # Backward pass
            model.cell.nfe = 0
            total_loss.backward()
            recorder['model_gradient_2norm']= gradnorm(model)
            # recorder['cell_gradient_2norm'] = gradnorm(model.cell)
            # recorder['ic_gradient_2norm'] = gradnorm(model.ic)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            recorder['mean_batch_time'] = time.time() - batch_start_time
            recorder['backward_nfe'] = model.cell.nfe

        recorder.capture(verbose=True)
        print('Epoch {} complete.'.format(epoch))

        if epoch % 20 == 0 or epoch == niter:
            recorder.writecsv(fname)
            torch.save(model.state_dict(), mname)

train(hbRob, robdat, "rober", "ROBER", "Heavy Ball NODE")
train(hbPol, poldat,"pollu", "POLLU", "Heavy Ball NODE")

#train(sc_rob, robdat, "rober", "ROBER", "Scaled NODE")
#train(sc_pol, poldat, "pollu", "POLLU", "Scaled NODE")