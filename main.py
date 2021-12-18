import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from HeavyBallNODE.source import *
from HeavyBallNODE import base, basehelper
from HeavyBallNODE.misc import *
import models

#load data
p_df = pd.read_csv("POLLU_data", index_col=0)
r_df = pd.read_csv("ROBER_data", index_col=0)
poldat = torch.as_tensor(p_df.to_numpy())
robdat = torch.as_tensor(r_df.to_numpy())

#instantiate models
ffRob = models.RobNet()
ffPol = models.PolNet()
hbRob = models.RobHB()
hbPol = models.PolHB()

#training
n_epoch = 500

def train(model, data, prob, fname, mname, niter, lr_dict=None, gradrec=None, pre_shrink = 0.01):
    lr_dict = {0: 0.001, 50: 0.0001} if lr_dict is None else lr_dict
    recorder = Recorder()
    torch.manual_seed(0)
    model = shrink_parameters(model, pre_shrink)
    criteria = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    print('Number of Parameters: {}'.format(count_parameters(model)))

    for epoch in range(niter):
        recorder['epoch'] = epoch

        # Train
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])

        if prob == "pollu":
            batchsize = 20
        else:
            batchsize = 50

        for b_n in range(0, data.shape[1], batchsize):
            model.cell.nfe = 0
            batch_start_time = time.time()
            model.zero_grad()

            # forward pass
            if prob == "pollu":
                train_times = torch.linspace(0, 60, 50)
            else:
                train_times = torch.linspace(1e-5, 1e5, 50)

            init, predict = model(train_times[:b_n+batchsize], data[:, b_n:b_n+batchsize])

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
            loss.backward()
            recorder['model_gradient_2norm']= gradnorm(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            recorder['mean_batch_time'] = time.time() - batch_start_time
            recorder['backward_nfe'] = model.cell.nfe

        recorder.capture(verbose=True)
        print('Epoch {} complete.'.format(epoch))

        if epoch % 20 == 0 or epoch == niter:
            recorder.writecsv(fname)
            torch.save(model.state_dict(), mname)

train(hbRob, robdat, "rober", "ROBER", "Heavy Ball NODE", niter = n_epoch)
train(hbPol, poldat,"pollu", "POLLU", "Heavy Ball NODE", niter = n_epoch)