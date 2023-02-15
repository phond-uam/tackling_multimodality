# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:18:53 2021

@author: Michel Frising
         michel.frising@uam.es
"""
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# Glorot uniform initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.normal_()
        m.bias.data *= 0.05

class CINN(nn.Module):
    '''cINN, including the conditioning network'''
    def __init__(self):
        super().__init__()
        # self.trainable_parameters = []
        
    def makeCondNet(self, cond_net, cond_features):
        self.cond_net = cond_net
        self.cond_features = cond_features
        # self.trainable_parameters += list(self.cond_net.parameters())

    def makeINN(self, dim_x, dense_neurons=512, N_coupling_blocks=6):
        # fully connected subnet for the invertible NN
        def sub_fc(ch_hidden):
            def subnet(ch_in, ch_out):
                s = nn.Sequential(nn.Linear(ch_in, ch_hidden),
                                  nn.ReLU(),
                                  nn.Linear(ch_hidden, ch_hidden),
                                  nn.ReLU(),
                                  nn.Linear(ch_hidden, ch_out))
                # initialize the weights of the INN to random values, as describes in the paper
                s.apply(init_weights)
                return s
            return subnet

        # output of the conditioning network to be mixed with the input
        cond = Ff.ConditionNode(self.cond_features)

        # input layer
        nodes = [Ff.InputNode(dim_x, name='Input')]
        # add subnets
        subnet = sub_fc(dense_neurons)
        for i in range(N_coupling_blocks):
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.AllInOneBlock,
                                 {'subnet_constructor': subnet,
                                  'global_affine_init': 1.0},
                                 conditions=cond))
            # # Permutation layers are also sometimes used, I don't
            # nodes.append(Ff.Node(nodes[-1],
            #                      Fm.PermuteRandom,
            #                      {'seed':i},
            #                      name=f'permute_{i}'))

        # split nodes might become necessary if the network becomes too big.
        # At this point I don't use them
        split_nodes = []
        # nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
        #                      Fm.Concat1d, {'dim':0}))
        
        # add a final output layer
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        
        # build CINN
        cinn = Ff.ReversibleGraphNet(nodes + split_nodes + [cond], verbose=False)
        # # extract trainable params
        # cinn_params = [p for p in cinn.parameters() if p.requires_grad]

        #for i, p in enumerate(cinn.parameters()):
            ## init weights to random values
            #p.data = 0.02 * torch.randn_like(p)

        # self.trainable_parameters += cinn_params
        self.cinn = cinn

    def forward(self, x, y):
        conds = self.cond_net(y)
        (z, log_jac_det) = self.cinn(x, c=(conds,))
        return dict(z=z, log_jac_det=log_jac_det)

    def reverse_sample(self, z, y, jac=False):
        conds = self.cond_net(y)
        (x, log_jac_det) = self.cinn(z, c=(conds,), rev=True, jac=jac)
        return x, log_jac_det

