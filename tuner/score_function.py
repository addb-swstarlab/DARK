import math

import numpy as np

def mse_loss(target, predict):
    return np.array([(target[:,i]-predict[:,i])**2 for i in range(2)]).sum(axis=0)

def ATR_loss(default, predict, weight):
    return sum([((-1**i)*weight[i]*(predict[:,i]-default[:,i]))/default[:,i] for i in range(len(weight))])

def throughput_loss(default, predict):
    loss = []
    for d, p in zip(default, predict):
        if p-d>0:
            loss.append(math.log10(p-d) + 1)
        elif p-d == 0:
            loss.append(0)
        elif p-d<0:
            loss.append(-(math.log10(abs(p-d)) + 1))
    return np.array(loss)

def latency_loss(default, predict):
    loss = []
    for d, p in zip(default, predict):
        if d-p>0:
            loss.append(-pow(2,d-p))
        elif d-p==0:
            loss.append(0)
        elif d-p<0:
            loss.append(pow(2,d-p))
    return np.array(loss)

def DRT_loss(default, predict, weight):
    losses = [throughput_loss, latency_loss]
    return sum([weight[i]*losses[i](default[:,i], predict[:,i]) for i in range(len(weight))])

def throughput_new_loss(default, predict):
    return (predict/default-1)*100

def latency_new_loss(default, predict):
    return (1-(predict/default))

def DRT_new_loss(default, predict, weight):
    losses = [throughput_new_loss, latency_new_loss]
    return sum([weight[i]*losses[i](default[:,i], predict[:,i])*((-1)**i) for i in range(len(weight))])