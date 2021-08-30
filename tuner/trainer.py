import os
import sys
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader

import utils
sys.path.append('../')
from models.dnn import RedisSingleDNN, RedisTwiceDNN


DEVICE = torch.device("cpu")

def mse_loss(targets: np.array, predict: np.array) -> list:
    return [np.mean((predict[:,i]-targets[:,i])**2) for i in range(2)]

def mae_loss(targets: np.array, predict: np.array) -> list:
    return [np.mean(np.absolute(predict[:,i]-targets[:,i])) for i in range(2)]

def train_single_epoch(model: RedisSingleDNN, trainDataloader: DataLoader, optimizer: AdamW) -> Tuple[float, float]:
    train_loss = 0.0
    train_ACC = 0
    train_steps = 0
    model.train()
    for _ , batch in enumerate(tqdm(trainDataloader, desc = "Iteration")):
        optimizer.zero_grad()
        knobs_with_info = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)
        outputs = model(knobs_with_info)
        loss = F.mse_loss(outputs,targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps +=1

    return train_loss / len(trainDataloader), train_ACC

def train_twice_epoch(model: RedisTwiceDNN, trainDataloader: DataLoader, optimizer: AdamW) -> Tuple[float, float]:
    train_loss = 0.0
    train_ACC = 0
    train_steps = 0
    model.train()
    weight = [0.6,0.4]
    for _ , batch in enumerate(tqdm(trainDataloader, desc = "Iteration")):
        optimizer.zero_grad()
        knobs_with_info = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)

        outputs = model(knobs_with_info)

        loss = 0.
        for i, output in enumerate(outputs):
            loss += weight[i]*F.mse_loss(output.squeeze(1), targets[:, i])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps +=1

    return train_loss / len(trainDataloader), train_ACC

def eval_single_epoch(model: RedisSingleDNN, valDataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    val_loss = 0
    val_ACC = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = F.mse_loss(outputs,targets)
            val_loss += loss.item()
    val_loss /= len(valDataloader)
    return val_loss, val_ACC

def eval_twice_epoch(model: RedisTwiceDNN, valDataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    val_loss = 0
    val_ACC = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = 0.
            for i, output in enumerate(outputs):
                loss += F.mse_loss(output.squeeze(1),targets[:,i])
            val_loss += loss.item()
    val_loss /= len(valDataloader)
    return val_loss, val_ACC

def test_single(model: RedisSingleDNN, testDataloader: DataLoader, scaler_y):
    th_mse, th_mae = 0., 0.
    la_mse, la_mae = 0., 0.
    model.eval()
    outputs = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader, desc = "Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = scaler_y.inverse_transform(batch[1].to(DEVICE))
            outputs = scaler_y.inverse_transform(model(knobs_with_info))
            t_mse, l_mse = mse_loss(targets,outputs)
            t_mae, l_mae = mae_loss(targets,outputs)
            th_mse+=t_mse.item()
            la_mse+=l_mse.item()
            th_mae+=t_mae.item()
            la_mae+=l_mae.item()
    ops_mse = th_mse/len(testDataloader)
    lat_mse = la_mse/len(testDataloader)
    ops_mae = th_mae/len(testDataloader)
    lat_mae = la_mae/len(testDataloader)
    return [ops_mse,lat_mse], [ops_mae,lat_mae]

def test_twice(model, testDataloader, scaler_y):
    th_mse, th_mae = 0., 0.
    la_mse, la_mae = 0., 0.
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader, desc = "Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = scaler_y.inverse_transform(batch[1].to(DEVICE))
            throughput, latency = model(knobs_with_info)
            outputs = torch.cat([throughput,latency], axis = 1)
            outputs = scaler_y.inverse_transform(outputs)

            t_mse, l_mse = mse_loss(targets, outputs)
            t_mae, l_mae = mae_loss(targets, outputs)

            th_mse+=t_mse.item()
            la_mse+=l_mse.item()
            th_mae+=t_mae.item()
            la_mae+=l_mae.item()

            true.extend(targets)
            pred.extend(outputs)
            
    ops_mse = th_mse/len(testDataloader)
    lat_mse = la_mse/len(testDataloader)
    ops_mae = th_mae/len(testDataloader)
    lat_mae = la_mae/len(testDataloader)

    with open("true.npy",'wb') as f:
        np.save(f, true)
    with open("pred.npy",'wb') as f:
        np.save(f, pred)
    
    return [ops_mse,lat_mse], [ops_mae,lat_mae]

def train(model, trainDataloader, valDataloader, testDataloader, optimizer, scaler_y, opt, logger):
    val_losses = []
    test_losses = []
    model_save_path = utils.make_date_dir("./model_save")
    logger.info(f"Model save path : {model_save_path}")
    logger.info(f"Learning Rate : {opt.lr}")

    if opt.model_mode == 'single':
        train_epoch = train_single_epoch
        eval_epoch = eval_single_epoch
        test = test_single
    elif opt.model_mode == 'twice':
        train_epoch = train_twice_epoch
        eval_epoch = eval_twice_epoch
        test = test_twice

    best_loss = float('inf')
    patience = 0

    for epoch in range(int(opt.n_epochs)):
        patience +=1
        logger.info("====================================Train====================================")
        train_loss, _ = train_epoch(model,trainDataloader,optimizer)
        logger.info(f"[Train Epoch {epoch+1}] train Loss : {train_loss}")

        logger.info("====================================Val====================================")
        val_loss, _ = eval_epoch(model,valDataloader)
        logger.info(f"[Eval Epoch {epoch+1}] val Loss : {val_loss}")

        logger.info("====================================Test====================================")
        test_loss, test_mae = test(model, testDataloader, scaler_y)
        logger.info(f"[Epoch {epoch+1}] Test_throughput_Loss: {test_loss[0]}, Test_latency_Loss: {test_loss[1]} , Test_throughput_MAE_Loss: {test_mae[0]}, Test_latency_MAE_Loss: {test_mae[1]}")

        if sum(test_loss) < best_loss:
            torch.save(model.state_dict(),os.path.join(model_save_path,"model_"+str(epoch+1)+".pt"))
            best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss  = test_loss[0], test_loss[1], test_mae[0], test_mae[1]
            best_loss = sum(test_loss)
            patience = 0
            best_epoch = epoch+1
        if patience == 10:
            break
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    return best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_save_path