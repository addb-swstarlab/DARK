import os
import math
import sys
import logging
import argparse
from tqdm import tqdm

import torch
import datetime
import numpy as np
import pandas as pd

from knobs import generate_config
import utils

sys.path.append('../')

from models.steps import (sinlge_fitness_function, twice_fitness_function, prepareForGA)
from models.double_steps import (double_fitness_function, double_prepareForGA)
from models.dnn import RedisSingleDNN, RedisTwiceDNN


parser = argparse.ArgumentParser()
parser.add_argument('--target', type = str, default = '1', help='Target Workload')
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument('--topk', type = int, default=4,)
parser.add_argument('--path',type= str)
parser.add_argument('--sk', type= str, default=' ')
parser.add_argument('--num', type = str, nargs='+')
parser.add_argument('--n_pool',type = int, default = 64)
parser.add_argument('--n_generation', type=int, default=8000,)
parser.add_argument("--model_mode", type = str, default = 'double', help = "model mode")

args = parser.parse_args()

if not os.path.exists('save_knobs'):
    assert "Do this file after running main.py"

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


print("======================MAKE GA LOGGER====================")
logger, log_dir = utils.get_logger(os.path.join('./GA_logs'))

def server_connection(args, top_k_config_path, name):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('34.64.145.240', username='jieun', password='1423')

    sftp = client.open_sftp()
    sftp.put(top_k_config_path, './redis-sample-generation/'+name)
    command = f'python ./redis-sample-generation/connection.py {args.persistence.lower()} {args.target} ./redis-sample-generation/'+name
    _, ssh_stdout, _ = client.exec_command(command)
    exit_status = ssh_stdout.channel.recv_exit_status()
    if exit_status == 0:
        sftp.get(f'/home/jieun/result_{args.persistence.lower()}_external_GA.csv', f'./GA_config/result_{args.persistence.lower()}_external_GA.csv')
        sftp.get(f'/home/jieun/result_{args.persistence.lower()}_internal_GA.csv', f'./GA_config/result_{args.persistence.lower()}_internal_GA.csv')
    sftp.close()
    client.exec_command('rm ./redis-sample-generation/'+name)
    client.exec_command(f'rm /home/jieun/result_{args.persistence.lower()}_external_GA.csv')
    client.exec_command(f'rm /home/jieun/result_{args.persistence.lower()}_internal_GA.csv')

    client.close()

def main():
    if args.sk==' ':
        args.sk = args.path

    top_k_knobs = np.load(os.path.join('./save_knobs',args.sk,f"knobs_{args.topk}.npy"))
    #import utils
    #predict_save_path = utils.make_date_dir('./save_predicts')

    if args.model_mode == 'single':
        model = RedisSingleDNN(args.topk+5,2)
        model.load_state_dict(torch.load(os.path.join('./model_save',args.path,'model_{}.pt'.format(args.num[0]))))
        fitness_function = sinlge_fitness_function
    elif args.model_mode == 'twice':
        model = RedisTwiceDNN(args.topk+5,2)
        model.load_state_dict(torch.load(os.path.join('./model_save',args.path,'model_{}.pt'.format(args.num[0]))))
        fitness_function = twice_fitness_function
    elif args.model_mode == 'double':
        models = [RedisSingleDNN(args.topk+5,1), RedisSingleDNN(args.topk+5,1)]
        models[0].load_state_dict(torch.load(os.path.join('./model_save',args.path,'Totals_Ops_sec_{}.pt'.format(args.num[0]))))
        models[1].load_state_dict(torch.load(os.path.join('./model_save',args.path,'Totals_p99_Latency_{}.pt'.format(args.num[1]))))
        fitness_function = double_fitness_function

    if args.model_mode == 'single' or args.model_mode == 'twice':
        pruned_configs, external_data, default, scaler_X, scaler_y = prepareForGA(args,top_k_knobs)
        temp_configs = pd.concat([pruned_configs, external_data], axis=1)
        temp_configs = temp_configs.sort_values(["Totals_Ops/sec","Totals_p99_Latency"], ascending=[False,True])
        target = temp_configs[["Totals_Ops/sec","Totals_p99_Latency"]].values[0]
        configs = temp_configs.drop(columns=["Totals_Ops/sec","Totals_p99_Latency"])
        current_solution_pool = configs[:args.n_pool].values
        target = np.repeat([default], args.n_pool, axis = 0)

    elif args.model_mode == 'double':
        """
            external_data: [ops_external_data, lat_external_data]
            default: [ops_deafult, lat_deafult]
            scaler_y: [scaler_ops, scaler_lat]
        """
        pruned_configs, external_datas, defaults, scaler_X, scaler_ys = double_prepareForGA(args,top_k_knobs)
        temp_configs = [pd.concat([pruned_configs, external_data], axis=1) for external_data in external_datas]
        temp_configs[0] = temp_configs[0].sort_values(["Totals_Ops/sec"], ascending=[False])
        temp_configs[1] = temp_configs[1].sort_values(["Totals_p99_Latency"], ascending=[True])
        targets = [None, None]
        targets[0] = temp_configs[0][["Totals_Ops/sec"]].values[0]
        targets[1] = temp_configs[1][["Totals_p99_Latency"]].values[0]
        configs = [None, None]
        configs[0] = temp_configs[0].drop(columns=["Totals_Ops/sec"])
        configs[1] = temp_configs[1].drop(columns=["Totals_p99_Latency"])
        current_solution_pools = [config[:args.n_pool].values for config in configs]
        targets = [np.repeat([default], args.n_pool, axis = 0) for default in defaults]
        
    #losses = [throughput_loss, latency_loss]
    losses = [throughput_new_loss, latency_new_loss]
    n_configs = top_k_knobs.shape[0]
    n_pool_half = args.n_pool//2
    mutation = int(n_configs*0.5)
    ops_predicts = []
    lat_predicts = []
    for i in tqdm(range(args.n_generation)):
        if args.model_mode == 'single' or args.model_mode == 'twice':
            scaled_pool = scaler_X.transform(current_solution_pool)
            predicts = fitness_function(scaled_pool, args, model)
            fitness = scaler_y.inverse_transform(predicts)
            ops_predicts.append(np.max(fitness[:,0]))
            lat_predicts.append(np.min(fitness[:,1]))
            #idx_fitness = ATR_loss(target, fitness,[0.5,0.5])
            idx_fitness = DRT_new_loss(target, fitness,[0.5,0.5])
            sorted_idx_fitness = np.argsort(idx_fitness)[n_pool_half:]
            best_solution_pool = current_solution_pool[sorted_idx_fitness,:]
            if i % 1000 == 999:
                logger.info(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {max(idx_fitness)}")
            pivot = np.random.choice(np.arange(1,n_configs))
            new_solution_pool = np.zeros_like(best_solution_pool)
            for j in range(n_pool_half):
                new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
                new_solution_pool[j][pivot:n_configs] = best_solution_pool[n_pool_half-1-j][pivot:n_configs]
                new_solution_pool[j][n_configs:] = current_solution_pool[0][n_configs:]
                import utils, random
                random_knobs = utils.make_random_option(top_k_knobs)
                knobs = list(random_knobs.values())
                random_knob_index = list(range(n_configs))
                random.shuffle(random_knob_index)
                random_knob_index = random_knob_index[:mutation]
                for k in range(len(random_knob_index)):
                    new_solution_pool[j][random_knob_index[k]] = knobs[random_knob_index[k]]
            current_solution_pool = np.vstack([best_solution_pool, new_solution_pool])
        elif args.model_mode == 'double':
            index = i%2
            scaled_pool = scaler_X.transform(current_solution_pools[index])
            predicts = fitness_function(scaled_pool, args, models[index])
            fitness = scaler_ys[index].inverse_transform(predicts)
            if index:
                lat_predicts.append(np.min(fitness))
            else:
                ops_predicts.append(np.max(fitness))
            idx_fitness = np.squeeze(losses[index](targets[index], fitness))
            #idx_fitness = ATR_loss(targets[index], fitness,[0.5])
            sorted_idx_fitness = np.argsort(idx_fitness)[n_pool_half:]
            best_solution_pool = current_solution_pools[index][sorted_idx_fitness,:]
            if i % 1000 == 998:
                logger.info(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {max(idx_fitness)}")
            if i % 1000 == 999:
                logger.info(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {max(idx_fitness)}")
            pivot = np.random.choice(np.arange(1,n_configs))
            new_solution_pool = np.zeros_like(best_solution_pool)
            for j in range(n_pool_half):
                new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
                new_solution_pool[j][pivot:n_configs] = best_solution_pool[n_pool_half-1-j][pivot:n_configs]
                new_solution_pool[j][n_configs:] = current_solution_pools[index][0][n_configs:]
                import utils, random
                random_knobs = utils.make_random_option(top_k_knobs)
                knobs = list(random_knobs.values())
                random_knob_index = list(range(n_configs))
                random.shuffle(random_knob_index)
                random_knob_index = random_knob_index[:mutation]
                for k in range(len(random_knob_index)):
                    new_solution_pool[j][random_knob_index[k]] = knobs[random_knob_index[k]]
            current_solution_pools[index] = np.vstack([best_solution_pool, new_solution_pool])
    np.save(os.path.join('save_predicts',f'{args.persistence}_{args.n_pool}_{args.target}_ops.npy'),np.array(ops_predicts))
    np.save(os.path.join('save_predicts',f'{args.persistence}_{args.n_pool}_{args.target}_lat.npy'),np.array(lat_predicts))
    
    final_solution_pool = pd.DataFrame(best_solution_pool)
    logger.info(top_k_knobs)
    logger.info(final_solution_pool)
    top_k_config_path, name, connect = generate_config(args, top_k_knobs, final_solution_pool)
    if connect:
        server_connection(args, top_k_config_path, name)
    else:
        logger.info("Because appednfsync is 'always', Fin GA")
        return 0
    i = 0
    today = datetime.datetime.now()
    name = 'result_'+args.persistence+'-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.csv'
    while os.path.exists(os.path.join('./GA_config/', name)):
        i += 1
        name = 'result_'+args.persistence+'-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.csv'
    os.rename(f'./GA_config/result_{args.persistence.lower()}_external_GA.csv', './GA_config/'+name)
    logger.info(name)
    df = pd.read_csv('./GA_config/'+name)
    logger.info(df["Totals_Ops/sec"])
    logger.info(df["Totals_p99_Latency"])

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
