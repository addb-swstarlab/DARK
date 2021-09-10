import argparse
import datetime
import logging
import math
import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from score_function import *
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

    TUNE_EM = ["Totals_Ops/sec","Totals_p99_Latency"]
    top_k_knobs = np.load(os.path.join('./save_knobs',args.sk,f"knobs_{args.topk}.npy"))

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
    # elif args.model_mode == 'double':
    #     models = [RedisSingleDNN(args.topk,1), RedisSingleDNN(args.topk,1)]
    #     models[0].load_state_dict(torch.load(os.path.join('./model_save',args.path,'Totals_Ops_sec_{}.pt'.format(args.num[0]))))
    #     models[1].load_state_dict(torch.load(os.path.join('./model_save',args.path,'Totals_p99_Latency_{}.pt'.format(args.num[1]))))
    #     fitness_function = double_fitness_function

    if args.model_mode == 'single' or args.model_mode == 'twice':
        pruned_configs, external_data, default, scaler_X, scaler_y = prepareForGA(args, top_k_knobs)
        temp_configs = pd.concat([pruned_configs, external_data], axis = 1)
        temp_configs = temp_configs.sort_values(TUNE_EM, ascending = [False, True])
        target = temp_configs[TUNE_EM].values[0]
        configs = temp_configs.drop(columns = TUNE_EM)
        current_solution_pool = configs[:args.n_pool].values
        target = np.repeat([default], args.n_pool, axis = 0)

    elif args.model_mode == 'double':
        """
            external_data: [ops_external_data, lat_external_data]
            default: [ops_deafult, lat_deafult]
            scaler_y: [scaler_ops, scaler_lat]
        """
        pruned_configs, external_datas, defaults, scaler_X, scaler_ys = double_prepareForGA(args, top_k_knobs)
        temp_configs = [pd.concat([pruned_configs, external_data], axis = 1) for external_data in external_datas]
        targets = []
        configs = []
        for i in range(len(TUNE_EM)):
            temp_configs[i] = temp_configs[i].sort_values(TUNE_EM[i], ascending = [i])

        for i in range(len(TUNE_EM)):
            targets.append(temp_configs[i][[TUNE_EM[i]]].values[0])

        for i in range(len(TUNE_EM)):
            configs.append(temp_configs[i].drop(columns = [TUNE_EM[i]]))

        current_solution_pools = [config[:args.n_pool].values for config in configs]
        targets = [np.repeat([default], args.n_pool, axis = 0) for default in defaults]
        
    #losses = [throughput_loss, latency_loss]
    losses = [throughput_new_loss, latency_new_loss]
    n_configs = top_k_knobs.shape[0]
    #set remain ratio
    n_pool_half = args.n_pool//2
    #mutation ratio
    mutation = int(n_configs*0.5)

    ops_predicts = []
    lat_predicts = []
    for i in tqdm(range(args.n_generation)):
        if args.model_mode == 'single' or args.model_mode == 'twice':
            #fitness function
            scaled_pool = scaler_X.transform(current_solution_pool)
            predicts = fitness_function(scaled_pool, args, model)
            fitness = scaler_y.inverse_transform(predicts)

            #save preidct ops and latency
            ops_predicts.append(np.max(fitness[:,0]))
            lat_predicts.append(np.min(fitness[:,1]))

            #idx_fitness = ATR_loss(target, fitness,[0.5,0.5])
            #score function and sort by score
            idx_fitness = DRT_new_loss(target, fitness,[0.5,0.5])
            sorted_idx_fitness = np.argsort(idx_fitness)[n_pool_half:]
            best_solution_pool = current_solution_pool[sorted_idx_fitness, :]

            if i % 1000 == 999:
                logger.info(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {max(idx_fitness)}")

            #random select crossover ratio
            pivot = np.random.choice(np.arange(1, n_configs))
            new_solution_pool = np.zeros_like(best_solution_pool)
            for j in range(n_pool_half):
                #crossover
                new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
                new_solution_pool[j][pivot:n_configs] = best_solution_pool[n_pool_half-1-j][pivot:n_configs]
                new_solution_pool[j][n_configs:] = current_solution_pool[0][n_configs:]

                #mutation
                import utils, random
                random_knobs = utils.make_random_option(top_k_knobs)
                knobs = list(random_knobs.values())
                random_knob_index = list(range(n_configs))
                random.shuffle(random_knob_index)
                random_knob_index = random_knob_index[:mutation]
                for k in range(len(random_knob_index)):
                    new_solution_pool[j][random_knob_index[k]] = knobs[random_knob_index[k]]

            #new solution pool for next generation
            current_solution_pool = np.vstack([best_solution_pool, new_solution_pool])

        elif args.model_mode == 'double':
            #fitness function
            index = i%2
            scaled_pool = scaler_X.transform(current_solution_pools[index])
            predicts = fitness_function(scaled_pool, args, models[index])
            fitness = scaler_ys[index].inverse_transform(predicts)

            #save preidct ops and latency
            if index:
                lat_predicts.append(np.min(fitness))
            else:
                ops_predicts.append(np.max(fitness))
            
            #score function and sort by score
            idx_fitness = np.squeeze(losses[index](targets[index], fitness))
            sorted_idx_fitness = np.argsort(idx_fitness)[n_pool_half:]
            best_solution_pool = current_solution_pools[index][sorted_idx_fitness,:]

            if i % 1000 == 998:
                logger.info(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {max(idx_fitness)}")
            if i % 1000 == 999:
                logger.info(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {max(idx_fitness)}")
            
            #random select crossover ratio
            pivot = np.random.choice(np.arange(1,n_configs))
            new_solution_pool = np.zeros_like(best_solution_pool)
            for j in range(n_pool_half):
                #crossover
                new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
                new_solution_pool[j][pivot:n_configs] = best_solution_pool[n_pool_half-1-j][pivot:n_configs]
                new_solution_pool[j][n_configs:] = current_solution_pools[index][0][n_configs:]
                
                #mutation
                import utils, random
                random_knobs = utils.make_random_option(top_k_knobs)
                knobs = list(random_knobs.values())
                random_knob_index = list(range(n_configs))
                random.shuffle(random_knob_index)
                random_knob_index = random_knob_index[:mutation]
                for k in range(len(random_knob_index)):
                    new_solution_pool[j][random_knob_index[k]] = knobs[random_knob_index[k]]

            #new solution pool for next generation        
            current_solution_pools[index] = np.vstack([best_solution_pool, new_solution_pool])
    np.save(os.path.join('save_predicts',f'{args.persistence}_{args.n_pool}_{args.target}_ops.npy'),np.array(ops_predicts))
    np.save(os.path.join('save_predicts',f'{args.persistence}_{args.n_pool}_{args.target}_lat.npy'),np.array(lat_predicts))
    
    #generation_config by GA results
    final_solution_pool = pd.DataFrame(best_solution_pool)
    logger.info(top_k_knobs)
    logger.info(final_solution_pool)
    top_k_config_path, name, connect = generate_config(args, top_k_knobs, final_solution_pool)

    #connect other server to run memtier_benchmarks
    if connect:
        server_connection(args, top_k_config_path, name)
    else:
        logger.info("Because appednfsync is 'always', Fin GA")
        return 0

    #save results
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
