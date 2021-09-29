# -*- coding: utf-8 -*-
"""
Train the model
"""
from collections import defaultdict
import os
import sys
import copy
import logging
import argparse
from random import randint

import numpy as np
import torch

import utils

from double_trainer import train
from config import Config

sys.path.append('../')
from models.double_steps import (data_preprocessing, metric_simplification, knobs_ranking, prepareForTraining, set_model)

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument("--db",type = str, choices = ["redis","rocksdb"], default = 'redis', help="DB type")
parser.add_argument("--cluster",type=str, choices = ['k-means','ms','gmm'],default = 'ms')
parser.add_argument("--rki",type = str, default = 'lasso', help = "knob_identification mode")
parser.add_argument("--topk",type = int, default = 4, help = "Top # knobs")
parser.add_argument("--n_epochs",type = int, default = 100, help = "Train # epochs with model")
parser.add_argument("--lr", type = float, default = 1e-5, help = "Learning Rate")
parser.add_argument("--model_mode", type = str, default = 'single', help = "model mode")

opt = parser.parse_args()
DATA_PATH = "../data/redis_data"
DEVICE = torch.device("cpu")

if not os.path.exists('save_knobs'):
    os.mkdir('save_knobs')

#expr_name = 'train_{}'.format(utils.config_exist(opt.persistence))

def main(opt: argparse , logger: logging, log_dir: str) -> Config:
    #Target workload loading
    logger.info("====================== {} mode ====================\n".format(opt.persistence))
    logger.info("Target workload name is {}".format(opt.target))


    """
        load knob data and IM datas, EM datas.
    """
    ### data load ###
    knob_data, aggregated_IM_data, aggregated_ops_data, aggregated_latency_data, target_knob_data, ops_target_external_data, latency_target_external_data = data_preprocessing(opt.target, opt.persistence, logger)


    ### clustering ###
    logger.info("====================== Metrics_Simplification ====================\n")
    pruned_metrics = metric_simplification(aggregated_IM_data, logger, opt)
    logger.info("Done pruning metrics for workload {} (# of pruned metrics: {}).\n\n""Pruned metrics: {}\n".format(opt.persistence, len(pruned_metrics), pruned_metrics))
    metric_idxs = [i for i, metric_name in enumerate(aggregated_IM_data['columnlabels']) if metric_name in pruned_metrics]
    ranked_metric_data = {
        'data' : aggregated_IM_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(aggregated_IM_data['rowlabels']),
        'columnlabels' : [aggregated_IM_data['columnlabels'][i] for i in metric_idxs]
    }
    """
        For example,
            pruned_metrics : ['allocator_rss_bytes', 'rss_overhead_bytes', 'used_memory_dataset', 'rdb_last_cow_size']
    """

    ### KNOBS RANKING STAGE ###
    rank_knob_data = copy.deepcopy(knob_data)
    logger.info("====================== Run_Knobs_Ranking ====================\n")
    logger.info("use mode = {}".format(opt.rki))
    ranked_knobs = knobs_ranking(knob_data = rank_knob_data,
                                metric_data = ranked_metric_data,
                                mode = opt.rki,
                                logger = logger)
    logger.info("Done ranking knobs for workload {} (# ranked knobs: {}).\n\n"
                 "Ranked knobs: {}\n".format(opt.persistence, len(ranked_knobs), ranked_knobs))

    top_k: dict = opt.topk
    top_k_knobs = utils.get_ranked_knob_data(ranked_knobs, knob_data, top_k)
    target_knobs = utils.get_ranked_knob_data(ranked_knobs, target_knob_data, top_k)
    knob_save_path = utils.make_date_dir('./save_knobs')
    logger.info("Knob save path : {}".format(knob_save_path))
    logger.info("Choose Top {} knobs : {}".format(top_k,top_k_knobs['columnlabels']))
    np.save(os.path.join(knob_save_path,'knobs_{}.npy'.format(top_k)),np.array(top_k_knobs['columnlabels']))
 
    #In double version
    aggregated_data = [aggregated_ops_data, aggregated_latency_data]
    target_external_data = [ops_target_external_data, latency_target_external_data]

    model, optimizer = set_model(opt)
    model_save_path = utils.make_date_dir("./model_save")
    logger.info("Model save path : {}".format(model_save_path))
    logger.info("Learning Rate : {}".format(opt.lr))
    best_epoch, best_loss, best_mae = defaultdict(int), defaultdict(float), defaultdict(float)
    columns=['Totals_Ops/sec','Totals_p99_Latency']

    ### train dnn ###
    for i in range(2):
        trainDataloader, valDataloader, testDataloader, scaler_y = prepareForTraining(opt, top_k_knobs, target_knobs, aggregated_data[i], target_external_data[i],i)
        logger.info("====================== {} Pre-training Stage ====================\n".format(opt.model_mode))

        best_epoch[columns[i]], best_loss[columns[i]], best_mae[columns[i]] = train(model, trainDataloader, valDataloader,testDataloader, optimizer, scaler_y, opt, logger, model_save_path, i)

    for name in best_epoch.keys():
        logger.info("\n\n[{} Best Epoch {}] Best_Loss : {} Best_MAE : {}".format(name, best_epoch[name], best_loss[name], best_mae[name]))
    
    config = Config(opt.persistence,opt.db,opt.cluster,opt.rki,opt.topk,opt.model_mode,opt.n_epochs,opt.lr)
    config.save_double_results(opt.target, best_epoch['Totals_Ops/sec'], best_epoch[name], best_loss['Totals_Ops/sec'], best_loss[name], best_mae['Totals_Ops/sec'], best_mae[name], model_save_path, log_dir, knob_save_path)

    return config

if __name__ == '__main__':
    print("======================MAKE LOGGER====================")
    logger, log_dir = utils.get_logger(os.path.join('./logs'))
    try:
        main(opt, logger, log_dir)
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()