# -*- coding: utf-8 -*-
"""
Train the model
"""
import os
import sys
import copy
import logging
import argparse

import numpy as np
import torch

import utils

from trainer import train
from config import Config

sys.path.append('../')
from models.steps import (data_preprocessing, metric_simplification, knobs_ranking, prepare_for_training, )

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
    logger.info(f"====================== {opt.persistence} mode ====================\n")

    logger.info(f"Target workload name is {opt.target}")

    knob_data, aggregated_IM_data, aggregated_EM_data, target_knob_data, target_external_data = data_preprocessing(opt.target, opt.persistence, logger)

    logger.info("====================== Metrics_Simplification ====================\n")
    pruned_metrics = metric_simplification(aggregated_IM_data, logger, opt)
    logger.info(f"Done pruning metrics for workload {opt.persistence} (# of pruned metrics: {len(pruned_metrics)}).\n\n"f"Pruned metrics: {pruned_metrics}\n")
    metric_idxs = [i for i, metric_name in enumerate(aggregated_IM_data['columnlabels']) if metric_name in pruned_metrics]
    ranked_metric_data = {
        'data' : aggregated_IM_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(aggregated_IM_data['rowlabels']),
        'columnlabels' : [aggregated_IM_data['columnlabels'][i] for i in metric_idxs]
    }

    ### KNOBS RANKING STAGE ###
    rank_knob_data = copy.deepcopy(knob_data)
    logger.info("====================== Run_Knobs_Ranking ====================\n")
    logger.info(f"use mode = {opt.rki}")
    ranked_knobs = knobs_ranking(knob_data = rank_knob_data,
                                metric_data = ranked_metric_data,
                                mode = opt.rki,
                                logger = logger)
    logger.info(f"Done ranking knobs for workload {opt.persistence} (# ranked knobs: {len(ranked_knobs)}).\n\n"
                 f"Ranked knobs: {ranked_knobs}\n")

    top_k: int = opt.topk
    top_k_knobs = utils.get_ranked_knob_data(ranked_knobs, knob_data, top_k)
    target_knobs = utils.get_ranked_knob_data(ranked_knobs, target_knob_data, top_k)
    knob_save_path = utils.make_date_dir('./save_knobs')
    logger.info(f"Knob save path : {knob_save_path}")
    logger.info(f"Choose Top {top_k} knobs : {top_k_knobs['columnlabels']}")
    np.save(os.path.join(knob_save_path,f'knobs_{top_k}.npy'),np.array(top_k_knobs['columnlabels']))

    model, optimizer, trainDataloader, valDataloader, testDataloader, scaler_y = prepare_for_training(opt, top_k_knobs, target_knobs, aggregated_EM_data, target_external_data)
    
    logger.info(f"====================== {opt.model_mode} Pre-training Stage ====================\n")

    best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_path = train(model, trainDataloader, valDataloader, testDataloader, optimizer, scaler_y, opt, logger)
    logger.info(f"\n\n[Best Epoch {best_epoch}] Best_th_Loss : {best_th_loss} Best_la_Loss : {best_la_loss} Best_th_MAE : {best_th_mae_loss} Best_la_MAE : {best_la_mae_loss}")
    
    config = Config(opt.persistence, opt.db, opt.cluster, opt.rki, opt.topk, opt.model_mode, opt.n_epochs, opt.lr)
    config.save_results(opt.target, best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_path, log_dir, knob_save_path)

    return config

if __name__ == '__main__':
    '''
        internal_metrics, external_metrics, knobs
        metric_data : internal metrics
        knobs_data : configuration knobs
    '''
    print("======================MAKE LOGGER====================")
    logger, log_dir = utils.get_logger(os.path.join('./logs'))
    try:
        main(opt, logger, log_dir)
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()