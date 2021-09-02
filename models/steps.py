import os
import json
import logging
import argparse
from collections import defaultdict

from typing import Tuple

import numpy as np
import pandas as pd

from models.cluster import GapStatistic, KMeansClusters, create_kselection_model, MeanShiftClustering, GMMClustering
from models.factor_analysis import FactorAnalysis
from models.preprocessing import (get_shuffle_indices, consolidate_columnlabels)
from models.redisDataset import RedisDataset
from models.ranking import Ranking
from models.dnn import RedisSingleDNN, RedisTwiceDNN

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader,RandomSampler

import utils
import knobs

DATA_PATH = "../data/redis_data"
DEVICE = torch.device("cpu")

import warnings

warnings.filterwarnings('ignore')

def data_preprocessing(target_num: int, persistence: str, logger: logging) -> Tuple[dict, dict, dict, dict, dict]:
    """
    workload{2~18} = workload datas composed of different key(workload2, workload3, ...) [N of configs, N of columnlabels]
    columnlabels  = Internal Metric names
    rowlabels = Index for Workload data
    """
    target_DATA_PATH = "../data/redis_data/workload{}".format(target_num)
    
    knobs_path:str = os.path.join(DATA_PATH, "configs")


    internal_metric_datas = defaultdict(list)
    external_metric_datas = {}
    knob_datas = {}

    for i in range(1,19):
        if target_num == i:
            target_external_data: dict = knobs.load_knob_metrics(metric_path = os.path.join(target_DATA_PATH ,f"result_{persistence.lower()}_external_{i}.csv"),
                                                knobs_path = knobs_path,
                                                metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
            target_knob_data, _ = knobs.load_knob_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{persistence.lower()}_internal_{i}.csv'),
                                                knobs_path = knobs_path,
                                                persistence = persistence,)
        else:
            knob_data, internal_metric_data = knobs.load_knob_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{persistence.lower()}_internal_{i}.csv'),
                                                            knobs_path = knobs_path,
                                                            persistence = persistence,)
            
            external_metric_data: dict  = knobs.load_knob_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{persistence.lower()}_external_{i}.csv'),
                                                knobs_path = knobs_path,
                                                metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
            assert knob_data['data']!=external_metric_data['data'], (len(knob_data['data']), len(external_metric_data['data']))
            knob_datas[f'workload{i}'] = knob_data['data']
            internal_metric_datas[f'workload{i}'] = internal_metric_data['data']
            external_metric_datas[f'workload{i}'] = external_metric_data['data']
            internal_metric_datas['rowlabels'].extend(knob_data['rowlabels'])

    knob_datas['columnlabels'] = knob_data['columnlabels']
    internal_metric_datas['columnlabels'] = internal_metric_data['columnlabels']
    external_metric_datas['columnlabels'] = ['Totals_Ops/sec', 'Totals_p99_Latency']
    logger.info("Finish Load knob and Internal and External Metrics Data")

    aggregated_IM_data: dict = knobs.aggregate_datas(internal_metric_datas)
    aggregated_EM_data: dict = knobs.aggregate_datas(external_metric_datas)
    aggregated_knob_data: dict = knobs.aggregate_datas(knob_datas)

    return aggregated_knob_data, aggregated_IM_data, aggregated_EM_data,\
     target_knob_data, target_external_data

#Step 1
def metric_simplification(metric_data: dict, logger: logging, args : argparse) -> list:
    matrix: list = metric_data['data']
    columnlabels: list = metric_data['columnlabels']

    # Remove any constant columns
    nonconst_matrix = []
    nonconst_columnlabels = []
    for col, (_, v) in zip(matrix.T, enumerate(columnlabels)):
        if np.any(col != col[0]):
            nonconst_matrix.append(col.reshape(-1, 1))
            nonconst_columnlabels.append(v)
    assert len(nonconst_matrix) > 0, "Need more data to train the model"

    nonconst_matrix = np.hstack(nonconst_matrix)
    logger.info(f"Workload characterization ~ nonconst data size: {nonconst_matrix.shape}")

    # Remove any duplicate columns
    unique_matrix, unique_idxs = np.unique(nonconst_matrix, axis=1, return_index=True)
    unique_columnlabels = [nonconst_columnlabels[idx] for idx in unique_idxs]

    logger.info(f"Workload characterization ~ final data size: {unique_matrix.shape}")
    n_rows, n_cols = unique_matrix.shape

    # Shuffle the matrix rows
    shuffle_indices = get_shuffle_indices(n_rows)
    shuffled_matrix: list = unique_matrix[shuffle_indices, :]

    #shuffled_matrix = RobustScaler().fit_transform(shuffled_matrix)
    shuffled_matrix = StandardScaler().fit_transform(shuffled_matrix)
    # shuffled_matrix = MinMaxScaler().fit_transform(shuffled_matrix)

    #FactorAnalysis
    fa_model = FactorAnalysis()
    fa_model.fit(shuffled_matrix, unique_columnlabels, n_components=5)
    # Components: metrics * factors
    components = fa_model.components_.T.copy()

    # Clustering method : Gaussian Mixture Model(GMM)
    logger.info("Clustering mode : {}".format(args.cluster))
    if args.cluster == 'gmm':
        cluster = GMMClustering(components)
        cluster.fit(components)
        pruned_metrics = cluster.get_closest_samples(unique_columnlabels)
        logger.info(f"Found optimal number of clusters: {cluster.optimK}")
    elif args.cluster == 'k-means':
        #KMeansClusters()
        kmeans_models = KMeansClusters()
        ##TODO: Check Those Options
        kmeans_models.fit(components, min_cluster=1,
                          max_cluster=min(n_cols - 1, 20),
                          sample_labels=unique_columnlabels,
                          estimator_params={'n_init': 100})
        gapk = create_kselection_model("gap-statistic")
        gapk.fit(components, kmeans_models.cluster_map_)

        logger.info(f"Found optimal number of clusters: {gapk.optimal_num_clusters_}")
        # Get pruned metrics, cloest samples of each cluster center
        pruned_metrics = kmeans_models.cluster_map_[gapk.optimal_num_clusters_].get_closest_samples()
    
    # Clustering method : Mean Shift
    elif args.cluster == 'ms':
        ms = MeanShiftClustering(components)
        ms.fit(components)
        pruned_metrics = ms.get_closest_samples(unique_columnlabels)
        logger.info(f"Found optimal number of clusters: {len(ms.centroid)}")

    return pruned_metrics


def knobs_ranking(knob_data: dict, metric_data: dict, mode: str, logger: logging) -> list:
    """
    knob_data : will be ranked by knobs_ranking
    metric_data : pruned metric_data by metric simplification
    mode : selct knob_identification(like lasso, xgb, rf)
    logger
    """
    knob_matrix: list = knob_data['data']
    knob_columnlabels: list = knob_data['columnlabels']

    metric_matrix: list = metric_data['data']
    #metric_columnlabels = metric_data['columnlabels']

    encoded_knob_columnlabels = knob_columnlabels
    encoded_knob_matrix = knob_matrix

    # standardize values in each column to N(0, 1)
    #standardizer = RobustScaler()
    standardizer = StandardScaler()
    # standardizer = MinMaxScaler()
    standardized_knob_matrix = standardizer.fit_transform(encoded_knob_matrix)
    standardized_metric_matrix = standardizer.fit_transform(metric_matrix)

    # shuffle rows (note: same shuffle applied to both knob and metric matrices)
    shuffle_indices = get_shuffle_indices(standardized_knob_matrix.shape[0], seed=17)
    shuffled_knob_matrix = standardized_knob_matrix[shuffle_indices, :]
    shuffled_metric_matrix = standardized_metric_matrix[shuffle_indices, :]

    model = Ranking(mode)
    model.fit(shuffled_knob_matrix,shuffled_metric_matrix,encoded_knob_columnlabels)
    encoded_knobs = model.get_ranked_features()
    feature_imp = model.get_ranked_importance()
    if feature_imp is None:
        pass
    else:
        logger.info('Feature importance')
        logger.info(feature_imp)

    consolidated_knobs = consolidate_columnlabels(encoded_knobs)

    return consolidated_knobs


def prepare_for_training(opt: argparse, top_k_knobs: dict, target_knobs: dict, aggregated_EM_data: dict, target_external_data: dict):
    with open("../data/workloads_info.json",'r') as f:
        workload_info = json.load(f)

    workloads=np.array([])
    target_workload = np.array([])
    for workload in range(1,len(workload_info.keys())):
        count = 3000
        if workload != opt.target:
            while count:
                if not len(workloads):
                    workloads = np.array(workload_info[str(workload)])
                    count-=1
                workloads = np.vstack((workloads,np.array(workload_info[str(workload)])))
                count-=1
        else:
            while count:
                if not len(target_workload):
                    target_workload = np.array(workload_info[str(workload)])
                    count-=1
                target_workload = np.vstack((target_workload,np.array(workload_info[str(workload)])))
                count-=1

    top_k_knobs = pd.DataFrame(top_k_knobs['data'], columns = top_k_knobs['columnlabels'])
    target_knobs = pd.DataFrame(target_knobs['data'], columns = target_knobs['columnlabels'])
    aggregated_EM_data = pd.DataFrame(aggregated_EM_data['data'], columns = ['Totals_Ops/sec', 'Totals_p99_Latency'])
    workload_infos = pd.DataFrame(workloads, columns = workload_info['info'])
    target_workload = pd.DataFrame(target_workload, columns= workload_info['info'])
    target_external_data = pd.DataFrame(target_external_data['data'], columns = ['Totals_Ops/sec', 'Totals_p99_Latency'])

    knob_with_workload = pd.concat([top_k_knobs,workload_infos], axis=1)
    target_workload = pd.concat([target_knobs,target_workload], axis=1)
    

    X_train, X_val, y_train, y_val = train_test_split(knob_with_workload, aggregated_EM_data, test_size = 0.33, random_state=42)

    # scaler_X = RobustScaler().fit(X_train)
    # scaler_y = RobustScaler().fit(y_train)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    # scaler_X = MinMaxScaler().fit(X_train)
    # scaler_y = MinMaxScaler().fit(y_train)

    X_tr = scaler_X.transform(X_train).astype(np.float32)
    X_val = scaler_X.transform(X_val).astype(np.float32)
    y_tr = scaler_y.transform(y_train).astype(np.float32)
    y_val = scaler_y.transform(y_val).astype(np.float32)     

    X_te = scaler_X.transform(target_workload).astype(np.float32)
    y_te = scaler_y.transform(target_external_data).astype(np.float32)    

    trainDataset = RedisDataset(X_tr, y_tr)
    valDataset = RedisDataset(X_val, y_val)
    testDataset = RedisDataset(X_te, y_te)

    trainSampler = RandomSampler(trainDataset)
    valSampler = RandomSampler(valDataset)
    testSampler = RandomSampler(testDataset)

    trainDataloader = DataLoader(trainDataset, sampler = trainSampler, batch_size = 32, collate_fn = utils.collate_function)
    valDataloader = DataLoader(valDataset, sampler = valSampler, batch_size = 16, collate_fn = utils.collate_function)
    testDataloader = DataLoader(testDataset, sampler = testSampler, batch_size = 4, collate_fn = utils.collate_function)

    if opt.model_mode == 'single':
        model = RedisSingleDNN(opt.topk+5,2).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr = opt.lr, weight_decay = 0.15)
    elif opt.model_mode == 'twice':
        model = RedisTwiceDNN(opt.topk+5,2).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr = opt.lr, weight_decay = 0.15)
        
    return model, optimizer, trainDataloader, valDataloader, testDataloader, scaler_y


def sinlge_fitness_function(solution: np.array, args: argparse, model: RedisSingleDNN) -> np.array:
    solDataset = RedisDataset(solution, np.zeros((len(solution), 2)))
    solDataloader = DataLoader(solDataset, shuffle = False, batch_size = args.n_pool, collate_fn = utils.collate_function)

    model.eval()

    fitness = np.array([])
    with torch.no_grad():
        for _, batch in enumerate(solDataloader):
            knobs_with_info = batch[0].to(DEVICE)
            fitness_batch = model(knobs_with_info).detach().cpu().numpy()
            if len(fitness) == 0:
                fitness = fitness_batch
            else:
                fitness = np.vstack([fitness, fitness_batch])
    return np.array(fitness)

def twice_fitness_function(solution: np.array, args: argparse, model: RedisTwiceDNN) -> np.array:
    solDataset = RedisDataset(solution, np.zeros((len(solution), 2)))
    solDataloader = DataLoader(solDataset, shuffle = False, batch_size = args.n_pool, collate_fn = utils.collate_function)

    model.eval()

    fitness = np.array([])
    with torch.no_grad():
        for _, batch in enumerate(solDataloader):
            knobs_with_info = batch[0].to(DEVICE)
            fitness_batch = np.array([o.detach().cpu().numpy() for o in model(knobs_with_info)])
            #fitness_batch = model(knobs_with_info)
            if len(fitness) == 0:
                fitness = fitness_batch
            else:
                fitness = np.vstack([fitness, fitness_batch])

    return np.reshape(np.squeeze(fitness, axis = -1), (args.n_pool, 2))


def prepareForGA(args, top_k_knobs):
    with open("../data/workloads_info.json",'r') as f:
        workload_info = json.load(f)

    target_workload_info = np.array([])
    count = 3000
    while count:
        if not len(target_workload_info):
            target_workload_info = np.array(workload_info[args.target])
            count -= 1
        target_workload_info = np.vstack((target_workload_info,np.array(workload_info[args.target])))
        count -= 1

    knobs_path = os.path.join(DATA_PATH, "configs")

    knob_data, _ = knobs.load_knob_metrics(metric_path = os.path.join(DATA_PATH,f'workload{args.target}',f'result_{args.persistence.lower()}_internal_{args.target}.csv'),
                                                            knobs_path = knobs_path,
                                                            persistence = args.persistence,)

    target_external_data = knobs.load_knob_metrics(metric_path = os.path.join(DATA_PATH,f'workload{args.target}',f'result_{args.persistence.lower()}_external_{args.target}.csv'),
                                    knobs_path = knobs_path,
                                    metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])

    target_default_external_data = knobs.load_knob_metrics(metric_path = os.path.join(DATA_PATH,f'workload{args.target}',f'result_{args.persistence.lower()}_external_{args.target}_default.csv'),
                                    knobs_path = knobs_path,
                                    metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])

    top_k_knobs = pd.DataFrame(knob_data['data'], columns = knob_data['columnlabels'])[top_k_knobs]                                 
    target_external_data = pd.DataFrame(target_external_data['data'], columns = ['Totals_Ops/sec', 'Totals_p99_Latency'])
    target_workload_infos = pd.DataFrame(target_workload_info, columns = workload_info['info'])

    knob_with_workload = pd.concat([top_k_knobs, target_workload_infos], axis=1)

    scaler_X = StandardScaler().fit(knob_with_workload)
    scaler_y = StandardScaler().fit(target_external_data)

    # scaler_X = MinMaxScaler().fit(knob_with_workload)
    # scaler_y = MinMaxScaler().fit(target_external_data)

    # scaler_X = StandardScaler().fit(knob_with_workload)
    # scaler_y = StandardScaler().fit(target_external_data)

    deafult = np.sum(np.array(target_default_external_data['data']), axis = 0)

    return knob_with_workload, target_external_data, deafult, scaler_X, scaler_y
