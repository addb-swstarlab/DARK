import datetime
import json
import os

from typing import Tuple, List, Any
from pandas.core.frame import DataFrame

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

def knobs_make_dict(knobs_path: str, pd_metrics: DataFrame, persistence: str) -> dict:
    '''
        input: DataFrame form (samples_num, knobs_num)
        output: Dictionary form --> RDB and AOF
            ex. dict_knobs = {'columnlabels'=array([['knobs_1', 'knobs_2', ...],['knobs_1', 'knobs_2', ...], ...]),
                                'rowlabels'=array([1, 2, ...]),
                                'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}

        For mode selection knob, "yes" -> 1 , "no" -> 0
    '''
    #config_files: List[str] = os.listdir(knobs_path)
    #config_files: List[str] = [0]*1000
    config_files = pd_metrics['Index']
    if persistence.lower() == 'rdb':
        config_files+=10000

    dict_RDB, dict_AOF = {}, {}
    RDB_datas, RDB_columns, RDB_rowlabels = [], [], []
    AOF_datas, AOF_columns, AOF_rowlabels = [], [], []
    ISAOF = 0
    ISRDB = 1
    for m in range(20000):
        flag = 0
        datas, columns = [], []
        knob_path: str = os.path.join(knobs_path, 'config'+str(m)+'.conf')
        f = open(knob_path, 'r')
        config_file: List[str] = f.readlines()
        knobs_list = config_file[62:]
        #TODO:: why whitespace in line 63
        #knobs_list = config_file[63:]
        #knobs_list = config_file[config_file.index('\n')+1:]
        cnt = 1

        for knobs in knobs_list:
            if knobs.split()[0] != 'save':
                knob, data = knobs.strip().split()
                if data.isalpha() or '-' in data:
                    if data in ["no","yes"]:
                        data = ["no","yes"].index(data)
                    elif data in ["always","everysec","no"]:
                        data = ["always","everysec","no"].index(data)
                    #maxmemory-policy
                    elif data in ["volatile-lru","allkeys-lru","volatile-lfu","allkeys-lfu","volatile-random","allkeys-random","volatile-ttl","noeviction"]:
                        data = ["volatile-lru","allkeys-lru","volatile-lfu","allkeys-lfu","volatile-random","allkeys-random","volatile-ttl","noeviction"].index(data)
                elif data.endswith("mb") or data.endswith("gb"):
                    data = data[:-2]
                datas.append(data)
                columns.append(knob)
            else:
                knob, data1, data2 = knobs.split()
                columns.append(knob+str(cnt)+"_sec")
                columns.append(knob+str(cnt)+"_changes")
                datas.append(data1)
                datas.append(data2)
                cnt += 1

            if knobs.split()[0] == 'appendonly':
                flag = ISAOF
            if knobs.split()[0] == 'save':
                flag = ISRDB

        # add active knobs when activedefrag is on annotation.
        if "activedefrag" not in columns:
            columns.append("activedefrag")
            # "0" means no
            datas.append("0")
            columns.append("active-defrag-threshold-lower")
            datas.append(10)
            columns.append("active-defrag-threshold-upper")
            datas.append(100)
            columns.append("active-defrag-cycle-min")
            datas.append(5)
            columns.append("active-defrag-cycle-max")
            datas.append(75)

        def str2Numbers(str: str)-> Any:
            try:
                number = int(str)
            except:
                number = float(str)
            return number

        datas = list(map(str2Numbers,datas))
        if flag == ISRDB:
            RDB_datas.append(datas)
            RDB_columns.append(columns)
            RDB_rowlabels.append(m)
        elif flag == ISAOF: 
            AOF_datas.append(datas)
            AOF_columns.append(columns)
            AOF_rowlabels.append(m)

    knobs_list = {}

    if len(RDB_columns):
        dict_RDB['data'] = np.array(RDB_datas)
        dict_RDB['rowlabels'] = np.array(RDB_rowlabels)
        dict_RDB['columnlabels'] = np.array(RDB_columns[0])
        knobs_list = dict_RDB

    if len(AOF_columns):
        dict_AOF['data'] = np.array(AOF_datas)
        dict_AOF['rowlabels'] = np.array(AOF_rowlabels)
        dict_AOF['columnlabels'] = np.array(AOF_columns[0])
        knobs_list = dict_AOF

    return knobs_list

def generate_config(args, top_k_knobs, final_solution_pool):
    """ Generate config file according to the top_k_knobs and final_solution_pool which are generated by GA

    Args:
        top_k_knobs (list[str]): the name of top k knobs
        final_solution_pool (DataFrame): the final value of top k knobs
    
    Return:
        config file: contain the top k config instead of original config
    """
    initial_config = json.load(open(f'../data/{args.persistence.lower()}_knobs.json', 'r'))

    top_k_knobs_value = list(final_solution_pool.iloc[0, 0:args.topk])
    top_k_knobs_config = {value: top_k_knobs_value[i]
                        for i, value in enumerate(top_k_knobs)}
    memory_dict, active_dict = dict(), dict()
    for key, value in initial_config.items():
        if 'memory' in key:
            memory_dict[key] = value
        elif 'active-defrag' in key:
            active_dict[key] = value
        if value == 'yes' or value == 'no':
            if key in top_k_knobs_config.keys():
                if top_k_knobs_config[key] == 0:
                    top_k_knobs_config[key] = 'no'
                else:
                    top_k_knobs_config[key] = 'yes'
    policy_dict = ["volatile-lru", "allkeys-lru", "volatile-lfu", "allkeys-lfu", 
                "volatile-random", "allkeys-random", "volatile-ttl", "noeviction"]
    fsync_dict = ["always", "everysec", "no"]
    # the rule of activedefrag related config
    if 'activedefrag' in top_k_knobs_config.keys():
        if top_k_knobs_config['activedefrag'] == 'no':
            for key, value in active_dict.items():
                del initial_config[key]
                initial_config["#"+key] = value
        else:
            for key in active_dict.keys():
                if key in top_k_knobs_config.keys():
                    initial_config[key] = int(top_k_knobs_config[key])
                    del top_k_knobs_config[key]
    else:
        for key in active_dict.keys():
            if key in top_k_knobs_config.keys():
                initial_config[key] = int(top_k_knobs_config[key])
                del top_k_knobs_config[key]
    # the rule of maxmemory related config
    for key in memory_dict.keys():
        if key in top_k_knobs_config.keys():
            if key == 'maxmemory':
                if top_k_knobs_config[key] == 1 or top_k_knobs_config[key] == 2 or top_k_knobs_config[key] == 3:
                    top_k_knobs_config[key] = int(top_k_knobs_config[key])
                initial_config[key] = str(top_k_knobs_config[key]) + 'gb'
                del top_k_knobs_config[key]
            elif key == 'maxmemory-policy':
                policy_id = int(top_k_knobs_config[key])
                if policy_id >=7:
                    policy_id = 7
                elif policy_id < 0:
                    policy_id = 0
                initial_config[key] = policy_dict[policy_id]
                del top_k_knobs_config[key]
            elif key == 'maxmemory-samples':
                initial_config[key] = int(top_k_knobs_config[key])
                del top_k_knobs_config[key]
        if 'maxmemory' not in top_k_knobs_config.keys():
            initial_config['maxmemory'] = '32mb' if args.target in range(1, 10) else '1gb'

    # deal with the rest knobs
    if args.persistence == 'RDB':
        for key, value in top_k_knobs_config.items():
            initial_config[key] = value if type(value) is str else int(value)
    elif args.persistence == 'AOF':
        for key, value in top_k_knobs_config.items():
            if key == 'appendfsync':
                initial_config[key] = fsync_dict[int(value)]
                if initial_config[key]=='always':
                    return None, None, False
            elif key == 'auto-aof-rewrite-min-size':
                initial_config[key] = str(int(value)) + 'mb'
            else:
                initial_config[key] = value if type(value) is str else int(value)
    
    i = 0
    today = datetime.datetime.now()
    name = args.persistence+'-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.conf'
    while os.path.exists(os.path.join('./GA_config/', name)):
        i += 1
        name = args.persistence+'-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.conf'
    top_k_config_path = os.path.join('./GA_config/', name)
    
    with open("../data/init_config.conf", "r") as read_file:
        lines = read_file.readlines() + ['\n']
        with open(top_k_config_path, "a+") as write_file:
            write_file.writelines(lines)
            for key, value in initial_config.items():
                if 'sec' in key or 'changes' in key:
                    line += f' {value}'
                    i += 1
                    if i < 2: 
                        continue
                    else:
                        write_file.write(line + '\n')
                else:
                    line = f'{key} ' + f'{value}'
                    write_file.write(line + '\n')
                i, line = 0, 'save'
    
    return top_k_config_path, name, True


def aggregate_datas(metric_datas: dict) -> dict:
    """
    Aggregate Internal Metrics from workloads in key 'data'.
    """
    aggregated_data = {}
    for workload in metric_datas.keys():
        if workload.startswith('workload'):
            if not (aggregated_data.get('data') is None):
                aggregated_data['data'] = np.concatenate((aggregated_data['data'], metric_datas[workload]))
            else:
                aggregated_data['data'] = metric_datas[workload]
        else:
            aggregated_data[workload] = metric_datas[workload]
    return aggregated_data


def metric_preprocess(metrics: DataFrame) -> Tuple[DataFrame, dict]:
    '''To invert for categorical internal metrics'''
    dict_le = {}
    c_metrics = metrics.copy()

    for col in metrics.columns:
        if isinstance(c_metrics[col][0], str):
            le = LabelEncoder()
            c_metrics[col] = le.fit_transform(c_metrics[col])
            dict_le[col] = le
    return c_metrics, dict_le

def metrics_make_dict(pd_metrics: DataFrame, knobs_list = None):
    '''
        input: DataFrame form (samples_num, metrics_num)
        output: Dictionary form
            ex. dict_metrics = {'columnlabels'= array([['metrics_1', 'metrics_2', ...],['metrics_1', 'metrics_2', ...], ...]),
                            'rowlabels'= array([1, 2, ...]),
                            'data'= array([[1,2,3,...], [2,3,4,...], ...[]])}
    '''
    # labels = RDB or AOF rowlabels
    
    dict_metrics = {}
    nan_columns = pd_metrics.columns[pd_metrics.isnull().any()]
    pd_metrics = pd_metrics.drop(columns=nan_columns)
    dict_metrics['columnlabels'] = np.array(pd_metrics.columns)

    if knobs_list:
        dict_metrics['rowlabels'] = np.array(knobs_list['rowlabels'])
    else:
        dict_metrics['rowlabels'] = np.array(range(len(pd_metrics)))
    dict_metrics['data'] = np.array(pd_metrics.values)
    
    return dict_metrics    

def load_knob_metrics(metric_path: str, knobs_path: str, persistence: str=None, metrics: list=None):
    """ 
    If metrics is None, it means internal metrics.
    """
    if metrics is None:
        pd_metrics = pd.read_csv(metric_path)
        knob_list = knobs_make_dict(knobs_path, pd_metrics, persistence)
        pd_metrics, _ = metric_preprocess(pd_metrics)
        return knob_list, metrics_make_dict(pd_metrics, knob_list)
    else:
        pd_metrics = pd.read_csv(metric_path)
        return metrics_make_dict(pd_metrics[metrics])
