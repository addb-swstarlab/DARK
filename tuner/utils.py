# -*- coding: utf-8 -*-

import json
import random
import numpy as np

import os
import logging
import logging.handlers

import datetime

import torch

def get_logger(log_path='./logs'):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format)
    i = 0
    today = datetime.datetime.now()
    name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    while os.path.exists(os.path.join(log_path, name)):
        i += 1
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)


def make_date_dir(path: str) -> str:
    """
    :param path
    :return: os.path.join(path, date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d')+'-'+'%02d' % i

    while os.path.exists(os.path.join(path, name)):
        i += 1
        name = today.strftime('%Y%m%d')+'-'+'%02d' % i
        
    os.mkdir(os.path.join(path, name))
    return os.path.join(path, name)


def get_ranked_knob_data(ranked_knobs: list, knob_data: dict, top_k: int) -> dict:
    '''
        ranked_knobs: sorted knobs with ranking 
                        ex. ['m3', 'm6', 'm2', ...]
        knob_data: dictionary data with keys(columnlabels, rowlabels, data)
        top_k: A standard to split knobs 
    '''
    ranked_knob_data = knob_data.copy()
    ranked_knob_data['columnlabels'] = np.array(ranked_knobs)
        
    for i, knob in enumerate(ranked_knobs):
        ranked_knob_data['data'][:,i] = knob_data['data'][:, list(knob_data['columnlabels']).index(knob)]
    
    # pruning with top_k
    ranked_knob_data['data'] = ranked_knob_data['data'][:,:top_k]
    ranked_knob_data['columnlabels'] = ranked_knob_data['columnlabels'][:top_k]

    #print('pruning data with ranking')
    #print('Pruned Ranked knobs: ', ranked_knob_data['columnlabels'])

    return ranked_knob_data

def collate_function(examples):
    knobs=[None]*len(examples)
    EMs=[None]*len(examples)
    for i,(knob,EM) in enumerate(examples):
        knobs[i] = knob
        EMs[i] = EM
    return torch.tensor(knobs),torch.tensor(EMs)

def make_random_option(top_k_knobs):
    with open('../data/test_range.json','r') as f:
        data = json.load(f)
    option = {}
    for top_k_knob in top_k_knobs:
        if data[top_k_knob][0] == 'categorical':
            option[top_k_knob] = data[top_k_knob][-1].index(random.choice(data[top_k_knob][-1]))
        elif data[top_k_knob][0] == 'boolean':
            option[top_k_knob] = random.choice([0,1])
        elif data[top_k_knob][0] == 'integer':
            option[top_k_knob] = random.choice(range(data[top_k_knob][-1][0],data[top_k_knob][-1][1]+1))
        else:
            option[top_k_knob] = round(random.choice(np.arange(data[top_k_knob][-1][0],data[top_k_knob][-1][1]+0.1,0.1)),1)
    return option

def config_exist(persistence):
    i = 0
    PATH = '../data/redis_data/config_results/{}'.format(persistence)
    NAME = persistence+'_rec_config{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH,NAME)):
        i+=1
        NAME = persistence+'_rec_config{}.conf'.format(i)
    return NAME[:-5]