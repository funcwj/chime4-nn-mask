#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.8

import argparse
import os

import torch as th
import torch.utils.data as data

from model import EstimatorTrainer
from dataset import MaskDataset, collate_func

def train(args):
    dataset = MaskDataset(args.data_dir, args.num_jobs, training=True)
    tr_loader = data.DataLoader(dataset=dataset, collate_fn=collate_func, shuffle=True)
    dataset = MaskDataset(args.data_dir, args.num_jobs, training=False)
    dt_loader = data.DataLoader(dataset=dataset, collate_fn=collate_func, shuffle=True)
    
    estimator = EstimatorTrainer(513, args.checkout_dir, learning_rate=args.lr)
    estimator.train(tr_loader, dt_loader, epoch=args.epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command to train a mask estimator")
    parser.add_argument("data_dir", type=str, 
                        help="root directory of the training & evaluate data") 
    parser.add_argument("--epoch", type=int, dest="epoch", default=10,
                        help="number of epoch to train the model")
    parser.add_argument("--nj", type=int, dest="num_jobs", default=15,
                        help="number of jobs to generate the dataset")
    parser.add_argument("--lr", type=float, dest="lr", default=0.001,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--checkout-dir", type=str, dest="checkout_dir", default='.',
                        help="directory to save model parameters")
    args = parser.parse_args()
    train(args) 

