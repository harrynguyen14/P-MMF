import os
import argparse
import numpy as np
import pandas as pd

from reranking import P_MMF_CPU, P_MMF_GPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='bpr')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument('--dataset-dir', type=str, default='datasets')
    parser.add_argument('--Time', type=int, default=256,
                        help='fair time sep.')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--gpu', type=str, default="False")


    args = parser.parse_args()
    print("Using GPU:",args.gpu)
    if args.gpu == "True":
        P_MMF_GPU(lambd=1e-1,args=args)
    else:
        P_MMF_CPU(lambd=1e-1,args=args)

    
    