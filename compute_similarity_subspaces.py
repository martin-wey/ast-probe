import glob
import argparse
import os
import pickle

import torch
from scipy.linalg import subspace_angles
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Script for generating the plots of the paper')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--layer', default=4, help='Layer')
    parser.add_argument('--model', default='codebert', help='Model')
    parser.add_argument('--rank', default='128', help='Rank')
    args = parser.parse_args()
    dic_models = {}
    for file in glob.glob(args.run_dir + "/*/metrics.log"):
        parent = os.path.dirname(file).split('/')[-1]
        rq4 = False
        if '_rq4' not in parent:
            model, lang, layer, rank = parent.split('_')
        else:
            model, lang, layer, rank, _ = parent.split('_')
            rq4 = True
        with open(file, 'rb') as f:
            results = pickle.load(f)
        if model == 'codebert0':
            model = 'codebert-baseline'

        checkpoint = torch.load(os.path.join(os.path.dirname(file), 'pytorch_model.bin'),
                                map_location=torch.device('cpu'))
        proj = checkpoint['proj'].cpu().detach().numpy()
        dic_models['#'.join((model, lang, str(layer), str(rank)))] = proj


    langs = ['python', 'javascript', 'go']
    current_model = args.model
    current_layer = args.layer
    current_rank = args.rank
    for l1 in langs:
        for l2 in langs:
            p1 = dic_models['#'.join((current_model, l1, str(current_layer), str(current_rank)))]
            p2 = dic_models['#'.join((current_model, l2, str(current_layer), str(current_rank)))]
            print(f'{l1} vs {l2}, {np.mean(subspace_angles(p1,p2))}, {np.rad2deg(np.mean(subspace_angles(p1,p2)))}')


if __name__ == '__main__':
    main()