import glob
import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Script for computing the F norm')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    args = parser.parse_args()

    for file in glob.glob(args.run_dir + "/*/pytorch_model.bin"):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        proj = checkpoint['proj'].cpu().detach().numpy()
        mult = np.matmul(proj.T, proj)
       # print(np.round(mult, 3))
        print(file)
        print('Fro norm', np.linalg.norm(mult - np.eye(mult.shape[0]), 'fro'))
        print('Inf norm', np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf))
        print('Inf norm normalized', np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf)/mult.shape[0])
        print(np.linalg.norm(mult - np.eye(mult.shape[0]), 'fro') < 0.05)
        print('vectors c', checkpoint['vectors_c'].shape)
        print('vectors u', checkpoint['vectors_u'].shape)



if __name__ == '__main__':
    main()
