from __future__ import print_function, division

import argparse
import scipy.io as sio
import numpy as np
import time
import os
from utils.evaluation import cmc, mean_ap
from utils.evaluation import compute_dist2, dsr_dist
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
#args
##############################################
parser = argparse.ArgumentParser()
parser.add_argument('--gallery_dir', type=str)
parser.add_argument('--query_dir', type=str)

args = parser.parse_args()

#######################################################################
# Evaluate

def loadmat(file_dir):
    feats = []
    labels = []
    cams = []
    for filename in os.listdir(file_dir):
        if 'multi' in filename:
            mat = sio.loadmat(os.path.join(file_dir, filename))
            feats.append(mat['feat'])
            labels.append(mat['label'].squeeze(0))
            cams.append(mat['cam'].squeeze(0))

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    cams = np.concatenate(cams, axis=0)
    return feats, labels, cams

def loadsinglemat(file_dir):
    feats = []
    labels = []
    cams = []
    for filename in os.listdir(file_dir):
        if 'multi' not in filename:
            mat = sio.loadmat(os.path.join(file_dir, filename))
            feats.append(mat['feat'])
            labels.append(mat['label'].squeeze(0))
            cams.append(mat['cam'].squeeze(0))

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    cams = np.concatenate(cams, axis=0)
    return feats, labels, cams

gallery_single_feature, _, _ = loadsinglemat(args.gallery_dir)
query_single_feature, _, _ = loadsinglemat(args.query_dir)

gallery_feature, gallery_label, gallery_cam = loadmat(args.gallery_dir)
query_feature, query_label, query_cam = loadmat(args.query_dir)

q_g_dist1 = compute_dist2(query_single_feature, gallery_single_feature, type='euclidean')
q_g_dist2 = dsr_dist(query_feature, gallery_feature, type='euclidean')

for ad in range(0, 11):
    lam = ad*0.1
    q_g_dist = (1-lam) * q_g_dist1 + lam * q_g_dist2
    res_rank = cmc(q_g_dist, query_label, gallery_label, query_cam, gallery_cam, first_match_break=True)
    res_map = mean_ap(q_g_dist, query_label, gallery_label, query_cam, gallery_cam)
    print('''ad:{:d}'''.format(ad))
    print(res_rank)
    print('''''')
    print(res_map)


# distmat = dsr_dist(query_feature, gallery_feature)

# res_rank = cmc(distmat, query_label, gallery_label, query_cam, gallery_cam, first_match_break=True)
# res_map = mean_ap(distmat, query_label, gallery_label, query_cam, gallery_cam)

# print(res_rank)
# print(res_map)
