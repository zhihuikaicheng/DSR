from __future__ import print_function, division

import argparse
import scipy.io as sio
import numpy as np
import time
import os
from utils.evaluation import cmc, mean_ap
from utils.evaluation import compute_dist2, dsr_dist
import pdb

#args
##############################################
parser = argparse.ArgumentParser()
parser.add_argument('--gallery_dir', type=str)
parser.add_argument('--query_dir', type=str)
parser.add_argument('--use_multi_scale', action='store_true')

args = parser.parse_args()

#######################################################################
# Evaluate

def loadmat(file_dir, is_multi_scale=False):
    for filename in os.listdir(file_dir):
        if ('multi' in filename) and (is_multi_scale):
            mat = sio.loadmat(os.path.join(file_dir, filename))
        elif not (('multi' in filename) or (is_multi_scale)):
            mat = sio.loadmat(os.path.join(file_dir, filename))
        feats = mat['feat']
        labels = mat['labels']
        cams = mat['cams']
    return feats, labels, cams

gallery_feature, gallery_label, gallery_cam = loadmat(args.gallery_dir, args.use_multi_scale)
query_feature, query_label, query_cam = loadmat(args.query_dir, args.use_multi_scale)

q_g_dist1 = compute_dist2(query_feature, gallery_feature, type='euclidean')
q_g_dist2 = dsr_dist(query_feature, gallery_feature, type='euclidean')

for ad in range(0, 11):
    lam = ad*0.1
    q_g_dist = (1-lam) * q_g_dist1 + lam * q_g_dist2
    res_rank = cmc(distmat, query_label, gallery_label, query_cam, gallery_cam, first_match_break=True)
    res_map = mean_ap(distmat, query_label, gallery_label, query_cam, gallery_cam)
    print('ad:{:d}, {:<30}'.format(ad, 'Single Query:'), end='')
    print(res_rank)
    print(res_map)


# distmat = dsr_dist(query_feature, gallery_feature)

# res_rank = cmc(distmat, query_label, gallery_label, query_cam, gallery_cam, first_match_break=True)
# res_map = mean_ap(distmat, query_label, gallery_label, query_cam, gallery_cam)

# print(res_rank)
# print(res_map)
