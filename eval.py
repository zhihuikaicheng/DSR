from __future__ import print_function

import argparse
import scipy.io as sio
import numpy as np
import time
import os
from utils.evaluation import cmc, mean_ap
import pdb

#args
##############################################
parser = argparse.ArgumentParser()
parser.add_argument('--gallery_dir', type=str)
parser.add_argument('--query_dir', type=str)
parser.add_argument('--use_multi_scale', type=bool, default=False)

args = parser.parse_args()

#######################################################################
# Evaluate

def loadmat(file_dir):
    for filename in os.listdir(file_dir):
        if ('multi' in filename) and (args.use_multi_scale==True):
            mat = sio.loadmat(os.path.join(file_dir, filename))
        elif not (('multi' in filename) or (args.use_multi_scale==True)):
            mat = sio.loadmat(os.path.join(file_dir, filename))
        pdb.set_trace()
    return feats, labels, cams

gallery_feature, gallery_label, gallery_cam = loadmat(args.gallery_dir)
query_feature, query_label, query_cam = loadmat(args.query_dir)

# distmat = dsr_dist(query_feature, gallery_feature)

res_rank = cmc(distmat, query_label, gallery_label, query_cam, gallery_cam, first_match_break=True)
res_map = mean_ap(distmat, query_label, gallery_label, query_cam, gallery_cam)

print(res_rank)
print(res_map)


# def evaluate(qf,ql,qc,gf,gl,gc):
#     query = qf
#     score = np.dot(gf,query)
#     # predict index
#     index = np.argsort(score)  #from small to large
#     index = index[::-1]
#     #index = index[0:2000]
#     # good index
#     query_index = np.argwhere(gl==ql)
#     camera_index = np.argwhere(gc==qc)

#     good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
#     junk_index1 = np.argwhere(gl==-1)
#     junk_index2 = np.intersect1d(query_index, camera_index)
#     junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
#     CMC_tmp = compute_mAP(index, good_index, junk_index)
#     return CMC_tmp


# def compute_mAP(index, good_index, junk_index):
#     ap = 0
#     cmc = torch.IntTensor(len(index)).zero_()
#     if good_index.size==0:   # if empty
#         cmc[0] = -1
#         return ap,cmc

#     # remove junk_index
#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]

#     # find good_index index
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     rows_good = np.argwhere(mask==True)
#     rows_good = rows_good.flatten()
    
#     cmc[rows_good[0]:] = 1
#     for i in range(ngood):
#         d_recall = 1.0/ngood
#         precision = (i+1)*1.0/(rows_good[i]+1)
#         if rows_good[i]!=0:
#             old_precision = i*1.0/rows_good[i]
#         else:
#             old_precision=1.0
#         ap = ap + d_recall*(old_precision + precision)/2

#     return ap, cmc

# ######################################################################
# result = scipy.io.loadmat('pytorch_result.mat')
# query_feature = result['query_f']
# query_cam = result['query_cam'][0]
# query_label = result['query_label'][0]
# gallery_feature = result['gallery_f']
# gallery_cam = result['gallery_cam'][0]
# gallery_label = result['gallery_label'][0]

# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# #print(query_label)
# for i in range(len(query_label)):
#     ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#     if CMC_tmp[0]==-1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     print(i, CMC_tmp[0])

# CMC = CMC.float()
# CMC = CMC/len(query_label) #average CMC
# print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
