python NewTest.py \
--sys_device_ids='7' \
--test_dir='/world/data-gpu-94/sysu-reid/person-reid-data/Market-1501/Market-1501-v15.09.15/pytorch' \
--model_save_dir='/world/data-gpu-94/sysu-reid/checkpoints/DSR_Market_softmax_2' \
--batch_size=64 \
--img_h=256 \
--img_w=128 \
--which_epoch='199' \
--gallery_feature_dir='/world/data-gpu-94/sysu-reid/features_save/DSR_Market_softmax_2/gallery' \
--query_feature_dir='/world/data-gpu-94/sysu-reid/features_save/DSR_Market_softmax_2/query' \
--useCAM
