python NewTest.py \
--sys_device_ids='(2,3,4,5)' \
--test_dir='/world/data-gpu-94/sysu-reid/person-reid-data/Market-1501/Market-1501-v15.09.15/pytorch' \
--model_save_dir='/world/data-gpu-94/sysu-reid/checkpoints/DSR_Market' \
--batch_size=8 \
--img_h=256 \
--img_w=128 \
--which_epoch='59' \
--gallery_feature_dir='/world/data-gpu-94/sysu-reid/features_save/DSR_Market/gallery' \
--query_feature_dir='/world/data-gpu-94/sysu-reid/features_save/DSR_Market/query'