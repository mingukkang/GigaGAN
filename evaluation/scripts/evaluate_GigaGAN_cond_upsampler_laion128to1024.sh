pip install --upgrade --no-cache-dir gdown

gdown https://drive.google.com/uc?id=1fZb8Z6jVykeuc3sXWc9w-d05Pt_DGNRk
tar -zxvf Real_laion1024.tar.gz

pip install --upgrade --no-cache-dir gdown

gdown https://drive.google.com/uc?id=1FFfoYxWVLw0TKwcmeYc_uD_pj4tR5Wrv
tar -zxvf GigaGAN_cond_upsampler_laion128to1024.tar.gz

how_many=10000
ref_data="laion4k"
ref_dir="./Real_laion1024"
ref_type="valid"
fake_dir="./GigaGAN_cond_upsampler_laion128to1024"
eval_res=1024
batch_size=8

CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --how_many $how_many --ref_data $ref_data --ref_dir $ref_dir --ref_type $ref_type --fake_dir $fake_dir --eval_res $eval_res --batch_size $batch_size