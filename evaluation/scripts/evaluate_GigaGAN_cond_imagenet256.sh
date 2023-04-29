pip install --upgrade --no-cache-dir gdown

gdown https://drive.google.com/uc?id=1vAuTUfkeRX045AMhUPcguxwov2dqAVtJ

tar -zxvf GigaGAN_cond_imagenet256.tar.gz

how_many=50000
ref_data="imagenet2012"
ref_dir="/home/ImageNet"
ref_type="train"
fake_dir="./GigaGAN_cond_imagenet256"
eval_res=256
batch_size=64

CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --how_many $how_many --ref_data $ref_data --ref_dir $ref_dir --ref_type $ref_type --fake_dir $fake_dir --eval_res $eval_res --batch_size $batch_size