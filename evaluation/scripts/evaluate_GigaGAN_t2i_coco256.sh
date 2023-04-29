pip install --upgrade --no-cache-dir gdown

gdown https://drive.google.com/uc?id=1sg180IaL9MS4yAxv8P8e7yqfgSVuUtUF

tar -zxvf GigaGAN_t2i_coco256.tar.gz

how_many=30000
ref_data="coco2014"
ref_dir="/home/COCO2014"
ref_type="val2014"
fake_dir="./GigaGAN_t2i_coco256"
eval_res=256
batch_size=64

CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --how_many $how_many --ref_data $ref_data --ref_dir $ref_dir --ref_type $ref_type --fake_dir $fake_dir --eval_res $eval_res --batch_size $batch_size