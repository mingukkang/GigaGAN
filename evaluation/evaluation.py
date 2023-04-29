# GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
# See license file or visit https://github.com/mingukkang/GigaGAN for details

# evaluation.py


import os
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_util import EvalDataset, CenterCropLongEdge



def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image


@torch.no_grad()
def compute_clip_score(
    dataset: DataLoader, clip_model="ViT-B/32", device="cuda", how_many=5000):
    print("Computing CLIP score")
    import clip as openai_clip 
    if clip_model == "ViT-B/32":
        clip, clip_preprocessor = openai_clip.load("ViT-B/32", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-G/14":
        import open_clip
        clip, _, clip_preprocessor = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s12b_b42k")
        clip = clip.to(device)
        clip = clip.eval()
        clip = clip.float()
    else:
        raise NotImplementedError

    cos_sims = []
    count = 0
    for imgs, txts in tqdm(dataset):
        imgs_pil = [clip_preprocessor(tensor2pil(img)) for img in imgs]
        imgs = torch.stack(imgs_pil, dim=0).to(device)
        tokens = openai_clip.tokenize(txts, truncate=True).to(device)
        # Prepending text prompts with "A photo depicts "
        # https://arxiv.org/abs/2104.08718
        prepend_text = "A photo depicts "
        prepend_text_token = openai_clip.tokenize(prepend_text)[:, 1:4].to(device)
        prepend_text_tokens = prepend_text_token.expand(tokens.shape[0], -1)
        
        start_tokens = tokens[:, :1]
        new_text_tokens = torch.cat(
            [start_tokens, prepend_text_tokens, tokens[:, 1:]], dim=1)[:, :77]
        last_cols = new_text_tokens[:, 77 - 1:77]
        last_cols[last_cols > 0] = 49407  # eot token
        new_text_tokens = torch.cat([new_text_tokens[:, :76], last_cols], dim=1)
        
        img_embs = clip.encode_image(imgs)
        text_embs = clip.encode_text(new_text_tokens)

        similarities = F.cosine_similarity(img_embs, text_embs, dim=1)
        cos_sims.append(similarities)
        count += similarities.shape[0]
        if count >= how_many:
            break
    
    clip_score = torch.cat(cos_sims, dim=0)[:how_many].mean()
    clip_score = clip_score.detach().cpu().numpy()
    return clip_score


@torch.no_grad()
def compute_fid(fake_dir: Path, gt_dir: Path,
    resize_size=None, feature_extractor="clip"):
    from cleanfid import fid
    center_crop_trsf = CenterCropLongEdge()
    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np) 
        image_pil = center_crop_trsf(image_pil)

        if resize_size is not None:
            image_pil = image_pil.resize((resize_size, resize_size),
                                         Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError(
            "Unrecognized feature extractor [%s]" % feature_extractor)
    fid = fid.compute_fid(gt_dir,
                          fake_dir,
                          model_name=model_name,
                          custom_image_tranform=resize_and_center_crop)
    return fid


def evaluate_model(opt):
    ### Generated images
    dset2 = EvalDataset(data_name=opt.ref_data,
                        data_dir=opt.fake_dir,
                        data_type="fake_images",
                        crop_long_edge=True,
                        resize_size=opt.eval_res,
                        resizer="lanczos",
                        normalize=True,
                        load_txt_from_file=True if opt.ref_data == "coco2014" else False)

    dset2_dataloader = DataLoader(dataset=dset2,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)

    if opt.ref_data == "coco2014":
        clip_score = compute_clip_score(dset2_dataloader, clip_model=opt.clip_model4eval, how_many=opt.how_many)
        print(f"CLIP score: {clip_score}")

    ref_sub_folder_name = "val2014" if opt.ref_data == "coco2014" else opt.ref_type
    fake_sub_folder_name = "fake_images"
    fid = compute_fid(
        os.path.join(opt.ref_dir, ref_sub_folder_name),
        os.path.join(opt.fake_dir, fake_sub_folder_name),
        resize_size=opt.eval_res,
        feature_extractor="inception")
    print(f"FID_{opt.eval_res}px: {fid}")
    return


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--how_many", default=30000, type=int)
    parser.add_argument("--clip_model4eval", default="ViT-B/32", type=str, help="[WO, ViT-B/32, ViT-G/14]")

    parser.add_argument("--ref_data", default="coco2014", type=str, help="in [imagenet2012, coco2014, laion4k]")
    parser.add_argument("--ref_dir",
                        default="/home/COCO2014/",
                        help="location of the reference images for evaluation")
    parser.add_argument("--ref_type",
                        default="train/valid/test", 
                        help="Type of reference dataset")
    parser.add_argument("--fake_dir",
                        default="/home/GigaGAN_images/",
                        help="location of fake images for evaluation")
    parser.add_argument("--eval_res", default=256, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    
    opt, _ = parser.parse_known_args()
    evaluate_model(opt)