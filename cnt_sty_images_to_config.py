import os
from pathlib import Path
import argparse
import glob
import yaml
from tqdm import tqdm
import random

def run(cfg):
    cnt_imgs = list(glob.glob(os.path.join(cfg.cnt, '*.png'))) + list(glob.glob(os.path.join(cfg.cnt, '*.jpg'))) + list(glob.glob(os.path.join(cfg.cnt, '*.jpeg')))
    sty_imgs = list(glob.glob(os.path.join(cfg.sty, '**','*.png'), recursive=True)) + list(glob.glob(os.path.join(cfg.sty, '**','*.jpg'), recursive=True)) + list(glob.glob(os.path.join(cfg.sty, '**', '*.jpeg'), recursive=True))

    print(f'Found {len(cnt_imgs)} content images')
    print(f'Found {len(sty_imgs)} style images')

    samples = []
    for i, cnt_p in enumerate(cnt_imgs):
        for sty_p in sty_imgs:
            if cfg.cnt_prompt_from_name:
                cnt_prompt = Path(cnt_p).stem
            else:
                cnt_prompt = 'An image'
            samples.append({
                'cnt_img_path': os.path.relpath(cnt_p), 'cnt_prompt': cnt_prompt,
                'sty_img_path': os.path.relpath(sty_p), 'sty_prompt': '', 'edit_prompt': ''
            })
    
    with open(os.path.join(cfg.save_to, f'{cfg.name}.yaml'), 'w') as fd:
        yaml.dump(samples, fd, default_flow_style=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', type=str)
    parser.add_argument('--sty', type=str)
    parser.add_argument('--save_to', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--cnt_prompt_from_name', action='store_true')
    args = parser.parse_args()

    run(args)
