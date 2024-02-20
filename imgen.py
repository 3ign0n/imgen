import sys

import config
from seed import seed_everything
from net import Diffusion
from utils import save_images

from datetime import datetime

def genimg(prompt: str):
    cfg = config.get_config()
    seed_everything(cfg.seed)

    
    model = Diffusion(config=cfg)
    model.load(model_cpkt_path="models", map_location=cfg.device)

    embeds = model.get_embeds([prompt])
    samples = model.sample(use_ema=False, labels=embeds, cfg_scale=6)
    #plot_images(samples)
    save_images(samples, f"output/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")

if __name__ == '__main__':
    prompt = sys.argv[1]
    genimg(prompt)
