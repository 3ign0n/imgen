from types import SimpleNamespace
from datetime import datetime
import torch

__config = SimpleNamespace(
    run_name = "ddpm" + datetime.now().strftime('%Y%m%d_%H%M%S'),
    epochs = 200,
    steps_per_epoch = 30080,
    noise_steps=1000,
    seed = 42,
    batch_size = 16,
    img_size = 32,
    num_channels = 3,
    time_dim = 256,
    clip_dim = 512,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    slice_size = 1,
    do_validation = False,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3,
    image_root = ''
    )

def get_config() -> SimpleNamespace:
    return __config
