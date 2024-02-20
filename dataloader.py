import japanese_clip as ja_clip
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

from PIL import Image
from typing import Tuple
import math
import os

from types import SimpleNamespace


class MsCocoDataset(Dataset):
    def __init__(self, image_root: str, caption_df: pd.DataFrame, device: str, transforms: nn.Transformer = None) -> None:
        super().__init__()
        self.transforms = transforms
        self.image_root = image_root
        self.caption_df = caption_df
        self.device = device

        clip_model, _ = ja_clip.load("rinna/japanese-cloob-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
        self.clip_model = clip_model
        self.clip_tokenizer = ja_clip.load_tokenizer()


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        row = self.caption_df.iloc[index,]

        # mscocoのデータはかなり膨大。Google Driveで１つのフォルダにたくさんファイルがあると、ファイル操作でタイムアウトが発生することがあるので、
        # image_idをもとにフォルダ分けする（１フォルダ、数千個程度）
        img_subdir = "{:06d}".format(math.floor(row.image_id / 100000) * 100000)
        img = Image.open(os.path.join(self.image_root, img_subdir, row.file_name))
        if self.transforms is not None:
            img = self.transforms(img)


        encodings = ja_clip.tokenize(
                        texts=[row.caption],
                        max_seq_len=77,
                        device=self.device,
                        tokenizer=self.clip_tokenizer, # this is optional. if you don't pass, load tokenizer each time
                     )
        with torch.no_grad():
            embed = self.clip_model.encode_text(encodings['input_ids']).squeeze().type(torch.float)
        #print(f'loader embed: {embed.shape}')
        return img, embed

    def __len__(self) -> int:
        return len(self.caption_df)


def get_dataloader(cfg: SimpleNamespace, df: pd.DataFrame) -> DataLoader:
    train_transforms = torchvision.transforms.Compose([
        T.Resize(cfg.img_size + int(.25 * cfg.img_size)),  # img_size + 1/4 *img_size
        T.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x ), # モノクロ画像を３チャネルに
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_loader = DataLoader(
        dataset=MsCocoDataset(cfg.image_root, df, train_transforms),
        batch_size=cfg.batch_size,
        shuffle=True
    )
    return train_loader