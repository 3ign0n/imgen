import copy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from fastprogress import master_bar, progress_bar
import numpy as np

import japanese_clip as ja_clip
from utils import plot_images, mk_folders
import os

class EMA:
    def __init__(self, beta, device):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.device = device

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict(), map_location=self.device)


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        #print('DoubleConv forward')
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        #print('Down forward')
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        #print('Up forward')
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, device, time_dim, clip_dim, c_in=3, c_out=3):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.clip_dim = clip_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        #print(f'UNet_conditional init, {clip_dim}, {time_dim}, {c_in}, {c_out}')
        self.label = nn.Linear(clip_dim, time_dim)


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        #print(f'UNet t1:{t.shape}')
        t = t.unsqueeze(-1).type(torch.float)
        #print(f'UNet t2:{t.shape}')
        t = self.pos_encoding(t, self.time_dim)

        #print(f'UNet t3:{t.shape}')
        if y is not None:
            #print(f'UNet y: {y.shape}, clip_dim: {self.clip_dim}, time_dim: {self.time_dim}')
            #print(self.label)
            lm = self.label(y)
            #print(f'UNet lm: {lm.shape}')
            t += lm

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output



class Diffusion:
    def __init__(self, config, beta_start=1e-4, beta_end=0.02, c_in=3, c_out=3, **kwargs):
        self.config = config

        self.noise_steps = config.noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = config.img_size
        self.model = UNet_conditional(config.device, config.time_dim, config.clip_dim, c_in, c_out, **kwargs).to(config.device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.lr,
                                                 steps_per_epoch=config.steps_per_epoch, epochs=config.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995, config.device)
        self.scaler = torch.cuda.amp.GradScaler()

        self.device = config.device
        self.c_in = c_in

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast(self.device) and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                #    print(f'train labels: None')
                #else:
                #    print(f'train labels: {labels.shape}')
                #print(f'x_t: {x_t.shape}, t: {t.shape}')
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
            pbar.comment = f"loss={loss.item():2.3f}"
            return avg_loss.mean().item()

    def get_embeds(self, captions: list) -> torch.Tensor:
        model, _ = ja_clip.load("rinna/japanese-cloob-vit-b-16", cache_dir="/tmp/japanese_clip", device=self.config.device)
        tokenizer = ja_clip.load_tokenizer()
        encodings = ja_clip.tokenize(
            texts=captions,
            max_seq_len=77,
            device=self.device,
            tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
        )
        with torch.no_grad():
            embed = model.encode_text(encodings['input_ids'])
        return embed

    def log_images(self):
        "Log images and save them to disk"
        captions = ['黒い猫', '向かい合った犬と猫', '飛行機みたいな鳥', '野菜でできた魚']
        labels = self.get_embeds(captions).to(self.device)
        #print(f'labels: {labels.shape}')
        sampled_images = self.sample(use_ema=False, labels=labels)
        plot_images(sampled_images)  #to display on jupyter if available
        # EMA model sampling
        # ema_sampled_images = self.sample(use_ema=True, labels=labels)

    def load(self, device, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt", optim_ckpt="optim.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt), map_location=device))
        self.optimizer.load_state_dict(torch.load(os.path.join(model_cpkt_path, optim_ckpt), map_location=device))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt), map_location=device))

    def save_model(self, run_name, epoch=-1):
        "Save model locally"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))

    def prepare(self, args, train_loader):
        mk_folders(args.run_name)
        self.train_dataloader = train_loader

    def fit(self, args):
        epoch_train_loss = []
        x_bounds = [0, args.epochs]
        y_bounds = None

        mb = master_bar(range(args.epochs))
        for epoch in mb:
            train_loss = self.one_epoch(train=True)

            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images()

            # 学習過程の出力
            epoch_train_loss.append(train_loss)

            if y_bounds is None:
                y_bounds = [0, train_loss * 1.1]

            graph_data = [[np.arange(len(epoch_train_loss)), epoch_train_loss]]
            mb.update_graph(graph_data, x_bounds, y_bounds)

        # save model
        self.save_model(run_name=args.run_name, epoch=epoch)
