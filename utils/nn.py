import os
import random
import functools
import numpy as np

import torch
import torch.nn as nn


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def normalize_img(img):
    return ((img / 255.0) - 0.5) * 2.0


def unnormalize_img(img):
    return ((img / 2.0) + 0.5) * 255.0


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def get_optim_groups(model, weight_decay):
    return [
        # do not decay biases and single-column parameters (rmsnorm), those are usually scales
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {"params": (p for p in model.parameters() if p.dim() >= 2), "weight_decay": weight_decay},
    ]


def get_grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm


def soft_update(target, source, tau=1e-3):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def _linear_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then linear decay to 0
    """
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = 1.0 - ((iteration - warmup_iterations) / (total_iterations - warmup_iterations))
    return multiplier


def linear_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    decay_func = functools.partial(
        _linear_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_func)
    return scheduler


class MLPBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand * dim),
            nn.ReLU6(),
            nn.Linear(expand * dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


# Residual encoder block adapted from the IMPALA CNN architecture.
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2, dropout=0.0, downscale=True):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._downscale = downscale
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self._downscale else 1,
        )
        # conv downsampling is faster that maxpool, with same perf
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.conv(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        if self._downscale:
            return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self._out_channels, h, w)


class DecoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # upsample + conv works fine, just slower than conv-transpose
        # also: upsample does not work well with orthogonal init (why?)!
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.conv = nn.ConvTranspose2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=2, stride=2
        )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels) for _ in range(num_res_blocks)])

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, h * 2, w * 2)


class Actor(nn.Module):
    def __init__(
        self,
        shape,
        num_actions,
        encoder_scale=1,
        encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=1,
        dropout=0.0,
    ):
        super().__init__()
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.final_encoder_shape = shape
        self.encoder = nn.Sequential(
            *conv_stack,
            # nn.Flatten(),
        )
        self.actor_mean = nn.Sequential(
            nn.ReLU6(),
            # works either way...
            # nn.Linear(math.prod(shape), num_actions),
            nn.Linear(shape[0], num_actions),
        )
        self.num_actions = num_actions
        self.apply(weight_init)

    def forward(self, obs):
        out = self.encoder(obs)
        out = out.flatten(2).mean(-1)
        act = self.actor_mean(out)
        return act, out


class ActionDecoder(nn.Module):
    def __init__(self, obs_emb_dim, latent_act_dim, true_act_dim, hidden_dim=128, use_state=True):
        super().__init__()
        self.obs_emb_dim = obs_emb_dim
        self.latent_act_dim = latent_act_dim
        self.true_act_dim = true_act_dim
        self.use_state = use_state

        # Dynamically set the input dimension based on the flag
        input_dim = latent_act_dim + obs_emb_dim if self.use_state else latent_act_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, true_act_dim),
        )

    def forward(self, obs_emb, latent_act):
        if self.use_state:
            # Concatenate the observation embedding (state) with the latent action
            hidden = torch.cat([obs_emb, latent_act], dim=-1)
            true_act_pred = self.model(hidden)
        else:
            # Only use the latent action
            true_act_pred = self.model(latent_act)
            
        return true_act_pred
