from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from utils.nn import DecoderBlock, EncoderBlock, weight_init
from otf_vqvae.model import OTFVQVAE, make_activation, make_motion_signal


def _cfg_get(cfg: Mapping | DictConfig, key: str, default):
    if isinstance(cfg, DictConfig):
        return cfg[key] if key in cfg else default
    return cfg.get(key, default)


class FrozenOTFVQVAEFactorExtractor(nn.Module):
    """Frozen OTF-VQ-VAE encoder/codebook wrapper for motion factor extraction."""

    def __init__(self, otf_vqvae: OTFVQVAE) -> None:
        super().__init__()
        self.otf_vqvae = otf_vqvae
        self.otf_vqvae.eval()
        for parameter in self.otf_vqvae.parameters():
            parameter.requires_grad_(False)

    @property
    def grid_size(self) -> Tuple[int, int]:
        return int(self.otf_vqvae.encoder.grid_height), int(self.otf_vqvae.encoder.grid_width)

    @property
    def num_codes(self) -> int:
        return int(self.otf_vqvae.num_codes)

    @property
    def code_dim(self) -> int:
        return int(self.otf_vqvae.cfg.latent_dim)

    def _motion_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        future = batch["future"] if "future" in batch else batch["next"]
        motion_batch = {
            "current": batch["current"],
            "next": future,
        }
        if self.otf_vqvae.motion_input_type == "acceleration":
            if "previous" not in batch:
                raise KeyError(
                    "previous frame is required because the frozen OTF-VQ-VAE uses acceleration motion input"
                )
            motion_batch["previous"] = batch["previous"]
        return motion_batch

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.otf_vqvae.eval()
        motion = make_motion_signal(
            self._motion_batch(batch),
            self.otf_vqvae.motion_input_type,
            self.otf_vqvae.motion_transform,
        )
        patch_embeddings = self.otf_vqvae.encoder(motion)
        vq_output = self.otf_vqvae.quantizer(patch_embeddings)

        indices_flat = vq_output["indices"]
        batch_size, num_patches = indices_flat.shape
        grid_height, grid_width = self.grid_size
        if num_patches != grid_height * grid_width:
            raise RuntimeError(
                f"Expected {grid_height * grid_width} patches, got {num_patches}"
            )

        one_hot = F.one_hot(indices_flat, num_classes=self.num_codes).to(patch_embeddings.dtype)
        counts = one_hot.sum(dim=1)
        weights = counts / float(num_patches)
        occupancy = one_hot.transpose(1, 2).reshape(
            batch_size,
            self.num_codes,
            grid_height,
            grid_width,
        )
        active_mask = counts > 0

        return {
            "indices": indices_flat.reshape(batch_size, grid_height, grid_width),
            "occupancy": occupancy,
            "weights": weights,
            "codebook": self.otf_vqvae.quantizer.embedding.detach(),
            "active_mask": active_mask,
            "motion": motion,
        }


# ---------------------------------------------------------------------------
# FiLM primitives
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for tensors with arbitrary leading batch dimensions.

    Residual form: output = (1 + gamma) * x + beta, where gamma and beta are
    predicted from a condition tensor that shares the same leading dimensions as x.
    Both x and condition must agree on all dimensions except the last.
    """

    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(int(condition_dim), int(feature_dim))
        self.beta_proj = nn.Linear(int(condition_dim), int(feature_dim))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)
        return (1.0 + gamma) * x + beta


class FiLMLayer2D(nn.Module):
    """
    Feature-wise Linear Modulation for spatial [B, C, H, W] tensors.

    The condition is a global vector of shape [B, condition_dim].  The predicted
    per-channel gamma and beta scalars are broadcast over the H and W dimensions.
    """

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(int(condition_dim), int(channels))
        self.beta_proj = nn.Linear(int(condition_dim), int(channels))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(condition).view(x.shape[0], -1, 1, 1)
        beta = self.beta_proj(condition).view(x.shape[0], -1, 1, 1)
        return (1.0 + gamma) * x + beta


class SpatialFiLMLayer(nn.Module):
    """
    Spatially-varying FiLM for [B, C, H, W] FDM decoder feature maps.

    The conditioning signal combines a spatially tiled z_action projection with
    the (bilinearly resized) state features.  A lightweight 1x1-conv head predicts
    per-pixel gamma and beta maps, enabling the action and current observation to
    modulate every spatial location of the decoded features independently.
    """

    def __init__(self, z_dim: int, state_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.z_proj = nn.Linear(int(z_dim), int(feature_channels))
        comb_channels = int(feature_channels) + int(state_channels)
        self.film_conv = nn.Sequential(
            nn.Conv2d(comb_channels, int(feature_channels), kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(int(feature_channels), 2 * int(feature_channels), kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        # Broadcast z_action across the spatial grid
        z_spatial = self.z_proj(z).view(B, -1, 1, 1).expand(-1, -1, H, W)
        # Resize state features to match the current decoder resolution
        if tuple(state.shape[-2:]) != (H, W):
            state = F.interpolate(state, size=(H, W), mode="bilinear", align_corners=False)
        combined = torch.cat([z_spatial, state.to(x.dtype)], dim=1)
        film_params = self.film_conv(combined)
        gamma, beta = film_params.chunk(2, dim=1)
        return (1.0 + gamma) * x + beta


# ---------------------------------------------------------------------------
# IDM component modules
# ---------------------------------------------------------------------------

class SmallCNNOccupancyEncoder(nn.Module):
    """
    Per-code CNN occupancy encoder, FiLM-conditioned on the global state at
    every convolutional layer.  Uses ReLU6 activations.
    """

    def __init__(
        self,
        embed_dim: int,
        state_feature_dim: int,
        activation: str = "gelu",  # kept for API compatibility; ReLU6 is used internally
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU6()
        self.film1 = FiLMLayer2D(int(state_feature_dim), 8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.act2 = nn.ReLU6()
        self.film2 = FiLMLayer2D(int(state_feature_dim), 16)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, int(embed_dim))

    def forward(self, occupancy: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            occupancy:    [B, K, H_grid, W_grid] binary occupancy maps.
            global_state: [B, state_feature_dim] mean-pooled state features.
        Returns:
            [B, K, embed_dim] per-code occupancy embeddings.
        """
        batch_size, num_codes, grid_height, grid_width = occupancy.shape
        x = occupancy.reshape(batch_size * num_codes, 1, grid_height, grid_width)
        # Expand global_state [B, D] → [B*K, D] so each code shares the same condition.
        gs = global_state.unsqueeze(1).expand(-1, num_codes, -1).reshape(batch_size * num_codes, -1)
        x = self.film1(self.act1(self.conv1(x)), gs)
        x = self.film2(self.act2(self.conv2(x)), gs)
        x = self.linear(self.flatten(self.pool(x)))
        return x.reshape(batch_size, num_codes, -1)


class MLPOccupancyEncoder(nn.Module):
    """
    Per-code MLP occupancy encoder, FiLM-conditioned on the global state after
    the hidden layer.  Uses ReLU6 activations.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        embed_dim: int,
        hidden_dim: int,
        state_feature_dim: int,
        activation: str = "gelu",  # kept for API compatibility; ReLU6 is used internally
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(int(grid_height) * int(grid_width), int(hidden_dim))
        self.act1 = nn.ReLU6()
        self.film1 = FiLMLayer(int(state_feature_dim), int(hidden_dim))
        self.linear2 = nn.Linear(int(hidden_dim), int(embed_dim))

    def forward(self, occupancy: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            occupancy:    [B, K, H_grid, W_grid] binary occupancy maps.
            global_state: [B, state_feature_dim] mean-pooled state features.
        Returns:
            [B, K, embed_dim] per-code occupancy embeddings.
        """
        batch_size, num_codes, grid_height, grid_width = occupancy.shape
        flat = occupancy.reshape(batch_size * num_codes, grid_height * grid_width)
        gs = global_state.unsqueeze(1).expand(-1, num_codes, -1).reshape(batch_size * num_codes, -1)
        x = self.film1(self.act1(self.linear1(flat)), gs)
        x = self.linear2(x)
        return x.reshape(batch_size, num_codes, -1)


class FactorEmbedding(nn.Module):
    """
    Maps each OTF-VQ-VAE code (codebook vector + usage weight + occupancy embedding)
    to a per-factor embedding.  Every hidden layer is FiLM-conditioned on the
    concatenation of the global state and the per-factor occupancy embedding.
    Uses ReLU6 activations.
    """

    def __init__(
        self,
        code_dim: int,
        occupancy_embed_dim: int,
        state_feature_dim: int,
        hidden_dim: int,
        embed_dim: int,
        activation: str = "gelu",  # kept for API compatibility; ReLU6 is used internally
    ) -> None:
        super().__init__()
        input_dim = int(code_dim) + 1 + int(occupancy_embed_dim)
        # FiLM condition: global_state ∥ occupancy_embedding, both per factor
        film_cond_dim = int(state_feature_dim) + int(occupancy_embed_dim)

        self.linear1 = nn.Linear(input_dim, int(hidden_dim))
        self.act1 = nn.ReLU6()
        self.film1 = FiLMLayer(film_cond_dim, int(hidden_dim))

        self.linear2 = nn.Linear(int(hidden_dim), int(embed_dim))
        self.act2 = nn.ReLU6()

    def forward(
        self,
        codebook: torch.Tensor,
        weights: torch.Tensor,
        occupancy_embedding: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            codebook:            [K, code_dim] OTF-VQ-VAE codebook embeddings.
            weights:             [B, K] per-code usage weights.
            occupancy_embedding: [B, K, occupancy_embed_dim] from the occupancy encoder.
            global_state:        [B, state_feature_dim] mean-pooled state features.
        Returns:
            [B, K, embed_dim] per-factor embeddings.
        """
        batch_size, num_codes = weights.shape[0], weights.shape[1]
        codebook = codebook.to(device=weights.device, dtype=weights.dtype)
        code_embeddings = codebook.unsqueeze(0).expand(batch_size, -1, -1)
        descriptor = torch.cat(
            [code_embeddings, weights.unsqueeze(-1), occupancy_embedding],
            dim=-1,
        )
        # FiLM condition: [global_state, occupancy_embedding] concatenated per factor
        gs_expanded = global_state.unsqueeze(1).expand(-1, num_codes, -1)
        film_cond = torch.cat([gs_expanded, occupancy_embedding], dim=-1)

        x = self.film1(self.act1(self.linear1(descriptor)), film_cond)
        return self.act2(self.linear2(x))


class OTFStyleStateEncoder(nn.Module):
    """
    Convolutional state encoder that encodes the current frame into spatial
    feature maps at the OTF-VQ-VAE patch grid resolution.

    Each EncoderBlock's output is FiLM-conditioned on a compact projection of
    the OTF-VQ-VAE codebook assignment weights, providing top-down guidance from
    the detected motion factors.
    """

    def __init__(
        self,
        input_channels: int,
        image_height: int,
        image_width: int,
        output_channels: int,
        target_size: Tuple[int, int],
        num_codes: int,
        occ_cond_dim: int,
        encoder_channels: Optional[Tuple[int, ...]] = None,
        num_res_blocks: int = 1,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = (32, 64, int(output_channels))
        shape = (int(input_channels), int(image_height), int(image_width))

        # Projects OTF-VQ-VAE assignment weights [B, K] → compact condition [B, occ_cond_dim]
        self.occ_projector = nn.Linear(int(num_codes), int(occ_cond_dim))

        encoder_blocks: list = []
        film_layers: list = []
        for out_channels in encoder_channels:
            block = EncoderBlock(
                shape,
                int(out_channels),
                num_res_blocks=int(num_res_blocks),
                downscale=True,
            )
            encoder_blocks.append(block)
            film_layers.append(FiLMLayer2D(int(occ_cond_dim), block.get_output_shape()[0]))
            shape = block.get_output_shape()

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.film_layers = nn.ModuleList(film_layers)

        self.channel_projection = (
            nn.Identity()
            if shape[0] == int(output_channels)
            else nn.Conv2d(shape[0], int(output_channels), kernel_size=1)
        )
        self.target_size = (int(target_size[0]), int(target_size[1]))

    def forward(self, current_frame: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_frame: [B, C, H, W] current observation.
            weights:       [B, num_codes] OTF-VQ-VAE codebook assignment probabilities.
        Returns:
            [B, output_channels, target_H, target_W] state feature map.
        """
        occ_cond = self.occ_projector(weights.to(current_frame.dtype))  # [B, occ_cond_dim]
        x = current_frame
        for block, film in zip(self.encoder_blocks, self.film_layers):
            x = film(block(x), occ_cond)
        features = self.channel_projection(x)
        if tuple(features.shape[-2:]) != self.target_size:
            features = F.interpolate(
                features,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
        return features


def pool_state_by_occupancy(
    state_features: torch.Tensor,
    occupancy: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if tuple(occupancy.shape[-2:]) != tuple(state_features.shape[-2:]):
        occupancy = F.interpolate(
            occupancy.float(),
            size=state_features.shape[-2:],
            mode="nearest",
        )
    occupancy = occupancy.to(dtype=state_features.dtype)
    numerator = torch.einsum("bkhw,bchw->bkc", occupancy, state_features)
    denominator = occupancy.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(float(eps))
    return numerator / denominator


class GateNetwork(nn.Module):
    """
    Computes a per-factor gating scalar alpha ∈ [0, 1] from the factor embedding,
    local pooled state, and global state.

    Architecture: 4 linear layers (2 more than the original 2-layer version).
    Every hidden layer is FiLM-conditioned on [global_state ∥ occupancy_embedding]
    per factor.  Uses ReLU6 activations.
    """

    def __init__(
        self,
        factor_embed_dim: int,
        state_feature_dim: int,
        occupancy_embed_dim: int,
        hidden_dim: int,
        activation: str = "gelu",  # kept for API compatibility; ReLU6 is used internally
    ) -> None:
        super().__init__()
        input_dim = int(factor_embed_dim) + 2 * int(state_feature_dim)
        # FiLM condition: global_state ∥ per-factor occupancy_embedding
        film_cond_dim = int(state_feature_dim) + int(occupancy_embed_dim)

        self.linear1 = nn.Linear(input_dim, int(hidden_dim))
        self.act1 = nn.ReLU6()
        self.film1 = FiLMLayer(film_cond_dim, int(hidden_dim))

        self.linear2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.act2 = nn.ReLU6()
        self.film2 = FiLMLayer(film_cond_dim, int(hidden_dim))

        self.linear3 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.act3 = nn.ReLU6()
        self.film3 = FiLMLayer(film_cond_dim, int(hidden_dim))

        self.out_linear = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        factor_embedding: torch.Tensor,
        local_state: torch.Tensor,
        global_state: torch.Tensor,
        occupancy_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            factor_embedding:    [B, K, factor_embed_dim]
            local_state:         [B, K, state_feature_dim] occupancy-pooled state.
            global_state:        [B, state_feature_dim] mean-pooled state.
            occupancy_embedding: [B, K, occupancy_embed_dim]
        Returns:
            [B, K, 1] gate values in (0, 1).
        """
        num_factors = factor_embedding.shape[1]
        expanded_global = global_state.unsqueeze(1).expand(-1, num_factors, -1)
        gate_input = torch.cat([factor_embedding, local_state, expanded_global], dim=-1)

        # FiLM condition per factor: global_state ∥ occupancy_embedding
        film_cond = torch.cat([expanded_global, occupancy_embedding], dim=-1)

        x = self.film1(self.act1(self.linear1(gate_input)), film_cond)
        x = self.film2(self.act2(self.linear2(x)), film_cond)
        x = self.film3(self.act3(self.linear3(x)), film_cond)
        return torch.sigmoid(self.out_linear(x))


class OTFStyleForwardDecoder(nn.Module):
    """
    OTF-style forward decoder (FDM).

    z_action is projected to a spatial grid and concatenated with state features
    to seed the decoder.  Every subsequent DecoderBlock is additionally
    FiLM-conditioned on z_action, with two modes controlled by ``film_mode``:

      * ``"z_action"``           — channel-wise FiLM from z_action alone
                                   (FiLMLayer2D, lighter weight).
      * ``"z_action_and_state"`` — spatially-varying FiLM from z_action and the
                                   full state feature map (SpatialFiLMLayer).

    Uses ReLU6 activations throughout.
    """

    def __init__(
        self,
        state_feature_dim: int,
        z_action_dim: int,
        output_channels: int,
        grid_size: Tuple[int, int],
        image_size: Tuple[int, int],
        hidden_dim: int,
        z_grid_channels: int,
        prediction_mode: str = "residual",
        activation: str = "gelu",  # kept for API compatibility; ReLU6 is used internally
        num_res_blocks: int = 1,
        num_upsample_blocks: Optional[int] = None,
        film_mode: str = "z_action_and_state", # or z_action_and_state
    ) -> None:
        super().__init__()
        if prediction_mode not in {"residual", "direct"}:
            raise ValueError(f"prediction_mode must be residual or direct, got {prediction_mode}")
        if film_mode not in {"z_action", "z_action_and_state"}:
            raise ValueError(
                f"film_mode must be 'z_action' or 'z_action_and_state', got {film_mode}"
            )
        self.prediction_mode = prediction_mode
        self.film_mode = film_mode
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.output_channels = int(output_channels)
        self.z_grid_channels = int(z_grid_channels)
        self.state_feature_dim = int(state_feature_dim)

        self.z_projector = nn.Sequential(
            nn.Linear(int(z_action_dim), int(hidden_dim)),
            nn.ReLU6(),
            nn.Linear(
                int(hidden_dim),
                self.z_grid_channels * self.grid_size[0] * self.grid_size[1],
            ),
        )
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                int(state_feature_dim) + self.z_grid_channels,
                int(hidden_dim),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU6(),
        )

        if num_upsample_blocks is None:
            scale_h = max(1.0, self.image_size[0] / max(1, self.grid_size[0]))
            scale_w = max(1.0, self.image_size[1] / max(1, self.grid_size[1]))
            num_upsample_blocks = int(math.ceil(math.log2(max(scale_h, scale_w))))
        num_upsample_blocks = max(0, int(num_upsample_blocks))

        shape = (int(hidden_dim), self.grid_size[0], self.grid_size[1])
        decoder_blocks: list = []
        film_layers: list = []
        for _ in range(num_upsample_blocks):
            block = DecoderBlock(shape, int(hidden_dim), num_res_blocks=int(num_res_blocks))
            decoder_blocks.append(block)
            out_channels = block.get_output_shape()[0]
            if film_mode == "z_action":
                film_layers.append(FiLMLayer2D(int(z_action_dim), out_channels))
            else:  # "z_action_and_state"
                film_layers.append(
                    SpatialFiLMLayer(int(z_action_dim), int(state_feature_dim), out_channels)
                )
            shape = block.get_output_shape()

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.film_layers = nn.ModuleList(film_layers)

        self.output_projection = nn.Sequential(
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Conv2d(int(hidden_dim), self.output_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        z_action: torch.Tensor,
        current_frame: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state_features.shape[0]
        z_grid = self.z_projector(z_action).reshape(
            batch_size,
            self.z_grid_channels,
            self.grid_size[0],
            self.grid_size[1],
        )
        decoder_input = torch.cat([state_features, z_grid.to(state_features.dtype)], dim=1)
        x = self.input_projection(decoder_input)

        for block, film in zip(self.decoder_blocks, self.film_layers):
            x = block(x)
            if self.film_mode == "z_action":
                x = film(x, z_action)
            else:  # "z_action_and_state"
                x = film(x, z_action, state_features)

        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(
                x,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        decoded = self.output_projection(x)
        if self.prediction_mode == "residual":
            return current_frame + decoded
        return decoded


class OTFLAM(nn.Module):
    def __init__(self, otf_vqvae_extractor: FrozenOTFVQVAEFactorExtractor, cfg: Mapping | DictConfig) -> None:
        super().__init__()
        self.otf_vqvae_extractor = otf_vqvae_extractor
        vq_cfg = otf_vqvae_extractor.otf_vqvae.cfg
        grid_height, grid_width = otf_vqvae_extractor.grid_size

        self.image_height = int(_cfg_get(cfg, "image_height", vq_cfg.image_height))
        self.image_width = int(_cfg_get(cfg, "image_width", vq_cfg.image_width))
        self.channels = int(_cfg_get(cfg, "channels", vq_cfg.channels))
        self.prediction_mode = str(_cfg_get(cfg, "prediction_mode", "residual"))
        self.mask_inactive_factors = bool(_cfg_get(cfg, "mask_inactive_factors", True))
        self.eps = float(_cfg_get(cfg, "eps", 1.0e-8))
        activation = str(_cfg_get(cfg, "activation", "gelu"))
        
        # New flag: "gate" (default) or "mean"
        self.aggregator_type = str(_cfg_get(cfg, "aggregator_type", "gate")).lower()

        # ------------------------------------------------------------------
        # Occupancy encoder
        # ------------------------------------------------------------------
        occupancy_embed_dim = int(_cfg_get(cfg, "occupancy_embed_dim", 16))
        occupancy_encoder_type = str(_cfg_get(cfg, "occupancy_encoder_type", "small_cnn"))
        state_feature_dim = int(_cfg_get(cfg, "state_feature_dim", 128))

        if occupancy_encoder_type == "small_cnn":
            self.occupancy_encoder = SmallCNNOccupancyEncoder(
                occupancy_embed_dim,
                state_feature_dim,
                activation,
            )
        elif occupancy_encoder_type == "mlp":
            self.occupancy_encoder = MLPOccupancyEncoder(
                grid_height,
                grid_width,
                occupancy_embed_dim,
                int(_cfg_get(cfg, "occupancy_hidden_dim", 128)),
                state_feature_dim,
                activation,
            )
        else:
            raise ValueError(f"Unknown occupancy_encoder_type: {occupancy_encoder_type}")

        # ------------------------------------------------------------------
        # Factor embedding
        # ------------------------------------------------------------------
        factor_hidden_dim = int(_cfg_get(cfg, "factor_hidden_dim", 128))
        self.factor_embed_dim = int(_cfg_get(cfg, "factor_embed_dim", 128))
        self.factor_embedding = FactorEmbedding(
            code_dim=otf_vqvae_extractor.code_dim,
            occupancy_embed_dim=occupancy_embed_dim,
            state_feature_dim=state_feature_dim,
            hidden_dim=factor_hidden_dim,
            embed_dim=self.factor_embed_dim,
            activation=activation,
        )

        # ------------------------------------------------------------------
        # State encoder
        # occ_cond_dim: projection dimension for the OTF-VQ-VAE weights [B, K]
        # that conditions the state encoder at every block via FiLM.
        # ------------------------------------------------------------------
        occ_cond_dim = int(_cfg_get(cfg, "occ_cond_dim", 32))
        state_channels_cfg = _cfg_get(cfg, "state_encoder_channels", None)
        state_channels = None
        if state_channels_cfg is not None:
            state_channels = tuple(int(value) for value in state_channels_cfg)
        self.state_encoder = OTFStyleStateEncoder(
            input_channels=self.channels,
            image_height=self.image_height,
            image_width=self.image_width,
            output_channels=state_feature_dim,
            target_size=(grid_height, grid_width),
            num_codes=otf_vqvae_extractor.num_codes,
            occ_cond_dim=occ_cond_dim,
            encoder_channels=state_channels,
            num_res_blocks=int(_cfg_get(cfg, "state_encoder_num_res_blocks", 1)),
        )

        # ------------------------------------------------------------------
        # Gate network (Conditionally Initialized)
        # ------------------------------------------------------------------
        if self.aggregator_type == "gate":
            self.gate = GateNetwork(
                factor_embed_dim=self.factor_embed_dim,
                state_feature_dim=state_feature_dim,
                occupancy_embed_dim=occupancy_embed_dim,
                hidden_dim=int(_cfg_get(cfg, "gate_hidden_dim", 128)),
                activation=activation,
            )
        elif self.aggregator_type == "mean":
            self.gate = None
        else:
            raise ValueError(f"Unknown aggregator_type: {self.aggregator_type}. Must be 'gate' or 'mean'.")

        # ------------------------------------------------------------------
        # Action projection + forward decoder
        # decoder_film_mode: "z_action" (default) or "z_action_and_state"
        # ------------------------------------------------------------------
        self.z_action_dim = int(_cfg_get(cfg, "z_action_dim", self.factor_embed_dim))
        self.action_projection = (
            nn.Identity()
            if self.z_action_dim == self.factor_embed_dim
            else nn.Linear(self.factor_embed_dim, self.z_action_dim)
        )
        decoder_film_mode = str(_cfg_get(cfg, "decoder_film_mode", "z_action"))
        self.forward_decoder = OTFStyleForwardDecoder(
            state_feature_dim=state_feature_dim,
            z_action_dim=self.z_action_dim,
            output_channels=self.channels,
            grid_size=(grid_height, grid_width),
            image_size=(self.image_height, self.image_width),
            hidden_dim=int(_cfg_get(cfg, "decoder_hidden_dim", 128)),
            z_grid_channels=int(_cfg_get(cfg, "decoder_z_grid_channels", state_feature_dim)),
            prediction_mode=self.prediction_mode,
            activation=activation,
            num_res_blocks=int(_cfg_get(cfg, "decoder_num_res_blocks", 1)),
            num_upsample_blocks=_cfg_get(cfg, "decoder_upsample_blocks", None),
            film_mode=decoder_film_mode,
        )
        
        # Only apply weight init to the gate if it exists
        modules_to_init = [
            self.occupancy_encoder,
            self.factor_embedding,
            self.state_encoder,
            self.action_projection,
            self.forward_decoder,
        ]
        if self.gate is not None:
            modules_to_init.append(self.gate)
            
        for module in modules_to_init:
            module.apply(weight_init)
            
        self.otf_vqvae_extractor.otf_vqvae.eval()

    def trainable_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("otf_vqvae_extractor."):
                continue
            if parameter.requires_grad:
                yield parameter

    def train(self, mode: bool = True):
        super().train(mode)
        self.otf_vqvae_extractor.otf_vqvae.eval()
        return self

    def second_stage_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("otf_vqvae_extractor.")
        }

    def load_second_stage_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        return self.load_state_dict(state_dict, strict=False)

    def forward(self, batch: Dict[str, torch.Tensor], decode: bool = True) -> Dict[str, torch.Tensor]:
        current = batch["current"]
        target = batch["target"] if "target" in batch else batch.get("future", batch.get("next"))
        if target is None:
            raise KeyError("batch must contain target, future, or next frame")

        # 1. Frozen OTF-VQ-VAE: extract occupancy, weights, codebook (no grad).
        factors = self.otf_vqvae_extractor(batch)
        occupancy = factors["occupancy"]
        weights = factors["weights"]
        active_mask = factors["active_mask"]

        # 2. State encoder runs first so global_state is available to condition
        #    subsequent IDM modules.  It is itself conditioned on OTF-VQ-VAE weights.
        state_features = self.state_encoder(current, weights)
        global_state = state_features.mean(dim=(-2, -1))  # [B, state_feature_dim]

        # 3. Occupancy encoder, conditioned on global_state at every layer.
        occupancy_embedding = self.occupancy_encoder(occupancy, global_state)

        # 4. Factor embedding, conditioned on [global_state ∥ occupancy_embedding].
        factor_embedding = self.factor_embedding(
            factors["codebook"],
            weights,
            occupancy_embedding,
            global_state,
        )

        # 5. Conditionally apply Gate OR Mean Pooling
        if self.aggregator_type == "gate":
            local_state = pool_state_by_occupancy(state_features, occupancy, self.eps)
            alpha_raw = self.gate(factor_embedding, local_state, global_state, occupancy_embedding)
        else: # self.aggregator_type == "mean"
            alpha_raw = torch.ones(
                factor_embedding.shape[0], factor_embedding.shape[1], 1,
                dtype=factor_embedding.dtype, device=factor_embedding.device
            )

        if self.mask_inactive_factors:
            alpha = alpha_raw * active_mask.unsqueeze(-1).to(alpha_raw.dtype)
        else:
            alpha = alpha_raw

        # 6. Weighted aggregation → latent action.
        alpha_sum = alpha.sum(dim=1).clamp_min(self.eps)
        z_factor = (alpha * factor_embedding).sum(dim=1) / alpha_sum
        z_action = self.action_projection(z_factor)

        if not decode:
            return {
            "target": target,
            "z_act": z_action,
            "z_factor": z_factor,
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "weights": weights,
            "occupancy": occupancy,
            "indices": factors["indices"],
            "active_mask": active_mask,
            "state_features": state_features,
            "motion": factors["motion"],
        }

        # 7. Forward decoder, FiLM-conditioned on z_action at every block.
        pred = self.forward_decoder(state_features, z_action, current)

        return {
            "pred": pred,
            "target": target,
            "z_act": z_action,
            "z_factor": z_factor,
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "weights": weights,
            "occupancy": occupancy,
            "indices": factors["indices"],
            "active_mask": active_mask,
            "state_features": state_features,
            "motion": factors["motion"],
        }
