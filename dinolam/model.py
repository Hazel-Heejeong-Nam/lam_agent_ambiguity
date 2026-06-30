from __future__ import annotations

import warnings
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from utils.nn import weight_init
from otf_vqvae.model import make_motion_signal


VQ_FINETUNE_MODES = {"frozen", "encoder", "encoder_codebook"}


def _cfg_get(cfg: Mapping | DictConfig, key: str, default):
    if isinstance(cfg, DictConfig):
        return cfg[key] if key in cfg else default
    return cfg.get(key, default)


class FrozenDINOv2Encoder(nn.Module):
    """Frozen Hugging Face DINOv2 encoder that exposes CLS and patch tokens."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINO-LAM requires transformers. Install requirements in the dino310 environment."
            ) from exc

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        patch_size = getattr(self.encoder.config, "patch_size", 14)
        if isinstance(patch_size, (tuple, list)):
            self.patch_height = int(patch_size[0])
            self.patch_width = int(patch_size[1])
        else:
            self.patch_height = int(patch_size)
            self.patch_width = int(patch_size)
        if self.image_size % self.patch_height != 0 or self.image_size % self.patch_width != 0:
            raise ValueError(
                f"dino_image_size={self.image_size} must be divisible by "
                f"DINO patch size {(self.patch_height, self.patch_width)}"
            )

        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden_size from DINO config for {self.model_name}")
        self.hidden_dim = int(hidden_size)

        mean_tensor = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor, persistent=False)
        self.register_buffer("std", std_tensor, persistent=False)

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.image_size // self.patch_height, self.image_size // self.patch_width

    @property
    def num_patches(self) -> int:
        grid_height, grid_width = self.grid_size
        return grid_height * grid_width

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected BCHW frames, got shape {tuple(frames.shape)}")
        if frames.shape[1] == 1:
            frames = frames.repeat(1, 3, 1, 1)
        elif frames.shape[1] != 3:
            raise ValueError(f"DINOv2 expects 1 or 3 input channels, got {frames.shape[1]}")

        frames = frames.float()
        if tuple(frames.shape[-2:]) != (self.image_size, self.image_size):
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (frames - self.mean.to(frames.dtype)) / self.std.to(frames.dtype)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.encoder.eval()
        pixel_values = self.preprocess(frames)
        outputs = self.encoder(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state.detach()
        patch_tokens = last_hidden_state[:, 1:, :]
        if patch_tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Expected {self.num_patches} DINO patch tokens from grid {self.grid_size}, "
                f"got {patch_tokens.shape[1]}"
            )
        return {
            "cls_token": last_hidden_state[:, 0],
            "patch_tokens": patch_tokens,
            "last_hidden_state": last_hidden_state,
        }

    @torch.no_grad()
    def encode_state(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(frames)
        mean_token = tokens["patch_tokens"].mean(dim=1)
        return torch.cat([tokens["cls_token"], mean_token], dim=-1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encode(frames)["patch_tokens"]


class JEPAVQMotionExtractor(nn.Module):
    """OTF-VQ-VAE motion extractor with optional encoder/codebook fine-tuning."""

    def __init__(
        self,
        otf_vqvae,
        *,
        finetune_mode: str = "frozen",
        use_ema_codebook_update: bool = True,
        use_dead_code_reinit: bool = True,
        dead_code_threshold_steps: int = 1000,
    ) -> None:
        super().__init__()
        if finetune_mode not in VQ_FINETUNE_MODES:
            raise ValueError(f"vq_finetune_mode must be one of {sorted(VQ_FINETUNE_MODES)}, got {finetune_mode}")
        self.otf_vqvae = otf_vqvae
        self.finetune_mode = str(finetune_mode)
        self.use_ema_codebook_update = bool(use_ema_codebook_update)
        self.use_dead_code_reinit = bool(use_dead_code_reinit)
        self.dead_code_threshold_steps = int(dead_code_threshold_steps)
        self._last_patch_embeddings: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None
        self._configure_trainable_parameters()

    @property
    def grid_size(self) -> Tuple[int, int]:
        return int(self.otf_vqvae.encoder.grid_height), int(self.otf_vqvae.encoder.grid_width)

    @property
    def num_codes(self) -> int:
        return int(self.otf_vqvae.num_codes)

    @property
    def code_dim(self) -> int:
        return int(self.otf_vqvae.cfg.latent_dim)

    @property
    def encoder_trainable(self) -> bool:
        return self.finetune_mode in {"encoder", "encoder_codebook"}

    @property
    def codebook_trainable(self) -> bool:
        return self.finetune_mode == "encoder_codebook"

    @property
    def codebook_gradient_trainable(self) -> bool:
        return self.codebook_trainable and not self.use_ema_codebook_update

    def _configure_trainable_parameters(self) -> None:
        self.otf_vqvae.eval()
        for parameter in self.otf_vqvae.parameters():
            parameter.requires_grad_(False)

        if self.encoder_trainable:
            for parameter in self.otf_vqvae.encoder.parameters():
                parameter.requires_grad_(True)

        if self.codebook_trainable:
            if self.use_ema_codebook_update and str(self.otf_vqvae.quantizer.update_mode) != "ema":
                warnings.warn(
                    "use_ema_codebook_update=true was requested, but the loaded VQ quantizer "
                    f"uses update_mode={self.otf_vqvae.quantizer.update_mode}; using optimizer-gradient codebook updates.",
                    stacklevel=2,
                )
                self.use_ema_codebook_update = False
            self.otf_vqvae.quantizer.embedding.requires_grad_(not self.use_ema_codebook_update)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_trainable and mode:
            self.otf_vqvae.encoder.train()
        else:
            self.otf_vqvae.encoder.eval()
        self.otf_vqvae.quantizer.eval()
        self.otf_vqvae.decoder.eval()
        if hasattr(self.otf_vqvae, "reference_encoder"):
            self.otf_vqvae.reference_encoder.eval()
        return self

    def _motion_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        future = batch["future"] if "future" in batch else batch["next"]
        motion_batch = {
            "current": batch["current"],
            "next": future,
        }
        if self.otf_vqvae.motion_input_type == "acceleration":
            if "previous" not in batch:
                raise KeyError("previous frame is required by the OTF-VQ-VAE acceleration motion input")
            motion_batch["previous"] = batch["previous"]
        return motion_batch

    def extract_motion_factors(self, motion_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        patch_embeddings = self.otf_vqvae.encoder(motion_input)
        vq_output = self.otf_vqvae.quantizer(patch_embeddings)

        indices_flat = vq_output["indices"]
        batch_size, num_patches = indices_flat.shape
        grid_height, grid_width = self.grid_size
        if num_patches != grid_height * grid_width:
            raise RuntimeError(f"Expected {grid_height * grid_width} patches, got {num_patches}")

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

        code_tokens = vq_output["quantized_st"] if self.encoder_trainable else vq_output["quantized"].detach()
        if self.codebook_trainable and not self.use_ema_codebook_update:
            codebook = self.otf_vqvae.quantizer.embedding
        else:
            codebook = self.otf_vqvae.quantizer.embedding.detach()

        self._last_patch_embeddings = patch_embeddings.detach()
        self._last_indices = indices_flat.detach()

        return {
            "indices": indices_flat.reshape(batch_size, grid_height, grid_width),
            "code_tokens": code_tokens,
            "occupancy": occupancy,
            "weights": weights,
            "active_mask": active_mask,
            "codebook": codebook,
            "patch_embeddings": patch_embeddings,
            "code_loss": vq_output["code_loss"],
            "commit_loss": vq_output["commit_loss"],
        }

    def _extract_impl(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        motion = make_motion_signal(
            self._motion_batch(batch),
            self.otf_vqvae.motion_input_type,
            self.otf_vqvae.motion_transform,
        )
        factors = self.extract_motion_factors(motion)
        factors["motion"] = motion
        return factors

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.finetune_mode == "frozen":
            with torch.no_grad():
                return self._extract_impl(batch)
        return self._extract_impl(batch)

    def orthogonality_loss(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indices is None:
            indices = self._last_indices
        return self.otf_vqvae.quantizer.orthogonality_loss(indices, self.otf_vqvae.orth_active_only)

    @torch.no_grad()
    def update_codebook(self, step: int) -> Dict[str, int]:
        if not (self.codebook_trainable and self.use_ema_codebook_update):
            return {"reinitialized_codes": 0}
        if self._last_patch_embeddings is None or self._last_indices is None:
            return {"reinitialized_codes": 0}

        original_dead_code_steps = int(self.otf_vqvae.quantizer.dead_code_steps)
        self.otf_vqvae.quantizer.dead_code_steps = (
            int(self.dead_code_threshold_steps) if self.use_dead_code_reinit else 0
        )
        try:
            return self.otf_vqvae.quantizer.update_codebook(
                self._last_patch_embeddings,
                self._last_indices,
                step=int(step),
            )
        finally:
            self.otf_vqvae.quantizer.dead_code_steps = original_dead_code_steps


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if int(dim) % int(heads) != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.norm_attn = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(dim),
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(int(dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(dim), int(mlp_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_dim), int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm_attn(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        return x + self.mlp(self.norm_mlp(x))


class PatchTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim=int(dim), heads=int(heads), mlp_dim=int(mlp_dim), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ActionAggregator(nn.Module):
    """Build compact global latent actions from current DINO tokens and local VQ codes."""

    def __init__(
        self,
        *,
        num_patches: int,
        dino_dim: int,
        code_dim: int,
        cfg: Mapping | DictConfig,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.dino_dim = int(dino_dim)
        self.code_dim = int(code_dim)
        self.aggregator_type = str(_cfg_get(cfg, "action_aggregator_type", "perceiver"))
        if self.aggregator_type not in {"perceiver", "transformer_pool", "mean_pool"}:
            raise ValueError(
                "action_aggregator_type must be perceiver, transformer_pool, or mean_pool, "
                f"got {self.aggregator_type}"
            )

        self.dim = int(_cfg_get(cfg, "aggregator_dim", 256))
        self.z_action_dim = int(_cfg_get(cfg, "z_action_dim", 256))
        self.num_action_queries = int(_cfg_get(cfg, "num_action_queries", 4))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim) * 0.02)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.dino_dim + self.code_dim + self.dim),
            nn.Linear(self.dino_dim + self.code_dim + self.dim, self.dim),
        )

        depth = int(_cfg_get(cfg, "aggregator_depth", 2))
        heads = int(_cfg_get(cfg, "aggregator_heads", 4))
        mlp_dim = int(_cfg_get(cfg, "aggregator_mlp_dim", 1024))
        dropout = float(_cfg_get(cfg, "aggregator_dropout", 0.1))

        if self.aggregator_type == "perceiver":
            if self.num_action_queries < 1:
                raise ValueError("num_action_queries must be >= 1 for perceiver aggregation")
            self.action_queries = nn.Parameter(torch.randn(self.num_action_queries, self.dim) * 0.02)
            self.query_norm = nn.LayerNorm(self.dim)
            self.context_norm = nn.LayerNorm(self.dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.latent_transformer = PatchTransformer(self.dim, depth, heads, mlp_dim, dropout)
            output_dim = self.num_action_queries * self.dim
        elif self.aggregator_type == "transformer_pool":
            self.patch_transformer = PatchTransformer(self.dim, depth, heads, mlp_dim, dropout)
            output_dim = self.dim
        else:
            output_dim = self.dim

        self.output_projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, self.z_action_dim),
        )
        self.apply(weight_init)

    def _input_tokens(self, dino_patch_tokens: torch.Tensor, motion_code_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, _ = dino_patch_tokens.shape
        if num_patches != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} DINO patches, got {num_patches}")
        if motion_code_tokens.shape[:2] != (batch_size, self.num_patches):
            raise ValueError(
                f"Expected motion_code_tokens [B,{self.num_patches},D], got {tuple(motion_code_tokens.shape)}"
            )
        pos = self.pos_embedding.to(dino_patch_tokens.dtype).expand(batch_size, -1, -1)
        tokens = torch.cat([dino_patch_tokens, motion_code_tokens.to(dino_patch_tokens.dtype), pos], dim=-1)
        return self.input_projection(tokens)

    def forward(self, dino_patch_tokens: torch.Tensor, motion_code_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self._input_tokens(dino_patch_tokens, motion_code_tokens)
        attn_weights = None

        if self.aggregator_type == "perceiver":
            batch_size = tokens.shape[0]
            queries = self.action_queries.to(tokens.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            try:
                attended, attn_weights = self.cross_attn(
                    self.query_norm(queries),
                    self.context_norm(tokens),
                    self.context_norm(tokens),
                    need_weights=True,
                    average_attn_weights=False,
                )
            except TypeError:
                attended, attn_weights = self.cross_attn(
                    self.query_norm(queries),
                    self.context_norm(tokens),
                    self.context_norm(tokens),
                    need_weights=True,
                )
            latents = self.latent_transformer(queries + attended)
            pooled = latents.reshape(batch_size, -1)
            z_action = self.output_projection(pooled)
            aggregator_output = latents
            action_query_norm = queries.norm(dim=-1).mean()
        elif self.aggregator_type == "transformer_pool":
            aggregator_output = self.patch_transformer(tokens)
            pooled = aggregator_output.mean(dim=1)
            z_action = self.output_projection(pooled)
            action_query_norm = pooled.new_tensor(0.0)
        else:
            aggregator_output = tokens
            pooled = tokens.mean(dim=1)
            z_action = self.output_projection(pooled)
            action_query_norm = pooled.new_tensor(0.0)

        return {
            "z_act": z_action,
            "aggregator_tokens": aggregator_output,
            "patch_input_tokens": tokens,
            "action_query_attn": attn_weights,
            "action_query_norm": action_query_norm,
            "aggregator_output_norm": aggregator_output.norm(dim=-1).mean(),
        }


class GlobalConditionedPatchPredictor(nn.Module):
    """Non-causal JEPA predictor with optional layer-wise global conditioning."""

    def __init__(
        self,
        *,
        num_patches: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        out_dim: int,
        z_action_dim: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        allow_global_token: bool = True,
        inject_global_each_layer: bool = True,
        global_conditioning_type: str = "additive",
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.allow_global_token = bool(allow_global_token)
        self.inject_global_each_layer = bool(inject_global_each_layer)
        self.global_conditioning_type = str(global_conditioning_type)
        if self.global_conditioning_type not in {"additive", "film"}:
            raise ValueError(
                f"global_conditioning_type must be additive or film, got {self.global_conditioning_type}"
            )

        num_tokens = self.num_patches + (1 if self.allow_global_token else 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, self.dim) * 0.02)
        self.dropout = nn.Dropout(float(emb_dropout))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim=self.dim, heads=int(heads), mlp_dim=int(mlp_dim), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )
        if self.inject_global_each_layer and self.allow_global_token:
            condition_dim = self.dim if self.global_conditioning_type == "additive" else 2 * self.dim
            self.conditioners = nn.ModuleList(
                [nn.Linear(int(z_action_dim), condition_dim) for _ in range(int(depth))]
            )
        else:
            self.conditioners = nn.ModuleList()
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_dim),
        )
        self.apply(weight_init)

    def _condition(
        self,
        x: torch.Tensor,
        z_action: Optional[torch.Tensor],
        layer_idx: int,
        has_global: bool,
    ) -> torch.Tensor:
        if not (self.inject_global_each_layer and has_global and z_action is not None):
            return x
        patch_slice = x[:, 1:, :]
        condition = self.conditioners[layer_idx](z_action).to(x.dtype)
        if self.global_conditioning_type == "additive":
            patch_slice = patch_slice + condition.unsqueeze(1)
        else:
            gamma, beta = condition.chunk(2, dim=-1)
            patch_slice = patch_slice * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return torch.cat([x[:, :1, :], patch_slice], dim=1)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        global_token: Optional[torch.Tensor] = None,
        z_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_patches, _ = patch_tokens.shape
        if num_patches != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {num_patches}")

        has_global = global_token is not None
        if has_global:
            if not self.allow_global_token:
                raise ValueError("global_token was provided but allow_global_token=False")
            if global_token.shape[:2] != (batch_size, 1):
                raise ValueError(f"Expected global_token [B,1,D], got {tuple(global_token.shape)}")
            x = torch.cat([global_token, patch_tokens], dim=1)
        else:
            x = patch_tokens

        x = x + self.pos_embedding[:, : x.shape[1]].to(x.dtype)
        x = self.dropout(x)
        for layer_idx, layer in enumerate(self.layers):
            x = self._condition(x, z_action, layer_idx, has_global)
            x = layer(x)
        if has_global:
            x = x[:, 1:, :]
        return self.prediction_head(x)


def jepa_loss(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, loss_type: str) -> torch.Tensor:
    target_tokens = target_tokens.detach()
    if loss_type == "mse":
        return F.mse_loss(pred_tokens, target_tokens)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred_tokens, target_tokens)
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(pred_tokens, target_tokens, dim=-1)).mean()
    raise ValueError(f"Unknown jepa_loss_type: {loss_type}")


def jepa_token_metrics(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> Dict[str, float]:
    pred_tokens = pred_tokens.detach()
    target_tokens = target_tokens.detach()
    cosine_similarity = F.cosine_similarity(pred_tokens, target_tokens, dim=-1).mean()
    feature_mse = F.mse_loss(pred_tokens, target_tokens)
    return {
        "cosine_similarity": float(cosine_similarity.item()),
        "feature_mse": float(feature_mse.item()),
    }


class DINOJEPALatentActionModel(nn.Module):
    """DINO-JEPA latent action labeler with a compact global action bottleneck."""

    def __init__(
        self,
        otf_vqvae_extractor: Optional[JEPAVQMotionExtractor],
        dino_encoder: FrozenDINOv2Encoder,
        cfg: Mapping | DictConfig,
    ) -> None:
        super().__init__()
        self.otf_vqvae_extractor = otf_vqvae_extractor
        self.dino_encoder = dino_encoder
        self.dino_encoder.eval()
        self.freeze_dino = bool(_cfg_get(cfg, "freeze_dino", True))
        if not self.freeze_dino:
            raise ValueError("DINO fine-tuning is not supported; set freeze_dino=true")
        for parameter in self.dino_encoder.parameters():
            parameter.requires_grad_(False)

        self.use_motion_codes = bool(_cfg_get(cfg, "use_motion_codes", True))
        self.use_global_action_token = bool(_cfg_get(cfg, "use_global_action_token", True))
        patch_motion_default = _cfg_get(cfg, "use_patch_motion_codes_in_predictor", None)
        if patch_motion_default is None:
            patch_motion_default = _cfg_get(cfg, "use_patch_motion_codes", False)
        self.use_patch_motion_codes_in_predictor = self.use_motion_codes and bool(patch_motion_default)
        if self.use_global_action_token and not self.use_motion_codes:
            raise ValueError("Global action token requires motion codes")
        if self.use_motion_codes and self.otf_vqvae_extractor is None:
            raise ValueError("use_motion_codes=true requires a OTF-VQ-VAE motion extractor")
        if self.otf_vqvae_extractor is not None:
            self.otf_vqvae_extractor.otf_vqvae.eval()

        self.dino_grid_size = dino_encoder.grid_size
        self.vq_grid_size = None if otf_vqvae_extractor is None else otf_vqvae_extractor.grid_size
        self.patch_grid_match = None if self.vq_grid_size is None else self.dino_grid_size == self.vq_grid_size
        self.grid_alignment = str(_cfg_get(cfg, "grid_alignment", "exact"))
        if self.grid_alignment not in {"exact", "nearest_resize"}:
            raise ValueError(f"grid_alignment must be exact or nearest_resize, got {self.grid_alignment}")
        if self.use_motion_codes and self.patch_grid_match is False and self.grid_alignment == "exact":
            raise ValueError(
                "DINO patch grid and OTF-VQ-VAE patch grid do not match. "
                f"DINO={self.dino_grid_size}, OTF-VQ-VAE={self.vq_grid_size}. "
                "Set grid_alignment=nearest_resize or choose a matching dino_image_size."
            )
        if self.use_motion_codes and self.patch_grid_match is False:
            warnings.warn(
                "DINO patch grid and OTF-VQ-VAE patch grid do not match; resizing VQ motion tokens "
                f"from {self.vq_grid_size} to {self.dino_grid_size} with nearest neighbor.",
                stacklevel=2,
            )

        self.target_mode = str(_cfg_get(cfg, "target_mode", "future"))
        if self.target_mode not in {"future", "delta"}:
            raise ValueError(f"target_mode must be future or delta, got {self.target_mode}")
        self.jepa_loss_type = str(_cfg_get(cfg, "jepa_loss_type", "mse"))
        if self.jepa_loss_type not in {"cosine", "mse", "smooth_l1"}:
            raise ValueError(f"jepa_loss_type must be cosine, mse, or smooth_l1, got {self.jepa_loss_type}")
        self.vq_finetune_mode = str(_cfg_get(cfg, "vq_finetune_mode", "frozen"))
        if self.vq_finetune_mode not in VQ_FINETUNE_MODES:
            raise ValueError(f"vq_finetune_mode must be one of {sorted(VQ_FINETUNE_MODES)}, got {self.vq_finetune_mode}")
        if self.vq_finetune_mode != "frozen" and not self.use_motion_codes:
            raise ValueError("VQ fine-tuning requires use_motion_codes=true")
        self.use_codebook_orthogonality_loss = bool(_cfg_get(cfg, "use_codebook_orthogonality_loss", True))
        self.codebook_orthogonality_weight = float(_cfg_get(cfg, "codebook_orthogonality_weight", 1.0e-4))

        predictor_dim = int(_cfg_get(cfg, "predictor_dim", 384))
        patch_input_dim = int(dino_encoder.hidden_dim)
        if self.use_patch_motion_codes_in_predictor:
            patch_input_dim += int(otf_vqvae_extractor.code_dim)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, predictor_dim),
        )

        self.action_aggregator = None
        self.global_projection = None
        z_action_dim = int(_cfg_get(cfg, "z_action_dim", 256))
        if self.use_global_action_token:
            self.action_aggregator = ActionAggregator(
                num_patches=dino_encoder.num_patches,
                dino_dim=dino_encoder.hidden_dim,
                code_dim=otf_vqvae_extractor.code_dim,
                cfg=cfg,
            )
            z_action_dim = int(self.action_aggregator.z_action_dim)
            self.global_projection = nn.Linear(z_action_dim, predictor_dim)

        self.predictor = GlobalConditionedPatchPredictor(
            num_patches=dino_encoder.num_patches,
            dim=predictor_dim,
            depth=int(_cfg_get(cfg, "predictor_depth", 2)),
            heads=int(_cfg_get(cfg, "predictor_heads", 6)),
            mlp_dim=int(_cfg_get(cfg, "predictor_mlp_dim", 1536)),
            out_dim=int(dino_encoder.hidden_dim),
            z_action_dim=z_action_dim,
            dropout=float(_cfg_get(cfg, "predictor_dropout", 0.1)),
            emb_dropout=float(_cfg_get(cfg, "predictor_emb_dropout", 0.1)),
            allow_global_token=self.use_global_action_token,
            inject_global_each_layer=bool(_cfg_get(cfg, "inject_global_each_layer", True)),
            global_conditioning_type=str(_cfg_get(cfg, "global_conditioning_type", "additive")),
        )

        for module in (self.input_projection, self.global_projection):
            if module is not None:
                module.apply(weight_init)

    @property
    def z_action_dim(self) -> Optional[int]:
        return None if self.action_aggregator is None else int(self.action_aggregator.z_action_dim)

    @property
    def state_dim(self) -> int:
        return 2 * int(self.dino_encoder.hidden_dim)

    def trainable_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("dino_encoder."):
                continue
            if parameter.requires_grad:
                yield parameter

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino_encoder.eval()
        if self.otf_vqvae_extractor is not None:
            self.otf_vqvae_extractor.train(mode)
        return self

    def second_stage_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for key, value in self.state_dict().items():
            if key.startswith("dino_encoder."):
                continue
            if key.startswith("otf_vqvae_extractor."):
                if self.vq_finetune_mode == "frozen":
                    continue
                if ".otf_vqvae.encoder." in key:
                    state[key] = value
                elif self.vq_finetune_mode == "encoder_codebook" and ".otf_vqvae.quantizer." in key:
                    state[key] = value
                continue
            state[key] = value
        return state

    def load_second_stage_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        return self.load_state_dict(state_dict, strict=False)

    def _align_motion_tokens(
        self,
        code_tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.vq_grid_size == self.dino_grid_size:
            return code_tokens, indices
        if self.grid_alignment == "exact":
            raise RuntimeError(
                "DINO patch grid and OTF-VQ-VAE patch grid do not match. "
                f"DINO={self.dino_grid_size}, OTF-VQ-VAE={self.vq_grid_size}."
            )
        if self.vq_grid_size is None:
            raise RuntimeError("Cannot align motion tokens without a VQ grid")

        batch_size, _, code_dim = code_tokens.shape
        vq_height, vq_width = self.vq_grid_size
        dino_height, dino_width = self.dino_grid_size
        token_grid = code_tokens.reshape(batch_size, vq_height, vq_width, code_dim).permute(0, 3, 1, 2)
        token_grid = F.interpolate(token_grid, size=(dino_height, dino_width), mode="nearest")
        aligned_tokens = token_grid.permute(0, 2, 3, 1).reshape(batch_size, dino_height * dino_width, code_dim)
        aligned_indices = F.interpolate(
            indices.unsqueeze(1).float(),
            size=(dino_height, dino_width),
            mode="nearest",
        ).squeeze(1).to(indices.dtype)
        return aligned_tokens, aligned_indices

    def _target_tokens(self, current_tokens: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "future":
            return future_tokens.detach()
        return (future_tokens - current_tokens).detach()

    @torch.no_grad()
    def encode_policy_state(self, current: torch.Tensor) -> torch.Tensor:
        return self.dino_encoder.encode_state(current)

    def action_labels(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        if not self.use_global_action_token or self.action_aggregator is None:
            raise ValueError("DINO-LAM action labels require use_global_action_token=true")
        current_tokens = self.dino_encoder(batch["current"])
        factors = self.otf_vqvae_extractor(batch)
        motion_code_tokens, motion_indices = self._align_motion_tokens(factors["code_tokens"], factors["indices"])
        aggregator_stats = self.action_aggregator(current_tokens, motion_code_tokens)
        return {
            "z_act": aggregator_stats["z_act"],
            "motion_indices": motion_indices,
            "factors": factors,
            "current_tokens": current_tokens,
            **aggregator_stats,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        current = batch["current"]
        future = batch["target"] if "target" in batch else batch.get("future", batch.get("next"))
        if future is None:
            raise KeyError("batch must contain target, future, or next frame")

        current_encoded = self.dino_encoder.encode(current)
        current_tokens = current_encoded["patch_tokens"]
        future_tokens = self.dino_encoder(future)

        needs_motion_factors = self.use_patch_motion_codes_in_predictor or self.use_global_action_token
        factors = self.otf_vqvae_extractor(batch) if needs_motion_factors else None
        motion_code_tokens = None
        motion_indices = None
        if factors is not None:
            motion_code_tokens, motion_indices = self._align_motion_tokens(
                factors["code_tokens"],
                factors["indices"],
            )

        patch_inputs = [current_tokens]
        if self.use_patch_motion_codes_in_predictor:
            if motion_code_tokens is None:
                raise RuntimeError("Patch motion codes requested but VQ factors were not extracted")
            patch_inputs.append(motion_code_tokens.to(current_tokens.dtype))
        projected_patches = self.input_projection(torch.cat(patch_inputs, dim=-1))

        aggregator_stats = None
        global_token = None
        z_act = None
        if self.use_global_action_token:
            if motion_code_tokens is None or self.action_aggregator is None or self.global_projection is None:
                raise RuntimeError("Global action token requested but VQ factors were not available")
            aggregator_stats = self.action_aggregator(current_tokens, motion_code_tokens)
            z_act = aggregator_stats["z_act"]
            global_token = self.global_projection(z_act).unsqueeze(1)

        pred_tokens = self.predictor(projected_patches, global_token=global_token, z_action=z_act)
        target_tokens = self._target_tokens(current_tokens, future_tokens)
        loss_jepa = jepa_loss(pred_tokens, target_tokens, self.jepa_loss_type)
        codebook_orthogonality_loss = None
        loss = loss_jepa
        if self.vq_finetune_mode == "encoder_codebook" and self.otf_vqvae_extractor is not None:
            codebook_orthogonality_loss = self.otf_vqvae_extractor.orthogonality_loss()
        if (
            codebook_orthogonality_loss is not None
            and self.use_codebook_orthogonality_loss
            and self.codebook_orthogonality_weight > 0.0
            and self.otf_vqvae_extractor.codebook_gradient_trainable
        ):
            loss = loss + self.codebook_orthogonality_weight * codebook_orthogonality_loss

        pred_future_tokens = current_tokens + pred_tokens if self.target_mode == "delta" else pred_tokens
        patch_error = 1.0 - F.cosine_similarity(pred_future_tokens.detach(), future_tokens.detach(), dim=-1)
        output = {
            "pred_tokens": pred_tokens,
            "target_tokens": target_tokens,
            "pred_future_tokens": pred_future_tokens,
            "future_tokens": future_tokens,
            "current_tokens": current_tokens,
            "loss": loss,
            "jepa_loss": loss_jepa,
            "codebook_orthogonality_loss": codebook_orthogonality_loss,
            "motion_indices": motion_indices,
            "motion_code_tokens": motion_code_tokens,
            "z_act": z_act,
            "patch_error": patch_error,
            "factors": factors,
            "dino_grid": self.dino_grid_size,
            "vq_grid": self.vq_grid_size,
            "patch_grid_match": self.patch_grid_match,
        }
        if aggregator_stats is not None:
            output.update(aggregator_stats)
        return output

    @torch.no_grad()
    def update_codebook(self, step: int) -> Dict[str, int]:
        if self.otf_vqvae_extractor is None:
            return {"reinitialized_codes": 0}
        return self.otf_vqvae_extractor.update_codebook(step)
