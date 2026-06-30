import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class RoboMimicDataset(Dataset):
    # Loads the .npz file
    # Download the ones with -img from robomimic
    # returns sliding windows of consecutive frames from within same trajectory as batch["obs"] of shape (T, C, H, W)

    def __init__(
        self,
        npz_path: str,
        window_size: int = 3,
        img_size: int = 64,
        k_min: int = 2,
        k_max: int = 2,
    ):
        data = np.load(npz_path, allow_pickle=True)

        self.images = data["images"] # (N, 3, 96, 96) uint8
        if self.images.shape[1] > 3:
            self.images = self.images[:, :3]
        self.actions = data["actions"] # (N, 7)
        traj_lengths = data["traj_lengths"] # (num_trajs,)

        self.window_size = window_size
        self.img_size = img_size
        self.k_min = k_min
        self.k_max = k_max

        assert self.window_size >= 3, "window_size must be >= 3 for triplets"
        assert self.k_min >= 2, "k_min must be >= 2"
        assert self.k_max >= self.k_min, "k_max must be >= k_min"

        self.valid_starts = []
        self.sample_meta = []
        idx = 0
        for length in traj_lengths:
            # Local-uniform k: for each start, sample uniformly among valid k in [k_min, k_max].
            for local_t in range(length):
                max_valid_k = min(self.k_max, length - 1 - local_t)
                if max_valid_k < self.k_min:
                    continue
                t = idx + local_t
                self.valid_starts.append(t)
                self.sample_meta.append((idx, length, local_t))
            idx += length

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        traj_start, traj_len, local_t = self.sample_meta[idx]
        max_valid_k = min(self.k_max, traj_len - 1 - local_t)
        k = np.random.randint(self.k_min, max_valid_k + 1)

        indices = [start, start + 1, start + k]
        frames = torch.from_numpy(self.images[indices]).float() / 255.0 - 0.5

        # always resize to ensure consistent resolution
        frames = F.interpolate(frames, size=self.img_size, mode="bilinear", align_corners=False)

        actions = torch.from_numpy(self.actions[indices]).float()
        return {
            "obs": frames,
            "obs_triplet": frames,
            "actions": actions,
            "k": torch.tensor(k, dtype=torch.long),
        }


def make_robomimic_loader(
    npz_path: str,
    batch_size: int = 32,
    window_size: int = 3,
    img_size: int = 64,
    num_workers: int = 4,
    k_min: int = 2,
    k_max: int = 2,
) -> DataLoader:
    dataset = RoboMimicDataset(
        npz_path,
        window_size=window_size,
        img_size=img_size,
        k_min=k_min,
        k_max=k_max,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)