import h5py
import random

import torch
from torch.utils.data import Dataset


class DCSInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, num_trajs=1000, frame_stack=1, device="cpu", precompute_stacked_obs=False):
        with h5py.File(hdf5_path, "r") as df:
            traj_keys = list(df.keys())
            traj_keys = random.sample(traj_keys, k=num_trajs)

            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in traj_keys]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in traj_keys]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]
        self.stacked_observations = None
        if precompute_stacked_obs:
            self.stacked_observations = [self.__stack_all_obs(obs).contiguous() for obs in self.observations]

    def __stack_all_obs(self, observations):
        idx = torch.arange(observations.shape[0], device=observations.device)
        frames = []
        for offset in range(self.frame_stack):
            frame_idx = (idx - (self.frame_stack - 1 - offset)).clamp_min(0)
            frames.append(observations.index_select(0, frame_idx))
        return torch.cat(frames, dim=-1)

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return len(self.actions) * (self.traj_len - 1)

    def __getitem__(self, idx):
        traj_idx, transition_idx = divmod(idx, self.traj_len - 1)

        if self.stacked_observations is not None:
            obs = self.stacked_observations[traj_idx][transition_idx]
            next_obs = self.stacked_observations[traj_idx][transition_idx + 1]
        else:
            obs = self.__get_padded_obs(traj_idx, transition_idx)
            next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        action = self.actions[traj_idx][transition_idx]

        return obs, next_obs, action
