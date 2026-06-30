import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import h5py
import numpy as np
import torch
from envs.dcs.experts.agent import Agent
from shimmy import DmControlCompatibilityV0
from tqdm.auto import trange

import envs.dcs.suite as suite
import random

def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DCS trajectories with a trained expert.")
    parser.add_argument("--checkpoint_path", default="checkpoints")
    parser.add_argument("--checkpoint_name", default="100.pt")
    parser.add_argument("--dcs_backgrounds_path", default="DAVIS/JPEGImages/480p")
    parser.add_argument("--dcs_backgrounds_split", default="train")
    parser.add_argument("--dcs_difficulty", default="easy")
    parser.add_argument("--dcs_dynamic", type=parse_bool, default=True)
    parser.add_argument("--dcs_img_hw", type=int, default=64)
    parser.add_argument("--greedy_actions", type=parse_bool, default=True)
    parser.add_argument("--num_trajectories", type=int, default=1000)
    parser.add_argument("--save_path", default="data.hdf5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=parse_bool, default=True)
    return parser.parse_args()


class PixelsToInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({k: v for k, v in env.observation_space.items() if k != "pixels"})

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        pixels = obs.pop("pixels")
        info["dcs_pixels"] = pixels
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        pixels = obs.pop("pixels")
        info["dcs_pixels"] = pixels
        return obs, reward, terminated, truncated, info


def make_env(args, config):
    def thunk():
        dm_env = suite.load(
            domain_name=config["domain"],
            task_name=config["task"],
            difficulty=args.dcs_difficulty,
            dynamic=args.dcs_dynamic,
            background_dataset_path=args.dcs_backgrounds_path,
            background_dataset_videos=args.dcs_backgrounds_split,
            pixels_only=False,
            render_kwargs=dict(height=args.dcs_img_hw, width=args.dcs_img_hw),
        )
        env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
        env = PixelsToInfo(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.DtypeObservation(env, np.float32)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    config = torch.load(os.path.join(args.checkpoint_path, "config.pt"))
    checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name), map_location=device)

    init_env = gym.vector.SyncVectorEnv([make_env(args, config) for i in range(1)])
    agent = Agent(init_env, hidden_dim=config["hidden_dim"]).to(device)
    agent.load_state_dict(checkpoint)

    dataset_returns = []
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with h5py.File(args.save_path, "x", track_order=True) as df:
        df.attrs["domain_name"] = config["domain"]
        df.attrs["task_name"] = config["task"]
        df.attrs["difficulty"] = args.dcs_difficulty
        df.attrs["dynamic"] = args.dcs_dynamic
        df.attrs["img_hw"] = args.dcs_img_hw
        df.attrs["split"] = args.dcs_backgrounds_split

        for idx in trange(args.num_trajectories):
            traj_return = 0.0
            pixels = []
            actions = []
            states = []

            env = make_env(args, config)()
            obs, info = env.reset(seed=args.seed + idx)
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.get_action_and_value(
                        torch.tensor(obs[None], device=device), greedy=args.greedy_actions
                    )[0].cpu()
                    action = np.asarray(action.squeeze())

                # recording obs and corresponding action
                pixels.append(info["dcs_pixels"])
                states.append(obs)
                actions.append(action)
                # stepping in the env
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                traj_return += reward

            # writing to the dataset
            group = df.create_group(str(idx))
            group.create_dataset("obs", shape=(len(pixels), *pixels[0].shape), data=np.array(pixels), dtype=np.uint8)
            group.create_dataset(
                "states", shape=(len(states), *states[0].shape), data=np.array(states), dtype=np.float32
            )
            group.create_dataset(
                "actions", shape=(len(actions), *actions[0].shape), data=np.array(actions), dtype=np.float32
            )
            group.attrs["traj_return"] = traj_return
            dataset_returns.append(traj_return)
            print(f"Collected trajectory {idx}.")

        df.attrs["dataset_return"] = np.mean(dataset_returns)

    print("Done! Mean dataset return: ", np.mean(dataset_returns))


if __name__ == "__main__":
    main(parse_args())
