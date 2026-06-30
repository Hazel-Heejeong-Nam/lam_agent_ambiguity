import gymnasium as gym
import h5py
import numpy as np
from pathlib import Path
from shimmy import DmControlCompatibilityV0

from envs.dcs import background, suite


class SelectPixelsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class FlattenStackedFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = old_shape[1:-1] + (old_shape[0] * old_shape[-1],)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        obs = obs.transpose((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs


def _resolve_background_videos(backgrounds_path, backgrounds_split):
    if backgrounds_split in ("train", "training"):
        return background.DAVIS17_TRAINING_VIDEOS
    if backgrounds_split in ("val", "validation"):
        return background.DAVIS17_VALIDATION_VIDEOS
    if backgrounds_split is None:
        return sorted(path.name for path in Path(backgrounds_path).expanduser().iterdir() if path.is_dir())
    if isinstance(backgrounds_split, str):
        return [backgrounds_split]
    return list(backgrounds_split)


def _validate_backgrounds(backgrounds_path, backgrounds_split):
    if not backgrounds_path:
        return

    root = Path(backgrounds_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"DCS background path does not exist: {root}. "
            "Pass --bc_dcs_backgrounds_path/--act_decoder_dcs_backgrounds_path to a DAVIS JPEGImages/480p directory."
        )

    videos = _resolve_background_videos(root, backgrounds_split)
    missing_dirs = []
    empty_dirs = []
    for video in videos:
        video_dir = root / video
        if not video_dir.is_dir():
            missing_dirs.append(str(video_dir))
        elif not any(video_dir.glob("*.jpg")):
            empty_dirs.append(str(video_dir))

    if missing_dirs or empty_dirs:
        details = []
        if missing_dirs:
            details.append(f"missing video dirs: {missing_dirs[:5]}")
        if empty_dirs:
            details.append(f"dirs with no .jpg frames: {empty_dirs[:5]}")
        raise FileNotFoundError(
            f"No usable DCS background frames found for split={backgrounds_split!r} under {root}; "
            + "; ".join(details)
        )


def create_env_from_df(
    hdf5_path,
    backgrounds_path,
    backgrounds_split,
    frame_stack=1,
    pixels_only=True,
    flatten_frames=True,
    difficulty=None,
    seed=None,
):
    with h5py.File(hdf5_path, "r") as df:
        env_difficulty = df.attrs["difficulty"] if difficulty is None else difficulty
        if env_difficulty:
            _validate_backgrounds(backgrounds_path, backgrounds_split)
        distraction_seed_kwargs = dict(seed=seed) if seed is not None and env_difficulty else None
        dm_env = suite.load(
            domain_name=df.attrs["domain_name"],
            task_name=df.attrs["task_name"],
            difficulty=env_difficulty,
            dynamic=df.attrs["dynamic"],
            background_dataset_path=backgrounds_path,
            background_dataset_videos=backgrounds_split,
            background_kwargs=distraction_seed_kwargs,
            camera_kwargs=distraction_seed_kwargs,
            color_kwargs=distraction_seed_kwargs,
            pixels_only=pixels_only,
            render_kwargs=dict(height=df.attrs["img_hw"], width=df.attrs["img_hw"]),
        )
        env = DmControlCompatibilityV0(dm_env)
        env = gym.wrappers.ClipAction(env)

        if pixels_only:
            env = SelectPixelsObsWrapper(env)

        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            if flatten_frames:
                env = FlattenStackedFrames(env)

    return env
