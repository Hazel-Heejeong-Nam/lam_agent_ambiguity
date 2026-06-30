# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of MuJoCo-based Reinforcement Learning environments.

The suite provides a similar API to the original dm_control suite.
Users can configure the distractions on top of the original tasks. The suite is
targeted for loading environments directly with similar configurations as those
used in the original paper. Each distraction wrapper can be used independently
though.
"""

try:
    from dm_control import suite  # pylint: disable=g-import-not-at-top
    from dm_control.suite.wrappers import pixels  # pylint: disable=g-import-not-at-top
except ImportError:
    suite = None

import envs.dcs.background as background
import envs.dcs.camera as camera
import envs.dcs.color as color
import envs.dcs.suite_utils as suite_utils


def _patch_dm_control_mujoco_named_indexer():
    """Allow dm_control/MuJoCo field-list drift in newer conda envs.

    Some dm_control builds ship generated mjbindings metadata for MuJoCo fields
    that are not present in the imported `mujoco` Python package. In that case
    dm_control fails while constructing `physics.named`, even when the missing
    fields are unrelated to the environment. Skipping absent fields preserves the
    named indexers that actually exist on the loaded model/data.
    """
    try:
        from dm_control.mujoco import index as mujoco_index
    except Exception:
        return
    if getattr(mujoco_index, "_lam_missing_field_patch", False):
        return

    def regular_named_axis_convert_key_item(self, key_item):
        mujoco_index._validate_key_item(key_item)

        if isinstance(key_item, str):
            key_item = self._names_to_offsets[mujoco_index.util.to_native_string(key_item)]
        elif isinstance(key_item, (list, mujoco_index.np.ndarray)):
            key_item = mujoco_index.np.asarray(key_item)
            original_shape = key_item.shape
            if isinstance(key_item.flat[0], str):
                key_item = mujoco_index.np.array(
                    [
                        self._names_to_offsets[mujoco_index.util.to_native_string(k)]
                        for k in key_item.flat
                    ]
                )
                key_item.shape = original_shape
        return key_item

    def struct_indexer(struct, struct_name, size_to_axis_indexer):
        struct_name = struct_name.lower()
        if struct_name not in mujoco_index.sizes.array_sizes:
            raise ValueError("Unrecognized struct name " + struct_name)

        array_sizes = mujoco_index.sizes.array_sizes[struct_name]
        field_indexers = {}

        for field_name in array_sizes:
            try:
                attr = getattr(struct, field_name)
            except AttributeError:
                continue
            if not isinstance(attr, mujoco_index.np.ndarray) or attr.dtype.fields:
                continue

            size_names = array_sizes[field_name]
            for new_col_size, field_set in mujoco_index._COLUMN_ID_TO_FIELDS.items():
                if field_name in field_set:
                    size_names = (size_names[0], new_col_size)
                    break

            axis_indexers = [size_to_axis_indexer[size_name] for size_name in size_names]
            field_indexers[field_name] = mujoco_index.FieldIndexer(
                parent_struct=struct,
                field_name=field_name,
                axis_indexers=axis_indexers,
            )

        return mujoco_index.make_struct_indexer(field_indexers)

    mujoco_index.RegularNamedAxis.convert_key_item = regular_named_axis_convert_key_item
    mujoco_index.struct_indexer = struct_indexer
    mujoco_index._lam_missing_field_patch = True


def is_available():
    return suite is not None


def load(
    domain_name,
    task_name,
    difficulty=None,
    dynamic=False,
    background_dataset_path=None,
    background_dataset_videos="train",
    background_kwargs=None,
    camera_kwargs=None,
    color_kwargs=None,
    task_kwargs=None,
    environment_kwargs=None,
    visualize_reward=False,
    render_kwargs=None,
    pixels_only=True,
    pixels_observation_key="pixels",
    env_state_wrappers=None,
):
    """Returns an environment from a domain name, task name and optional settings.

    ```python
    env = suite.load('cartpole', 'balance')
    ```

    Adding a difficulty will configure distractions matching the reference paper
    for easy, medium, hard.

    Users can also toggle dynamic properties for distractions.

    Args:
      domain_name: A string containing the name of a domain.
      task_name: A string containing the name of a task.
      difficulty: Difficulty for the suite. One of 'easy', 'medium', 'hard'.
      dynamic: Boolean controlling whether distractions are dynamic or static.
      background_dataset_path: String to the davis directory that contains the
        video directories.
      background_dataset_videos: String ('train'/'val') or list of strings of the
        DAVIS videos to be used for backgrounds.
      background_kwargs: Dict, overwrites settings for background distractions.
      camera_kwargs: Dict, overwrites settings for camera distractions.
      color_kwargs: Dict, overwrites settings for color distractions.
      task_kwargs: Dict, dm control task kwargs.
      environment_kwargs: Optional `dict` specifying keyword arguments for the
        environment.
      visualize_reward: Optional `bool`. If `True`, object colours in rendered
        frames are set to indicate the reward at each step. Default `False`.
      render_kwargs: Dict, render kwargs for pixel wrapper.
      pixels_only: Boolean controlling the exclusion of states in the observation.
      pixels_observation_key: Key in the observation used for the rendered image.
      env_state_wrappers: Env state wrappers to be called before the PixelWrapper.

    Returns:
      The requested environment.
    """
    if not is_available():
        raise ImportError(
            "dm_control module is not available. Make sure you "
            "follow the installation instructions from the "
            "dm_control package."
        )
    _patch_dm_control_mujoco_named_indexer()

    # if difficulty not in [None, "vanilla", "single_video", "easy", "medium", "hard"]:
    # raise ValueError("Difficulty should be one of: 'vanilla', 'single_video', 'easy', 'medium', 'hard'.")

    render_kwargs = render_kwargs or {}
    if "camera_id" not in render_kwargs:
        render_kwargs["camera_id"] = 2 if domain_name == "quadruped" else 0

    assert suite is not None
    env = suite.load(
        domain_name,
        task_name,
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
    )

    # Apply background distractions.
    if difficulty or background_kwargs:
        background_dataset_path = background_dataset_path or suite_utils.DEFAULT_BACKGROUND_PATH
        final_background_kwargs = dict()
        if difficulty:
            # Get kwargs for the given difficulty.
            num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
            final_background_kwargs.update(
                suite_utils.get_background_kwargs(
                    domain_name, num_videos, dynamic, background_dataset_path, background_dataset_videos
                )
            )
        else:
            # Set the dataset path and the videos.
            final_background_kwargs.update(
                dict(dataset_path=background_dataset_path, dataset_videos=background_dataset_videos)
            )
        if background_kwargs:
            # Overwrite kwargs with those passed here.
            final_background_kwargs.update(background_kwargs)
        env = background.DistractingBackgroundEnv(env, **final_background_kwargs)

    # Apply camera distractions.
    if difficulty or camera_kwargs:
        final_camera_kwargs = dict(camera_id=render_kwargs["camera_id"])
        if difficulty:
            # Get kwargs for the given difficulty.
            scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            final_camera_kwargs.update(suite_utils.get_camera_kwargs(domain_name, scale, dynamic))
        if camera_kwargs:
            # Overwrite kwargs with those passed here.
            final_camera_kwargs.update(camera_kwargs)
        env = camera.DistractingCameraEnv(env, **final_camera_kwargs)

    # Apply color distractions.
    if difficulty or color_kwargs:
        final_color_kwargs = dict()
        if difficulty:
            # Get kwargs for the given difficulty.
            scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            final_color_kwargs.update(suite_utils.get_color_kwargs(scale, dynamic))
        if color_kwargs:
            # Overwrite kwargs with those passed here.
            final_color_kwargs.update(color_kwargs)
        env = color.DistractingColorEnv(env, **final_color_kwargs)

    if env_state_wrappers is not None:
        for wrapper in env_state_wrappers:
            env = wrapper(env)

    # Apply Pixel wrapper after distractions. This is needed to ensure the
    # changes from the distraction wrapper are applied to the MuJoCo environment
    # before the rendering occurs.
    env = pixels.Wrapper(
        env, pixels_only=pixels_only, render_kwargs=render_kwargs, observation_key=pixels_observation_key
    )

    return env
