# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--control_mode", type=str, default="ACCBR", help="Control mode for the agent.")
parser.add_argument("--save_plots", action="store_true", default=False, help="Save plots during training.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run in real-time (default: True). Pass --no-real-time for unthrottled execution.",
)
parser.add_argument("--headless", action="store_true", default=False, help="Force headless execution.")
parser.add_argument("--max_timesteps", type=int, default=None, help="Maximum timesteps to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# In Isaac Lab 3.0, unless headless or livestream is requested, default to the Kit visualizer
# to open the GUI viewport window for real-time visualization.
if not args_cli.headless and args_cli.visualizer is None and getattr(args_cli, "livestream", 0) == 0:
    args_cli.visualizer = ["kit"]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

from MARL_mav_carry_ext.plotting_tools import DirectMARLPlotter

# register the gym environment

# check for minimum supported skrl version
SKRL_VERSION = "2.1.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # configure video recording on environment config
    # configure camera visualizer on environment config
    from isaaclab_visualizers.kit import KitVisualizerCfg

    if not hasattr(env_cfg.sim, "visualizer_cfgs") or env_cfg.sim.visualizer_cfgs is None:
        env_cfg.sim.visualizer_cfgs = []
    env_cfg.sim.visualizer_cfgs.append(
        KitVisualizerCfg(
            headless=args_cli.headless,
            eye=(2.4, -2.4, 1.5),
            lookat=(0.0, 0.0, 1.0),
        )
    )

    # configure video recording on environment config
    if args_cli.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        video_dir = os.path.join(log_dir, "videos", "play")
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=video_dir,
                video_length=args_cli.video_length,
                video_interval=0,
            )
        ]
        print(f"[INFO] Video recording enabled: saving {args_cli.video_length} steps to {video_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env, state_as_observation=True)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    agents = runner.agent.values() if isinstance(runner.agent, dict) else [runner.agent]
    for ag in agents:
        if hasattr(ag, "enable_training_mode"):
            ag.enable_training_mode(False, apply_to_models=True)
        elif hasattr(ag, "set_running_mode"):
            ag.set_running_mode("eval")
    plotter = DirectMARLPlotter(env, control_mode=args_cli.control_mode)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        start_time = time.time()
        with torch.inference_mode():
            # agent stepping
            if hasattr(env, "possible_agents"):
                states = {a: env.state() for a in env.possible_agents} if hasattr(env, "state") else None
                actions, outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                # deterministic actions
                actions = {a: outputs.get(a, {}).get("mean_actions", actions[a]) for a in env.possible_agents}
            else:
                actions, outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
                # deterministic actions
                actions = outputs.get("mean_actions", actions)
            # env stepping
            if env.num_envs == 1:
                plotter.collect_data()
            obs, _, _, _, _ = env.step(actions)

        timestep += 1
        if args_cli.max_timesteps and timestep >= args_cli.max_timesteps:
            break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    if env.num_envs == 1:
        # save plots without showing
        plot_path = os.path.join(log_dir, "plots", "play")
        plotter.plot(save=True, save_dir=plot_path)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
