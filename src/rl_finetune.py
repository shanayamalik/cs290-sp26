"""
Phase 7: PPO fine-tuning from a behavioral-cloning warm start.

The environment wrapper uses the same 27-dim observation representation as BC:
flattened 5x5 highway-env obs + d_min + step, normalized with the saved BC
stats. The PPO actor is initialized from the BC policy weights when possible.

Quick smoke run:
    python3 src/rl_finetune.py --timesteps 2000 --eval-episodes 10

Longer run:
    python3 src/rl_finetune.py --timesteps 100000 --eval-episodes 50
"""

import argparse
import os
import random
import sys
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.insert(0, str(Path(__file__).parent))

from eval_policy import TRAFFIC_MIXES
from generate_data import _patch_merge_env_continuous_rewards
from mpc_expert import ACC_SCALE
from policy_network import PolicyNetwork

ENV_CONFIG = {
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "action": {"type": "ContinuousAction"},
    "speed_limit": 25,
}

MAX_STEPS = 50
DEFAULT_BC_MODEL = Path("models/bc_policy_default_mix.pt")


class MergePPOWrapper(gym.Wrapper):
    """Normalize BC-style observations, shape reward, and prevent reversing."""

    def __init__(self, env, obs_mean: np.ndarray, obs_std: np.ndarray,
                 traffic_mix: str = "default_mix", seed: int = 0):
        super().__init__(env)
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-8)
        self.driver_fns = TRAFFIC_MIXES[traffic_mix]
        self.rng = random.Random(seed)
        self.max_steps = MAX_STEPS
        self.step_count = 0
        self.prev_x = 0.0
        self.total_clamp_count = 0
        self.total_action_count = 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for vehicle in self.env.unwrapped.road.vehicles[1:]:
            self.rng.choice(self.driver_fns)(vehicle)
        self.step_count = 0
        self.prev_x = float(self.env.unwrapped.road.vehicles[0].position[0])
        return self._augment_obs(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action, -1.0, 1.0)
        self.total_action_count += 1

        ego = self.env.unwrapped.road.vehicles[0]
        ego_speed = float(ego.speed)
        min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE
        speed_clamped = False
        if action[0] < min_acc_norm:
            action[0] = float(np.clip(min_acc_norm, -1.0, 1.0))
            self.total_clamp_count += 1
            speed_clamped = True

        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        ego = self.env.unwrapped.road.vehicles[0]
        x = float(ego.position[0])
        speed = float(ego.speed)
        dx = x - self.prev_x
        self.prev_x = x

        crashed = bool(self.env.unwrapped.vehicle.crashed)
        reward = self._rl_reward(float(env_reward), speed, dx, crashed, terminated, speed_clamped)
        truncated = truncated or self.step_count >= self.max_steps
        info = dict(info)
        info.update({
            "ego_speed": speed,
            "speed_clamped": self.total_clamp_count,
            "action_count": self.total_action_count,
            "crashed": crashed,
        })
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _rl_reward(self, env_reward: float, speed: float, dx: float,
                   crashed: bool, terminated: bool, speed_clamped: bool) -> float:
        if crashed:
            return -100.0

        # Encourage forward commitment while keeping the signal small enough
        # that PPO still respects highway-env's built-in merge reward.
        reward = env_reward
        reward += 0.04 * min(max(speed, 0.0), 20.0)
        reward += 0.03 * max(dx, 0.0)
        reward -= 0.05
        if speed > 20.0:
            reward -= 0.15 * (speed - 20.0)
        if speed < 0.5:
            reward -= 0.5
        if speed_clamped:
            reward -= 1.0
        if terminated:
            reward += 20.0
        return float(reward)

    def _augment_obs(self, obs) -> np.ndarray:
        ego = self.env.unwrapped.road.vehicles[0]
        others = self.env.unwrapped.road.vehicles[1:]
        d_min = float(min(
            np.linalg.norm(ego.position - v.position) for v in others
        )) if others else 100.0
        obs_aug = np.append(obs.reshape(-1), [d_min, float(self.step_count), float(ego.speed)]).astype(np.float32)
        obs_norm = (obs_aug - self.obs_mean) / self.obs_std
        return np.clip(obs_norm, -10.0, 10.0).astype(np.float32)


class TanhMeanActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy whose Gaussian action mean is tanh-bounded.

    This makes the deterministic PPO actor match the BC policy convention:
    hidden MLP -> Linear -> tanh -> action in [-1, 1].
    """

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        mean_actions = torch.tanh(self.action_net(latent_pi))
        if self.use_sde:
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)


def make_env(obs_mean: np.ndarray, obs_std: np.ndarray, traffic_mix: str, seed: int):
    _patch_merge_env_continuous_rewards()
    env = gym.make("merge-v0", config=ENV_CONFIG)
    return MergePPOWrapper(env, obs_mean, obs_std, traffic_mix=traffic_mix, seed=seed)


def load_bc_stats(bc_model: Path) -> tuple[np.ndarray, np.ndarray]:
    stats_path = bc_model.with_suffix(".npz")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing BC normalization stats: {stats_path}")
    stats = np.load(stats_path)
    mean = stats["mean"]
    std = stats["std"]
    # BC was trained on 27-dim obs. PPO adds ego_speed as 28th feature.
    # Pad with mean=0, std=1 so ego_speed passes through unnormalized.
    if len(mean) == 27:
        mean = np.append(mean, 0.0)
        std = np.append(std, 1.0)
    return mean, std


def warm_start_actor(ppo: PPO, bc_model_path: Path) -> bool:
    if not bc_model_path.exists():
        print(f"BC model not found at {bc_model_path}; PPO will start randomly.")
        return False

    bc = PolicyNetwork()
    bc.load_state_dict(torch.load(bc_model_path, weights_only=True))
    bc.eval()

    # BC net: Linear/ReLU/Linear/ReLU/Linear/ReLU/Linear/Tanh
    bc_linears = [bc.net[0], bc.net[2], bc.net[4], bc.net[6]]
    policy_net = ppo.policy.mlp_extractor.policy_net
    ppo_linears = [policy_net[0], policy_net[2], policy_net[4], ppo.policy.action_net]

    with torch.no_grad():
        # First layer: BC expects 27 inputs, PPO now expects 28.
        # Copy BC weights for the first 27 columns; zero the 28th (ego_speed)
        # so the network starts behaving identically to BC.
        src = bc_linears[0]
        dst = ppo_linears[0]
        dst.weight[:, :27].copy_(src.weight)
        dst.weight[:, 27:].zero_()
        dst.bias.copy_(src.bias)
        for src, dst in zip(bc_linears[1:], ppo_linears[1:]):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
        ppo.policy.log_std.fill_(-1.0)

    print(f"Warm-started PPO actor from {bc_model_path}")
    return True


def evaluate(model: PPO, obs_mean: np.ndarray, obs_std: np.ndarray,
             traffic_mix: str, episodes: int, seed: int) -> dict:
    env = make_env(obs_mean, obs_std, traffic_mix, seed)
    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        speeds = []
        steps = 0
        first_crash_step = None
        clamp_before = env.total_clamp_count
        action_before = env.total_action_count
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            speeds.append(float(info.get("ego_speed", 0.0)))
            steps += 1
            if info.get("crashed", False) and first_crash_step is None:
                first_crash_step = steps
        results.append({
            "crashed": first_crash_step is not None,
            "steps": steps,
            "reward": total_reward,
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "clamps": env.total_clamp_count - clamp_before,
            "actions": env.total_action_count - action_before,
            "first_crash_step": first_crash_step,
        })
    env.close()
    n = len(results)
    return {
        "episodes": n,
        "crash_rate": sum(r["crashed"] for r in results) / n,
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_speed": float(np.mean([r["avg_speed"] for r in results])),
        "clamp_rate": (
            sum(r["clamps"] for r in results)
            / max(sum(r["actions"] for r in results), 1)
        ),
        "spawn_crashes": sum(
            1 for r in results
            if r["first_crash_step"] is not None and r["first_crash_step"] <= 3
        ),
    }


def print_eval(label: str, summary: dict) -> None:
    print(f"\n{label}")
    print(f"  episodes    : {summary['episodes']}")
    print(f"  crash rate  : {100*summary['crash_rate']:.1f}%")
    print(f"  mean reward : {summary['mean_reward']:.2f}")
    print(f"  mean steps  : {summary['mean_steps']:.1f}")
    print(f"  mean speed  : {summary['mean_speed']:.2f} m/s")
    print(f"  clamp rate  : {100*summary['clamp_rate']:.1f}%")
    print(f"  spawn crashes: {summary['spawn_crashes']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-model", type=str, default=str(DEFAULT_BC_MODEL))
    parser.add_argument("--out", type=str, default="models/ppo_finetuned_merge")
    parser.add_argument("--traffic-mix", choices=list(TRAFFIC_MIXES.keys()), default="default_mix")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    bc_model = Path(args.bc_model)
    obs_mean, obs_std = load_bc_stats(bc_model)
    env = make_env(obs_mean, obs_std, args.traffic_mix, args.seed)

    policy_kwargs = {
        "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        "activation_fn": torch.nn.ReLU,
    }
    model = PPO(
        TanhMeanActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=64,
        gamma=0.98,
        seed=args.seed,
        verbose=1,
    )
    warm_start_actor(model, bc_model)

    before = evaluate(model, obs_mean, obs_std, args.traffic_mix, args.eval_episodes, args.seed)
    print_eval("Before PPO fine-tuning", before)

    print(f"\nTraining PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\nSaved PPO model to {out_path}.zip")

    after = evaluate(model, obs_mean, obs_std, args.traffic_mix, args.eval_episodes, args.seed)
    print_eval("After PPO fine-tuning", after)
    env.close()


if __name__ == "__main__":
    main()
