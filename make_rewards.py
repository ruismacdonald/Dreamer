import os
import random
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.distributions as distributions

from collections import OrderedDict

import env_wrapper
from replay_buffer import ReplayBuffer
from models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder
from state_distance import SimpleContrastiveStateDistanceModel
from utils import *

os.environ["MUJOCO_GL"] = "egl"


def make_env(args, loca_phase="phase_1", loca_mode="train"):
    seed = args.seed
    if loca_mode == "eval":
        seed = None
    env = env_wrapper.DMCLoCA(
        args.env, seed, loca_phase=loca_phase, loca_mode=loca_mode
    )
    # env = env_wrapper.ActionRepeat(env, args.action_repeat)
    # env = env_wrapper.NormalizeActions(env)
    # env = env_wrapper.TimeLimit(env, args.time_limit / args.action_repeat)
    return env


def preprocess_obs(obs):
    obs = obs.to(torch.float32) / 255.0 - 0.5
    return obs


class Dreamer:
    def __init__(
        self,
        args,
        obs_shape,
        action_size,
        device,
        restore=False,
        loca_state_distance=False,
        state_distance_model=None,
    ):

        self.args = args
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device
        self.restore = args.restore
        self.restore_path = args.checkpoint_path

        self.loca_state_distance = loca_state_distance
        self.state_distance_model = None
        if self.loca_state_distance:
            self.data_buffer = ReplayBuffer(
                self.args.buffer_size,
                self.obs_shape,
                self.action_size,
                self.args.train_seq_len,
                self.args.batch_size,
                distance_process=True,
                obs_repr_rad=args.loca_replay_rad,
                obs_repr_count=args.loca_replay_count,
            )
            self.state_distance_model = state_distance_model
        else:
            self.data_buffer = ReplayBuffer(
                self.args.buffer_size,
                self.obs_shape,
                self.action_size,
                self.args.train_seq_len,
                self.args.batch_size,
            )

        self._build_model(restore=self.restore)

    def _build_model(self, restore):

        self.rssm = RSSM(
            action_size=self.action_size,
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            hidden_size=self.args.deter_size,
            obs_embed_size=self.args.obs_embed_size,
            activation=self.args.dense_activation_function,
        ).to(self.device)

        self.actor = ActionDecoder(
            action_size=self.action_size,
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            units=self.args.num_units,
            n_layers=4,
            activation=self.args.dense_activation_function,
        ).to(self.device)
        self.obs_encoder = ConvEncoder(
            input_shape=self.obs_shape,
            embed_size=self.args.obs_embed_size,
            activation=self.args.cnn_activation_function,
        ).to(self.device)
        self.obs_decoder = ConvDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=self.obs_shape,
            activation=self.args.cnn_activation_function,
        ).to(self.device)
        self.reward_model = DenseDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=(1,),
            n_layers=2,
            units=self.args.num_units,
            activation=self.args.dense_activation_function,
            dist="normal",
        ).to(self.device)
        self.value_model = DenseDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=(1,),
            n_layers=3,
            units=self.args.num_units,
            activation=self.args.dense_activation_function,
            dist="normal",
        ).to(self.device)
        if self.args.use_disc_model:
            self.discount_model = DenseDecoder(
                stoch_size=self.args.stoch_size,
                deter_size=self.args.deter_size,
                output_shape=(1,),
                n_layers=2,
                units=self.args.num_units,
                activation=self.args.dense_activation_function,
                dist="binary",
            ).to(self.device)

        if self.args.use_disc_model:
            self.world_model_params = (
                list(self.rssm.parameters())
                + list(self.obs_encoder.parameters())
                + list(self.obs_decoder.parameters())
                + list(self.reward_model.parameters())
                + list(self.discount_model.parameters())
            )
        else:
            self.world_model_params = (
                list(self.rssm.parameters())
                + list(self.obs_encoder.parameters())
                + list(self.obs_decoder.parameters())
                + list(self.reward_model.parameters())
            )

        self.world_model_opt = optim.Adam(
            self.world_model_params, self.args.model_learning_rate
        )
        self.value_opt = optim.Adam(
            self.value_model.parameters(), self.args.value_learning_rate
        )
        self.actor_opt = optim.Adam(
            self.actor.parameters(), self.args.actor_learning_rate
        )

        if self.args.use_disc_model:
            self.world_model_modules = [
                self.rssm,
                self.obs_encoder,
                self.obs_decoder,
                self.reward_model,
                self.discount_model,
            ]
        else:
            self.world_model_modules = [
                self.rssm,
                self.obs_encoder,
                self.obs_decoder,
                self.reward_model,
            ]
        self.value_modules = [self.value_model]
        self.actor_modules = [self.actor]

        if restore:
            self.restore_checkpoint(self.restore_path)

    def world_model_loss(self, obs, acs, rews, nonterms, reward_mask):

        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:])
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(
            obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len - 1
        )
        features = torch.cat([self.posterior["stoch"], self.posterior["deter"]], dim=-1)
        rew_dist = self.reward_model(features)
        obs_dist = self.obs_decoder(features)
        if self.args.use_disc_model:
            disc_dist = self.discount_model(features)

        prior_dist = self.rssm.get_dist(prior["mean"], prior["std"])
        post_dist = self.rssm.get_dist(self.posterior["mean"], self.posterior["std"])

        if self.args.algo == "Dreamerv2":
            post_no_grad = self.rssm.detach_state(self.posterior)
            prior_no_grad = self.rssm.detach_state(prior)
            post_mean_no_grad, post_std_no_grad = (
                post_no_grad["mean"],
                post_no_grad["std"],
            )
            prior_mean_no_grad, prior_std_no_grad = (
                prior_no_grad["mean"],
                prior_no_grad["std"],
            )

            kl_loss = self.args.kl_alpha * (
                torch.mean(
                    distributions.kl.kl_divergence(
                        self.rssm.get_dist(post_mean_no_grad, post_std_no_grad),
                        prior_dist,
                    )
                )
            )
            kl_loss += (1 - self.args.kl_alpha) * (
                torch.mean(
                    distributions.kl.kl_divergence(
                        post_dist,
                        self.rssm.get_dist(prior_mean_no_grad, prior_std_no_grad),
                    )
                )
            )
        else:
            kl_loss = torch.mean(distributions.kl.kl_divergence(post_dist, prior_dist))
            kl_loss = torch.max(
                kl_loss, kl_loss.new_full(kl_loss.size(), self.args.free_nats)
            )

        obs_loss = -torch.mean(obs_dist.log_prob(obs[1:]))
        rew_loss = rew_dist.log_prob(rews[:-1]) * reward_mask[:-1].squeeze(-1)
        rew_loss = -rew_loss.sum() / reward_mask[:-1].sum()

        if self.args.use_disc_model:
            disc_loss = -torch.mean(disc_dist.log_prob(nonterms[:-1]))

        if self.args.use_disc_model:
            model_loss = (
                self.args.kl_loss_coeff * kl_loss
                + obs_loss
                + rew_loss
                + self.args.disc_loss_coeff * disc_loss
            )
        else:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss

        return model_loss

    def actor_loss(self):

        with torch.no_grad():
            posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states = self.rssm.imagine_rollout(
                self.actor, posterior, self.args.imagine_horizon
            )

        self.imag_feat = torch.cat([imag_states["stoch"], imag_states["deter"]], dim=-1)

        with FreezeParameters(self.world_model_modules + self.value_modules):
            imag_rew_dist = self.reward_model(self.imag_feat)
            imag_val_dist = self.value_model(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts = self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(
            imag_rews[:-1],
            imag_vals[:-1],
            discounts[:-1],
            self.args.td_lambda,
            imag_vals[-1],
        )

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        return actor_loss

    def value_loss(self):

        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            discount = self.discounts.detach()
            value_targ = self.returns.detach()

        value_dist = self.value_model(value_feat)
        value_loss = -torch.mean(
            self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1)
        )

        return value_loss

    def train_one_batch(self):

        obs, acs, rews, terms, reward_mask = self.data_buffer.sample()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs = torch.tensor(acs, dtype=torch.float32).to(self.device)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = (
            torch.tensor((1.0 - terms), dtype=torch.float32)
            .to(self.device)
            .unsqueeze(-1)
        )
        reward_mask = (
            torch.tensor(reward_mask, dtype=torch.float32).to(self.device).unsqueeze(-1)
        )

        model_loss = self.world_model_loss(obs, acs, rews, nonterms, reward_mask)
        self.world_model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_clip_norm)
        self.world_model_opt.step()

        actor_loss = self.actor_loss()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
        self.actor_opt.step()

        value_loss = self.value_loss()
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.value_model.parameters(), self.args.grad_clip_norm
        )
        self.value_opt.step()

        return model_loss.item(), actor_loss.item(), value_loss.item()

    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):

        obs = obs["image"]
        obs = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
        action = self.actor(features, deter=not explore)
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        imag_rew_dist = self.reward_model(features)
        imag_rews = imag_rew_dist.mean

        return posterior, action, imag_rews

    def act_and_collect_data(self, env, collect_steps):

        obs = env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):

            with torch.no_grad():
                posterior, action = self.act_with_world_model(
                    obs, prev_state, prev_action, explore=True
                )
            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = env.step(action)
            representation = None
            if self.loca_state_distance:
                representation = self.state_distance_model.get_representation(
                    preprocess_obs(torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0))
                )
            self.data_buffer.add(obs, action, rew, done, representation)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                if i != collect_steps - 1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs
                prev_state = posterior
                prev_action = (
                    torch.tensor(action, dtype=torch.float32)
                    .to(self.device)
                    .unsqueeze(0)
                )

        return np.array(episode_rewards)

    def evaluate(self, env, eval_episodes, render=False):

        episode_rew = np.zeros((eval_episodes))

        video_images = [[] for _ in range(eval_episodes)]

        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.rssm.init_state(1, self.device)
            prev_action = torch.zeros(1, self.action_size).to(self.device)

            while not done:
                with torch.no_grad():
                    posterior, action, rw__ = self.act_with_world_model(
                        obs, prev_state, prev_action
                    )
                print(rw__.item(), " -----")
                action = action[0].cpu().numpy()
                next_obs, rew, done, _ = env.step(action)
                prev_state = posterior
                prev_action = (
                    torch.tensor(action, dtype=torch.float32)
                    .to(self.device)
                    .unsqueeze(0)
                )

                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs["image"].transpose(1, 2, 0).copy())
                obs = next_obs
        return episode_rew, np.array(video_images[: self.args.max_videos_to_save])

    def collect_random_episodes(self, env, seed_steps):

        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(action)

            representation = None
            if self.loca_state_distance:
                representation = self.state_distance_model.get_representation(
                    preprocess_obs(torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0))
                )
            self.data_buffer.add(obs, action, rew, done, representation)
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
                if i != seed_steps - 1:
                    seed_episode_rews.append(0.0)
                done = False
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save(self, save_path):

        torch.save(
            {
                "rssm": self.rssm.state_dict(),
                "actor": self.actor.state_dict(),
                "reward_model": self.reward_model.state_dict(),
                "obs_encoder": self.obs_encoder.state_dict(),
                "obs_decoder": self.obs_decoder.state_dict(),
                "value_model": self.value_model.state_dict(),
                "discount_model": self.discount_model.state_dict()
                if self.args.use_disc_model
                else None,
                "actor_optimizer": self.actor_opt.state_dict(),
                "value_optimizer": self.value_opt.state_dict(),
                "world_model_optimizer": self.world_model_opt.state_dict(),
            },
            save_path,
        )

    def restore_checkpoint(self, ckpt_path):

        checkpoint = torch.load(ckpt_path)
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.obs_decoder.load_state_dict(checkpoint["obs_decoder"])
        self.value_model.load_state_dict(checkpoint["value_model"])
        if self.args.use_disc_model and (checkpoint["discount_model"] is not None):
            self.discount_model.load_state_dict(checkpoint["discount_model"])

        self.world_model_opt.load_state_dict(checkpoint["world_model_optimizer"])
        self.actor_opt.load_state_dict(checkpoint["actor_optimizer"])
        self.value_opt.load_state_dict(checkpoint["value_optimizer"])

    def _get_reward_estimate_from_model(self, obs, prev_state, prev_action):
        obs = obs["image"]
        obs = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
        imag_rew_dist = self.reward_model(features)
        imag_rews = imag_rew_dist.mean
        return posterior, prev_action, imag_rews 

    def get_reward_est(self, pos, env):
        obs_ = env.reset()

        def inverse_kinematics(pos):
            x, y = pos[0], pos[1]
            a1, a2 = env._actuators_length[0], env._actuators_length[1]
            d = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)
            d = np.clip(d, -1.0, 1.0)
            if np.random.rand() < 0.5:
                theta2 = np.arccos(d)
                theta1 = np.arctan(y / x) - np.arctan(
                    (a2 * np.sin(theta2)) / (a1 + a2 * np.cos(theta2))
                )
            else:
                theta2 = -np.arccos(d)
                theta1 = np.arctan(y / x) - np.arctan(
                    (a2 * np.sin(theta2)) / (a1 + a2 * np.cos(theta2))
                )
            return theta1, theta2

        def forward_kinematics(theta1, theta2):
            a1, a2 = env._actuators_length[0], env._actuators_length[1]
            x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
            y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
            return np.array([x, y])

        theta1, theta2 = inverse_kinematics(pos)
        tmp_finger_pos = forward_kinematics(theta1, theta2)
        if np.linalg.norm(pos - tmp_finger_pos) > 1e-5:
            theta1 += np.pi

        with env._env._physics.reset_context():
            env._env._physics.named.data.qpos["shoulder"] = theta1
            env._env._physics.named.data.qpos["wrist"] = theta2
        obs = dict()
        obs["image"] = env.render().transpose(2, 0, 1).copy()

        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device) + .5
        reward = 0.

        for _ in range(1):
            with torch.no_grad():
                posterior, action, reward = self._get_reward_estimate_from_model(
                    obs, prev_state, prev_action
                )
            action = action[0].cpu().numpy()
            prev_state = posterior

        return reward


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env", type=str, default="walker_walk", help="Control Suite environment"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="Dreamerv1",
        choices=["Dreamerv1", "Dreamerv2"],
        help="choosing algorithm",
    )
    parser.add_argument(
        "--exp-name", type=str, default="lr1e-3", help="name of experiment for logging"
    )
    parser.add_argument("--train", action="store_true", help="trains the model")
    parser.add_argument("--evaluate", action="store_true", help="tests the model")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--no-gpu", action="store_true", help="GPUs aren't used if passed true"
    )
    # Data parameters
    parser.add_argument(
        "--max-episode-length", type=int, default=1000, help="Max episode length"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=1000000, help="Experience replay size"
    )  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument(
        "--time-limit", type=int, default=1000, help="time limit"
    )  # Environment TimeLimit
    # Models parameters
    parser.add_argument(
        "--cnn-activation-function",
        type=str,
        default="relu",
        help="Model activation function for a convolution layer",
    )
    parser.add_argument(
        "--dense-activation-function",
        type=str,
        default="elu",
        help="Model activation function a dense layer",
    )
    parser.add_argument(
        "--obs-embed-size", type=int, default=1024, help="Observation embedding size"
    )  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument(
        "--num-units",
        type=int,
        default=400,
        help="num hidden units for reward/value/discount models",
    )
    parser.add_argument(
        "--deter-size",
        type=int,
        default=200,
        help="GRU hidden size and deterministic belief size",
    )
    parser.add_argument(
        "--stoch-size", type=int, default=30, help="Stochastic State/latent size"
    )
    # Actor Exploration Parameters
    parser.add_argument("--action-repeat", type=int, default=2, help="Action repeat")
    parser.add_argument("--action-noise", type=float, default=0.3, help="Action noise")
    # Training parameters
    parser.add_argument("--seed-steps", type=int, default=5000, help="seed episodes")
    parser.add_argument(
        "--update-steps",
        type=int,
        default=100,
        help="num of train update steps per iter",
    )
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=1000,
        help="actor collect steps per 1 train iter",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="batch size")
    parser.add_argument(
        "--train-seq-len",
        type=int,
        default=50,
        help="sequence length for training world model",
    )
    parser.add_argument(
        "--imagine-horizon", type=int, default=15, help="Latent imagination horizon"
    )
    parser.add_argument(
        "--use-disc-model", action="store_true", help="whether to use discount model"
    )
    # Coeffecients and constants
    parser.add_argument("--free-nats", type=float, default=3, help="free nats")
    parser.add_argument(
        "--discount", type=float, default=0.99, help="discount factor for actor critic"
    )
    parser.add_argument(
        "--td-lambda", type=float, default=0.95, help="discount rate to compute return"
    )
    parser.add_argument(
        "--kl-loss-coeff",
        type=float,
        default=1.0,
        help="weightage for kl_loss of model",
    )
    parser.add_argument(
        "--kl-alpha",
        type=float,
        default=0.8,
        help="kl balancing weight; used for Dreamerv2",
    )
    parser.add_argument(
        "--disc-loss-coeff",
        type=float,
        default=10.0,
        help="weightage of discount model",
    )

    # Optimizer Parameters
    parser.add_argument(
        "--model_learning-rate",
        type=float,
        default=6e-4,
        help="World Model Learning rate",
    )
    parser.add_argument(
        "--actor_learning-rate", type=float, default=8e-5, help="Actor Learning rate"
    )
    parser.add_argument(
        "--value_learning-rate",
        type=float,
        default=8e-5,
        help="Value Model Learning rate",
    )
    parser.add_argument(
        "--adam-epsilon", type=float, default=1e-7, help="Adam optimizer epsilon value"
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=100.0, help="Gradient clipping norm"
    )
    # Eval parameters
    parser.add_argument("--test", action="store_true", help="Test only")
    parser.add_argument(
        "--test-interval", type=int, default=10000, help="Test interval (episodes)"
    )
    parser.add_argument(
        "--test-episodes", type=int, default=10, help="Number of test episodes"
    )
    # saving and checkpoint parameters
    parser.add_argument(
        "--scalar-freq", type=int, default=1e3, help="scalar logging freq"
    )
    parser.add_argument(
        "--log-video-freq", type=int, default=-1, help="video logging frequency"
    )
    parser.add_argument(
        "--max-videos-to-save", type=int, default=2, help="max_videos for saving"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Checkpoint interval (episodes)",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="", help="Load model checkpoint"
    )
    parser.add_argument(
        "--restore", action="store_true", help="restores model from checkpoint"
    )
    parser.add_argument(
        "--experience-replay", type=str, default="", help="Load experience replay"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")

    parser.add_argument("--loca-all-phases", action="store_true", help="")
    parser.add_argument("--loca-phase", type=str, default="phase_1", help="")
    parser.add_argument("--loca-phase1-steps", type=int, default=0, help="")
    parser.add_argument("--loca-phase2-steps", type=int, default=0, help="")
    parser.add_argument("--loca-phase3-steps", type=int, default=0, help="")

    parser.add_argument("--loca-sate-distance", action="store_true", help="")
    parser.add_argument("--loca-replay-rad", type=float, default=0.05)
    parser.add_argument("--loca-replay-count", type=int, default=2, help="")
    
    parser.add_argument("--checkpoint-model-path", type=str)

    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # if torch.cuda.is_available() and not args.no_gpu:
    #     device = torch.device("cuda")
    #     torch.cuda.manual_seed(args.seed)
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    
    loca_phase = "phase_1"
    train_env = make_env(args, loca_phase, "train")
    test_env = make_env(args, loca_phase, "eval")
    obs_shape = train_env.observation_space["image"].shape
    action_size = train_env.action_space.shape[0]
    dreamer = Dreamer(args, obs_shape, action_size, device, args.restore)
    dreamer.restore_checkpoint(args.checkpoint_model_path)
    # dreamer.evaluate(test_env, 1)
    # return

    cnt_grid = 150
    bb, aa = np.meshgrid(
        np.linspace(-0.2, 0.2, cnt_grid), np.linspace(-0.2, 0.2, cnt_grid)
    )
    pos_pairs = np.vstack((bb.flatten(), aa.flatten())).T
    data = []
    for pos in pos_pairs:
        reward_ = dreamer.get_reward_est(np.array(pos), test_env).item() / args.action_repeat
        data.append([pos[0], pos[1], reward_])
    with open("heat_data.npy", "wb") as f:
        np.save(f, np.array(data))


if __name__ == "__main__":
    main()

