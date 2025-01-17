import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import geoopt
from agent.ddpg import DDPGAgent
from agent.hyper_utils import PoincarePlaneDistance, ClipNorm, apply_sn_until_instance, final_weight_init_hyp_small, PoincareDist
from radam import RiemannianAdam

class RND(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 clip_val=5.):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        max_euclidean_norm = 60
        dimensions_per_space = 32
        hyperbolic_layer_kwargs = dict(rescale_euclidean_norms_gain=1.0, rescale_normal_params=True, effective_softmax_rescale=0.5)

        self.predictor = [encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                    #    nn.Linear(hidden_dim, rnd_rep_dim),
                                    ClipNorm(max_norm=max_euclidean_norm,
                          dimensions_per_space=dimensions_per_space),
            PoincarePlaneDistance(in_features=hidden_dim, 
            num_planes=rnd_rep_dim,
            dimensions_per_space=dimensions_per_space,
            **hyperbolic_layer_kwargs)
                                       ]
        self.target = [copy.deepcopy(encoder),
                                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    # nn.Linear(hidden_dim, rnd_rep_dim),
                                    ClipNorm(max_norm=max_euclidean_norm,
                                    dimensions_per_space=dimensions_per_space),
                                    PoincarePlaneDistance(in_features=hidden_dim, 
                                    num_planes=rnd_rep_dim,
                                    dimensions_per_space=dimensions_per_space,
                                    **hyperbolic_layer_kwargs),
                                    ]
        self.apply(utils.weight_init)
        apply_sn_until_instance(self.predictor, PoincarePlaneDistance)
        apply_sn_until_instance(self.target, PoincarePlaneDistance)
        final_weight_init_hyp_small(self.predictor[-1])
        final_weight_init_hyp_small(self.target[-1])
        
        self.predictor = nn.Sequential(*self.predictor)
        self.target = nn.Sequential(*self.target)

        self.hyper_ball = geoopt.PoincareBall(c=1)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        # prediction_error = torch.square(target.detach() - prediction).mean(
        #     dim=-1, keepdim=True)
        ## Use hyperdistance
        prediction_error = self.hyper_ball.dist2(target.detach(), prediction)[:,None]
        return prediction_error


class HyperRNDAgent(DDPGAgent):
    def __init__(self, rnd_rep_dim, update_encoder, rnd_scale=1., **kwargs):
        super().__init__(**kwargs)
        self.rnd_scale = rnd_scale
        self.update_encoder = update_encoder

        self.rnd = RND(self.obs_dim, self.hidden_dim, rnd_rep_dim,
                       self.encoder, self.aug, self.obs_shape,
                       self.obs_type).to(self.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        # optimizers
        # self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.lr)
        self.rnd_opt = RiemannianAdam(self.rnd.parameters(), lr=self.lr)

        self.rnd.train()

    def update_rnd(self, obs, step):
        metrics = dict()

        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        self.rnd_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['rnd_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # update RND first
        if self.reward_free:
            # note: one difference is that the RND module is updated off policy
            metrics.update(self.update_rnd(obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

            metrics['pred_error_mean'] = self.intrinsic_reward_rms.M
            metrics['pred_error_std'] = torch.sqrt(self.intrinsic_reward_rms.S)

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
