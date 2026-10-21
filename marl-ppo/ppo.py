"""PPO without an RL training library. Shapes: [time, environment, agent]."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    """Bootstrap time limits, but never carry advantages across episode boundaries."""
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        adv[t] = carry
    return adv, adv + values


def ppo_loss(logp, old_logp, advantages, value, returns, entropy, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages).mean()
    vf = 0.5 * (value - returns).square().mean()
    total = policy + vf_coef * vf - ent_coef * entropy.mean()
    with torch.no_grad():
        kl = ((ratio - 1) - (logp - old_logp)).mean()
        clipped = ((ratio - 1).abs() > clip).float().mean()
    return total, dict(policy_loss=policy, value_loss=vf, entropy=entropy.mean(), approx_kl=kl, clip_fraction=clipped)


def mlp(inputs, outputs, final_gain=1):
    net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, final_gain)
    return net


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, state_dim, actions, agents=1, share=True, centralized=False):
        super().__init__()
        self.agents, self.share, self.centralized = agents, share, centralized
        # Shared actor receives an identity vector to break agent symmetry explicitly.
        self.actors = nn.ModuleList([mlp(obs_dim + (agents if share and agents > 1 else 0), actions, 0.01)
                                    for _ in range(1 if share else agents)])
        self.critics = nn.ModuleList([mlp(state_dim if centralized else obs_dim, 1)
                                     for _ in range(1 if share else agents)])

    def forward(self, obs, state):
        logits, values = [], []
        for a in range(self.agents):
            x = obs[:, a]
            if self.share and self.agents > 1:
                ids = torch.eye(self.agents, device=x.device)[a].expand(len(x), -1)
                x = torch.cat([x, ids], -1)
            i = 0 if self.share else a
            logits.append(self.actors[i](x))
            values.append(self.critics[i](state if self.centralized else obs[:, a]).squeeze(-1))
        return Categorical(logits=torch.stack(logits, 1)), torch.stack(values, 1)

    @torch.no_grad()
    def act(self, obs, state, greedy=False):
        dist, value = self(obs, state)
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value


def update(model, optimizer, batch, epochs=4, minibatch=256, ent_coef=0.01, clip=0.2):
    obs, states, actions, old_logp, advantages, returns = batch
    # Normalize per agent, preventing agents with high reward scale dominating IPPO.
    advantages = (advantages - advantages.mean(0)) / (advantages.std(0, unbiased=False) + 1e-8)
    collected = []
    for _ in range(epochs):
        for idx in torch.randperm(len(obs)).split(minibatch):
            dist, value = model(obs[idx], states[idx])
            loss, stats = ppo_loss(dist.log_prob(actions[idx]), old_logp[idx], advantages[idx], value,
                                   returns[idx], dist.entropy(), clip=clip, ent_coef=ent_coef)
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            collected.append(dict(stats, grad_norm=grad))
    return {k: float(torch.stack([x[k].detach() for x in collected]).mean()) for k in collected[0]}
