"""Train on real Gymnasium environments, with fixed held-out evaluation seeds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from common.experiment import Logger, seed_all
from ppo import ActorCritic, gae, update


def evaluate(model, env_id, episodes=20, seed=10000):
    env = gym.make(env_id)
    scores = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed+i)
        score = 0
        while True:
            x = torch.as_tensor(obs, dtype=torch.float32)[None, None]
            action, _, _ = model.act(x, x[:, 0], greedy=True)
            obs, reward, term, trunc, _ = env.step(action.item())
            score += reward
            if term or trunc:
                break
        scores.append(score)
    env.close()
    return float(np.mean(scores)), float(np.std(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='CartPole-v1', choices=['CartPole-v1', 'LunarLander-v3'])
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=10000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--out', required=True)
    p.add_argument('--wandb', action='store_true')
    a = p.parse_args()
    if a.steps % a.envs:
        p.error('--steps must be divisible by --envs')
    seed_all(a.seed)
    log = Logger(a.out, dict(vars(a), kind='single'), wandb=a.wandb)
    # Allocate individual environments so timeout final observations remain accessible.
    envs = [gym.make(a.env) for _ in range(a.envs)]
    obs = np.stack([e.reset(seed=a.seed*100+i)[0] for i,e in enumerate(envs)])
    model = ActorCritic(obs.shape[-1], obs.shape[-1], envs[0].action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    episode_returns = np.zeros(a.envs)
    total, last_eval = 0, 0
    while total < a.steps:
        storage = [[] for _ in range(8)]
        for _ in range(min(a.horizon, (a.steps-total+a.envs-1)//a.envs)):
            x = torch.tensor(obs, dtype=torch.float32)[:,None]
            actions, logp, values = model.act(x, x[:,0])
            next_obs, rewards, terms, truncs = [], [], [], []
            for i,e in enumerate(envs):
                o,r,t,u,_ = e.step(actions[i].item())
                next_obs.append(o); rewards.append(r); terms.append(t); truncs.append(u)
            nx = torch.tensor(np.array(next_obs), dtype=torch.float32)[:,None]
            with torch.no_grad():
                nv = model(nx, nx[:,0])[1]
            entries = (x, actions, logp, values, torch.tensor(rewards)[:,None], nv,
                       torch.tensor(terms)[:,None], torch.tensor(truncs)[:,None])
            for buf,v in zip(storage, entries):
                buf.append(v)
            episode_returns += rewards
            obs = np.array(next_obs)
            total += a.envs
            for i in range(a.envs):
                if terms[i] or truncs[i]:
                    log.log(total, episode_return=episode_returns[i])
                    episode_returns[i] = 0
                    obs[i] = envs[i].reset()[0]
        x,actions,lp,v,r,nv,t,u = [torch.stack(b) for b in storage]
        adv,ret = gae(r,v,nv,t,u)
        flat = lambda z: z.flatten(0,1)
        stats = update(model,opt,(flat(x), flat(x)[:,0], flat(actions),flat(lp),flat(adv),flat(ret)))
        log.log(total, **stats)
        if total-last_eval >= a.eval_every or total >= a.steps:
            mean,std = evaluate(model,a.env,a.eval_episodes)
            log.log(total, eval_return=mean, eval_return_std=std)
            print(f'{a.env} seed={a.seed} steps={total} eval={mean:.2f} ± {std:.2f}', flush=True)
            last_eval = total
    torch.save(dict(model=model.state_dict(), args=vars(a)), Path(a.out)/'policy.pt')
    for e in envs:
        e.close()
    log.close()
    (Path(a.out)/'complete.json').write_text(json.dumps(dict(steps=total,eval_return=mean,eval_return_std=std),indent=2))

if __name__ == '__main__':
    main()
