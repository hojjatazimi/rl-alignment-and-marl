import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from common.experiment import Logger, seed_all
from ppo import ActorCritic, gae, update


def make_env(render=False):
    return simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25,
                                        continuous_actions=False, render_mode='rgb_array' if render else None)


def observe(env, obs):
    return np.stack([obs[a] for a in env.possible_agents]), np.array(env.state())


def geometry(env):
    world = env.unwrapped.world
    positions = np.stack([a.state.p_pos for a in world.agents])
    targets = np.stack([l.state.p_pos for l in world.landmarks])
    distances = np.linalg.norm(targets[:,None]-positions[None,:], axis=-1)
    # Fraction of landmarks within a single agent radius; unique pairs per step.
    coverage = (distances.min(1) < world.agents[0].size).mean()
    collisions = sum(np.linalg.norm(positions[i]-positions[j]) < world.agents[i].size+world.agents[j].size
                     for i in range(3) for j in range(i+1,3))
    return float(coverage), float(collisions)


@torch.no_grad()
def evaluate(model, seed=20000, episodes=20, gif=None):
    env = make_env(render=bool(gif))
    returns, coverage, collisions, frames = [], [], [], []
    for i in range(episodes):
        obs,_ = env.reset(seed=seed+i)
        ret, cov, col, count = 0.,0.,0.,0
        while env.agents:
            x,s = observe(env,obs)
            action,_,_ = model.act(torch.tensor(x)[None],torch.tensor(s)[None],greedy=True)
            obs,reward,_,_,_ = env.step(dict(zip(env.possible_agents,action[0].tolist())))
            c,k = geometry(env)
            ret += np.mean(list(reward.values())); cov+=c; col+=k; count+=1
            if gif and i==0:
                frames.append(env.render())
        returns.append(ret); coverage.append(cov/count); collisions.append(col/count)
    if gif:
        import imageio.v2 as imageio
        imageio.mimsave(gif,frames,duration=0.1,loop=0)
    env.close()
    return dict(eval_return=np.mean(returns), eval_return_std=np.std(returns),
                coverage=np.mean(coverage), collisions=np.mean(collisions))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algo',choices=['ippo','mappo','shared-local','independent-central'],default='mappo')
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--envs',type=int,default=4)
    p.add_argument('--horizon',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=10000)
    p.add_argument('--eval-episodes',type=int,default=20)
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--out',required=True)
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--gif',action='store_true')
    a=p.parse_args(); seed_all(a.seed)
    if a.steps % a.envs:
        p.error('--steps must be divisible by --envs (exact joint environment-step budget)')
    share=a.algo in ['mappo','shared-local']; central=a.algo in ['mappo','independent-central']
    log=Logger(a.out,dict(vars(a),kind='marl',share=share,centralized=central),wandb=a.wandb)
    envs=[make_env() for _ in range(a.envs)]
    observed=[observe(e,e.reset(seed=a.seed*100+i)[0]) for i,e in enumerate(envs)]
    obs,state=map(np.stack,zip(*observed))
    model=ActorCritic(obs.shape[-1],state.shape[-1],envs[0].action_space('agent_0').n,3,share,central)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,eps=1e-5)
    scores=np.zeros(a.envs); total=last_eval=0
    while total<a.steps:
        buffers=[[] for _ in range(9)]
        for _ in range(min(a.horizon,(a.steps-total)//a.envs)):
            x,s=torch.tensor(obs),torch.tensor(state)
            act,lp,v=model.act(x,s)
            nxt=[]; rewards=[]; terms=[]; truncs=[]
            for i,e in enumerate(envs):
                o,r,t,u,_=e.step(dict(zip(e.possible_agents,act[i].tolist())))
                nxt.append(observe(e,o)); rewards.append([r[k] for k in e.possible_agents])
                terms.append([t[k] for k in e.possible_agents]); truncs.append([u[k] for k in e.possible_agents])
            no,ns=map(np.stack,zip(*nxt))
            with torch.no_grad():
                nv=model(torch.tensor(no),torch.tensor(ns))[1]
            entries=(x,s,act,lp,v,torch.tensor(rewards,dtype=torch.float32),nv,torch.tensor(terms),torch.tensor(truncs))
            for buf,z in zip(buffers,entries): buf.append(z)
            total+=a.envs; scores+=np.mean(rewards,axis=1)
            obs,state=no,ns
            for i,e in enumerate(envs):
                if any(terms[i]) or any(truncs[i]):
                    log.log(total,episode_return=scores[i]); scores[i]=0
                    obs[i],state[i]=observe(e,e.reset()[0])
        x,s,act,lp,v,r,nv,t,u=[torch.stack(b) for b in buffers]
        adv,ret=gae(r,v,nv,t,u)
        batch=tuple(z.flatten(0,1) for z in (x,s,act,lp,adv,ret))
        log.log(total,**update(model,opt,batch))
        if total-last_eval>=a.eval_every or total>=a.steps:
            stats=evaluate(model,episodes=a.eval_episodes)
            log.log(total,**stats); last_eval=total
            print(f'{a.algo} seed={a.seed} steps={total}: {stats}',flush=True)
    torch.save(dict(model=model.state_dict(),args=vars(a)),Path(a.out)/'policy.pt')
    if a.gif:
        evaluate(model,episodes=1,gif=str(Path(a.out)/'policy.gif'))
    for e in envs: e.close()
    log.close()
    (Path(a.out)/'complete.json').write_text(json.dumps(dict(steps=total,**stats),indent=2))

if __name__=='__main__': main()
