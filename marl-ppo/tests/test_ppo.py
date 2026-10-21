import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ppo import gae, ppo_loss


def test_gae_matches_hand_calculation():
    r = torch.tensor([[1.], [2.], [3.]])
    v = torch.tensor([[.5], [.6], [.7]])
    nv = torch.tensor([[.6], [.7], [.8]])
    term = torch.tensor([[False], [True], [False]])
    trunc = torch.tensor([[False], [False], [True]])
    adv, ret = gae(r, v, nv, term, trunc, gamma=.9, lam=.8)
    expected = torch.tensor([[1 + .9*.6 - .5 + .9*.8*(2-.6)], [2-.6], [3+.9*.8-.7]])
    torch.testing.assert_close(adv, expected)
    torch.testing.assert_close(ret, adv+v)


def test_clipping_and_gradient():
    logp = torch.tensor([1.5, .5]).log().requires_grad_()
    # positive and negative advantages clip in opposite directions
    loss, stats = ppo_loss(logp, torch.zeros(2), torch.tensor([1., -1.]),
                           torch.tensor([2., 0.]), torch.tensor([0., 0.]), torch.zeros(2))
    # policy=(-1.2 + .8)/2=-.2; vf=1; total=.3
    torch.testing.assert_close(loss, torch.tensor(.3))
    torch.testing.assert_close(stats['clip_fraction'], torch.tensor(1.))
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.zeros(2))


def test_unclipped_gradient_is_nonzero():
    lp = torch.zeros(2, requires_grad=True)
    loss, _ = ppo_loss(lp, torch.zeros(2), torch.tensor([1.,-1.]), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.tensor([-.5,.5]))


def test_truncation_bootstraps_without_episode_leakage():
    r=torch.tensor([[1.],[100.]])
    v=torch.zeros_like(r)
    nv=torch.tensor([[5.],[0.]])
    term=torch.tensor([[False],[True]])
    trunc=torch.tensor([[True],[False]])
    adv,_=gae(r,v,nv,term,trunc,gamma=.9,lam=.95)
    torch.testing.assert_close(adv,torch.tensor([[5.5],[100.]]))
