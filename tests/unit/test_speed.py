import torch

from ntfields.speed import Clipped, ClippedEased, Smoothstep


def test_clipped():
    s = Clipped(dmin=0.01, dmax=0.1)(torch.tensor([-1.0, 0.05, 1.0]))
    torch.testing.assert_close(s, torch.tensor([0.1, 0.5, 1.0]))


def test_smoothstep_range():
    s = Smoothstep(dmax=0.1, floor=0.1)(torch.tensor([-1.0, 0.0, 0.05, 0.1, 1.0]))
    torch.testing.assert_close(s, torch.tensor([0.1, 0.1, 0.55, 1.0, 1.0]))


def test_clipped_eased():
    s = ClippedEased(dmin=0.01, dmax=0.1)(torch.tensor([-1.0, 0.05, 1.0]))
    torch.testing.assert_close(s, torch.tensor([0.1**2 * 1.9**2, 0.5**2 * 1.5**2, 1.0]))
