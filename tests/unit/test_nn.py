import torch

from ntfields.nn import RandomFourierFeatures


def test_fourier_features():
    ff = RandomFourierFeatures(dim=3, n=8, scale=0.2)
    q = torch.randn(5, 3)
    out = ff(q)
    assert out.shape == (5, ff.out_features) == (5, 16)
    torch.testing.assert_close(out[:, :8] ** 2 + out[:, 8:] ** 2, torch.ones(5, 8))
    assert "b" in ff.state_dict()
