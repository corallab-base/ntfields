import torch

from ntfields import SceneCondition
from ntfields.methods.td_ntfields.model import TDMetricModel


def test_shapes_symmetry_and_self_distance():
    torch.manual_seed(0)
    model = TDMetricModel(dim=3, hidden=64)
    q_s, q_g = torch.rand(10, 3), torch.rand(10, 3)
    scene = SceneCondition()
    t = model(q_s, q_g, scene)
    assert t.shape == (10,) and (t > 0).all()
    torch.testing.assert_close(t, model(q_g, q_s, scene))
    # sqrt(Δ² + 1e-6) floors every coordinate at 1e-3, so T(q, q) = out_scale · groups · 1e-3
    torch.testing.assert_close(model(q_s, q_s, scene), torch.full((10,), 0.1 * 4 * 1e-3), atol=1e-6, rtol=0)
