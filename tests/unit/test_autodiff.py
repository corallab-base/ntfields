import torch

from ntfields.autodiff import grad_and_hessian_diag


def test_hessian_diag_closed_form():
    x = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    y = (x[:, 0] ** 3) * x[:, 1] + torch.sin(x[:, 2])
    g, h = grad_and_hessian_diag(y, x)
    a, b, c = x.detach().unbind(-1)
    torch.testing.assert_close(g, torch.stack([3 * a**2 * b, a**3, torch.cos(c)], dim=-1))
    torch.testing.assert_close(h, torch.stack([6 * a * b, torch.zeros_like(b), -torch.sin(c)], dim=-1))


def test_hessian_diag_matches_finite_differences():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Softplus(beta=10), torch.nn.Linear(16, 1)).double()
    x = torch.rand(5, 4, dtype=torch.float64, requires_grad=True)
    _, h = grad_and_hessian_diag(net(x).squeeze(-1), x)
    eps = 1e-4
    with torch.no_grad():
        f0 = net(x).squeeze(-1)
        fd = torch.stack(
            [
                (net(x + eps * e).squeeze(-1) - 2 * f0 + net(x - eps * e).squeeze(-1)) / eps**2
                for e in torch.eye(4, dtype=torch.float64)
            ],
            dim=-1,
        )
    torch.testing.assert_close(h, fd, rtol=1e-5, atol=1e-6)
