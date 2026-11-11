"""Interval Quasimetric Embedding (Wang & Isola, https://arxiv.org/abs/2211.15120), IQE-maxmean only.

Vendored from torch-quasimetric (github.com/quasimetric-learning/torch-quasimetric, `torchqmet/iqe.py` and the
`maxmean` reduction), with the parameter names kept so its state dicts load unchanged.

Copyright (c) 2022 Tongzhou Wang. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torch import Tensor, nn


def iqe_components(x: Tensor, y: Tensor) -> Tensor:
    """Per-component IQE distance: the length of the union of the intervals [x_i, y_i] with x_i < y_i."""
    d = x.shape[-1]
    valid = x < y
    xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)
    sxy, ixy = xy.sort(dim=-1)
    neg_inc_copies = torch.gather(valid, dim=-1, index=ixy % d) * torch.where(ixy < d, -1, 1)
    neg_inp_copies = torch.cumsum(neg_inc_copies, dim=-1)
    neg_f = (neg_inp_copies < 0) * (-1.0)
    neg_incf = torch.cat([neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1)
    return (sxy * neg_incf).sum(-1)


class MaxMean(nn.Module):
    """Convex combination of the max and the mean of the components, with a learned weight."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.ones(()).neg_())  # pre-sigmoid

    def forward(self, d: Tensor) -> Tensor:
        return torch.lerp(d.mean(dim=-1), d.max(dim=-1).values, self.raw_alpha.sigmoid())


class IQE(nn.Module):
    """IQE-maxmean quasimetric between two (..., input_size) embeddings; d(x, x) = 0 and the triangle inequality hold
    by construction, symmetry does not."""

    def __init__(self, input_size: int, dim_per_component: int = 16) -> None:
        super().__init__()
        if input_size % dim_per_component:
            raise ValueError(f"input_size={input_size} is not divisible by dim_per_component={dim_per_component}")
        self.shape = torch.Size([input_size // dim_per_component, dim_per_component])
        self.reduction = MaxMean()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.reduction(iqe_components(x.unflatten(-1, self.shape), y.unflatten(-1, self.shape)))
