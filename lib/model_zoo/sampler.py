"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)

class Sampler(object):
    def __init__(self, net, type="ddim", steps=50, output_dim=[512, 512], n_samples=4, scale=7.5):
        super().__init__()
        self.net = net
        self.type = type
        self.steps = steps
        self.output_dim = output_dim
        self.n_samples = n_samples
        self.scale = scale
        self.sigmas = ((1 - net.alphas_cumprod) / net.alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

    def get_sigmas(self, n=None):
        def append_zero(x):
            return torch.cat([x, x.new_zeros([1])])
        if n is None:
            return append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return append_zero(self.t_to_sigma(t))

    @torch.no_grad()
    def sample(self, x_info, c_info):
        h, w = self.output_dim
        shape = [self.n_samples, 4, h//8, w//8]
        device, dtype = self.net.get_device(), self.net.get_dtype()

        if ('xt' in x_info) and (x_info['xt'] is not None):
            xt = x_info['xt'].astype(dtype).to(device)
            x_info['x'] = xt
        elif ('x0' in x_info) and (x_info['x0'] is not None):
            x0 = x_info['x0'].type(dtype).to(device)
            ts = timesteps[x_info['x0_forward_timesteps']].repeat(self.n_samples)
            ts = torch.Tensor(ts).long().to(device)
            timesteps = timesteps[:x_info['x0_forward_timesteps']]
            x0_nz = self.model.q_sample(x0, ts)
            x_info['x'] = x0_nz
        else:
            x_info['x'] = torch.randn(shape, device=device, dtype=dtype)

        sigmas = self.get_sigmas(n=self.steps)

        if self.type == 'eular_a':
            rv = self.sample_euler_ancestral(
                x_info=x_info,
                c_info=c_info,
                sigmas = sigmas)
            return rv

    @torch.no_grad()
    def sample_euler_ancestral(
            self, x_info, c_info, sigmas, eta=1., s_noise=1.,):

        x = x_info['x']
        x = x * sigmas[0]

        noise_sampler = default_noise_sampler(x)

        s_in = x.new_ones([x.shape[0]])
        for i in range(len(sigmas)-1):
            denoised = self.net.apply_model(x, sigmas[i] * s_in, )

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        return x
