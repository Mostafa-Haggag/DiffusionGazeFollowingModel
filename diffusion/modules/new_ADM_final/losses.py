"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th

def calculate_kl_divergence(heatmap_p, heatmap_q):
    # Flatten the heatmaps
    p = heatmap_p.view(-1,4096)
    q = heatmap_q.view(-1,4096)
    
    # # Add a small epsilon to avoid division by zero
    epsilon =  th.full(p.shape, 1e-8).to(p.device, dtype=p.dtype)

    p = p + epsilon
    q = q + epsilon
 
    # # Normalize the heatmaps
    p = p / p.sum(1).unsqueeze(1).expand(p.shape)
    q = q / q.sum(1).unsqueeze(1).expand(q.shape)

    # # Calculate KL divergence
    kl_divergence = th.sum(p * th.log(p / q),1).mean()
    return kl_divergence

def softargmax2d(input, beta=100,device='0'):
    *_, h, w = input.shape
    input = input.reshape(*_, h * w)
    input = th.nn.functional.softmax(beta * input, dim=-1,dtype=input.dtype)# bring out float 32

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )# float 64

    indices_r = th.tensor(np.reshape(indices_r, (-1, h * w))).cuda().to(device, dtype=input.dtype)
    indices_c = th.tensor(np.reshape(indices_c, (-1, h * w))).cuda().to(device, dtype=input.dtype)

    result_r = th.sum((h - 1) * input * indices_r, dim=-1)
    result_c = th.sum((w - 1) * input * indices_c, dim=-1)

    result = th.stack([result_r, result_c], dim=-1)

    return result

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
