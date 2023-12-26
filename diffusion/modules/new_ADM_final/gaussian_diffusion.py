"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import copy
import random

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood,softargmax2d,calculate_kl_divergence
from einops import rearrange
from utils import get_label_map_1,batch_argmax
def wrap_clamp_tensor(input_tensor, min_val, max_val):
    """
    Clamp a PyTorch tensor along the first dimension to the range [min_val, max_val] by wrapping it around.
    
    Args:
    input_tensor (torch.Tensor): The input tensor to be clamped.
    min_val (float): The minimum value of the desired range.
    max_val (float): The maximum value of the desired range.
    
    Returns:
    torch.Tensor: The clamped tensor within the specified range.
    """
    
    # Create a condition for values outside the range
    lower_bound_condition = input_tensor <= min_val
    upper_bound_condition = input_tensor >= max_val

    # Calculate range size
    range_size = max_val - min_val

    # Apply clamping element-wise
    clamped_tensor = input_tensor.clone()
    clamped_tensor[lower_bound_condition] = min_val + ((input_tensor[lower_bound_condition] - min_val) % range_size)
    clamped_tensor[upper_bound_condition] = min_val + ((input_tensor[upper_bound_condition] - min_val) % range_size)
    
    return clamped_tensor
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps,factor_multiplier):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001*factor_multiplier
        beta_end = scale * 0.02*factor_multiplier
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    VELOCITY = enum.auto() # the model predicts v


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
def identity(t, *args, **kwargs):
    return t
# normalization functions
def normalize_to_neg_value_to_value(img,value):
    new_value=th.tensor([value]).to(device=img.device,dtype=img.dtype)
    return img.mul(2) * new_value - new_value
# Unormalization functions 
def unnormalize_to_neg_value_to_value(t,value):
    new_value=th.tensor([value]).to(device=t.device,dtype=t.dtype)
    return (t + new_value)/(new_value.mul(2))

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        sample_steps,
        rescale_timesteps=False,
        auto_normalize=True,
        normalization_value=1,
        normalizaiton_std_flag=False,
        noise_changer=False,
        mse_loss_weight_type='constant',
        enforce_snr=False,

    ):
        self.mse_loss_weight_type = mse_loss_weight_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.noise_changer=noise_changer
        self.rescale_timesteps = rescale_timesteps # it should do nothing because in the end of the day i am working with 1000
        self.sample_steps=sample_steps# this is used with the samplers 
        self.normalization_value=normalization_value
        self.normalizaiton_std_flag=normalizaiton_std_flag
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        if enforce_snr:
            betas = self.enforce_zero_terminal_snr(betas)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)# used for ddim reverse sample
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        
        # why do they calll beta has the variance ???
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # very interestinggg
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.normalize = normalize_to_neg_value_to_value if auto_normalize else identity
        self.unnormalize = unnormalize_to_neg_value_to_value if auto_normalize else identity
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        use to calculate Q sample 
        UNDERSTOOD
        Equation 8 in IDDM
        Check the Improved DDPM equation8 you will find the equation
         Q(x_t | x_0) = N(x_t,
            mean---->            \sqrt(alpha_cumprod)*x_0,
            std----->            (1-alpha_cumprod)*I)
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance
    def enforce_zero_terminal_snr(self,betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_sqrt = np.sqrt(alphas_bar)

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = np.concatenate([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        # caluclating the varianceeeeee *noise +mean *x0 = posterioier 
        Equation 9 
        X_t=sqrt(alpha_cumprod)*(x_0)+sqrt(1-alpha_cumprod)*noise
        UNDERSTOOD
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)
            equation 12 found .
        UNDERSTOOD
        """
        assert x_start.shape == x_t.shape
        # this is equation 11
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # to be extracted this is equation 10
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,Flag_unetsampling=False
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.model

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        # model is just dumpy what we computed previously we pass the noised images and the time steps 
        
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]# batch by number of channels 
        assert t.shape == (B,)
        # print(type(model))
        # we come here again
        # x and ts are things in the wrapped model, the only problem in here is that model kwargs is null 
        # scale timestep is not super usefull at alll
        model_kwargs['Flag_unetsampling']=Flag_unetsampling
        model_output,inout,scene_face_feat,conditioning = model(x=x,ts=self._scale_timesteps(t), **model_kwargs)
        # we donot enter the forward function but we should return in here
        # this is where we have the error. 
        # 2,6,64,64 for the mean and teh varianceee

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # we arelearning the variance 
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:# model var is learned range
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                # we calculate equation 15 in the GANS papers. 
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]#slice the dictionary
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            # extract for the current step 
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(0, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:# this is V case!!!
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )# this is very important point. you are predicting x start in here
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON,ModelMeanType.VELOCITY]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)# you predict directly x start
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )# i output eps and i use to predict x starttt using the same main formulaaa
                # use it to find the model mean 
            else:
                pred_xstart = process_xstart(
                self._predict_xstart_from_v(x_t=x, t=t, v=model_output)
                )
            # do you understand what is xstarttt? it is the  itself 
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )# you output the model mean in here !!!!
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "model_output":model_output,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "inout": inout,
            "scene_face_feat":scene_face_feat,
            "conditioning":conditioning,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        # reformulations of equation 11
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        # equation 11 in here we can see it is it
        # we assume xprev is the mean of the equation
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )
    def _predict_eps_for_v(self, model_output, t, sample):
        # equation 9 being reformulate to extract the needed inforatmion 
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, model_output.shape) * model_output
            +_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, model_output.shape) * sample
        )
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        # equation 9 being reformulate to extract the needed inforatmion 
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    def _predict_xstart_from_v(self, x_t, t, v):
        assert x_t.shape == v.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, v.shape) * v
        )
    def _predict_v(self, x_start, t, noise):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    '''
    used in p_mean_variance
    it scale the time steps before starting to work with it and pass it to the model 
    '''
    def _scale_timesteps(self, t):
        # if nume of timesteps is 1000 this is uselesss and doesnot work at all. 
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # we computer the posterior mean and variance
        # forward process processs
        #  q(x{t_1}|x_t,x_0) 
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # you get callled in here  learning mean and variance this is the learnend reverser processss
        # p_\theta(x_{t-1}|x_{t})
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # kl duvergnecnes between our learning psotieor mean and variance 
        # and our learniend reversed process 
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )# variantional lower bound terms nothing tooo fancy in here 
        
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start,x_point, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the heatmap 
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # the difference is that the sampling is outside in here !!! 
        if model_kwargs is None:
            model_kwargs = {}
        
        if noise is None:
            if self.noise_changer:
                offset_noise = th.randn(x_point.shape[:2], device = x_point.get_device())
                noise = th.randn_like(x_point)+ model_kwargs['noise_strength']*rearrange(offset_noise, 'b c -> b c 1 1')# calculate the noise like you always do ! 
            else:
                noise = th.randn_like(x_point)
        del model_kwargs['noise_strength']
        # the forward process after adding the noise .
        # we didnot normalize in here . 
        # print("x-start min before",th.min(x_start.view(16,-1),1)[0])
        # print("x-start max before ",th.max(x_start.view(16,-1),1)[0])
        x_point = self.normalize(x_point,self.normalization_value)
        # print("x-start min after",th.min(x_start.view(16,-1),1)[0])
        # print("x-start max after",th.max(x_start.view(16,-1),1)[0])
        x_t = self.q_sample(x_point, t, noise=noise)
        x_t  = x_t / x_t.std(axis=(1), keepdims=True) if self.normalizaiton_std_flag else x_t
        # x_t = wrap_clamp_tensor(x_t,-1 * self.normalization_value,self.normalization_value)
        x_t = th.clamp(x_t, min=-1 * self.normalization_value, max=self.normalization_value)
        x_t = self.unnormalize(x_t,self.normalization_value)
        # x start is the image before doing anyhting at all 
        ## generate heatmaps
        sigma = model_kwargs['sigma']
        del model_kwargs['sigma']

        my_list = []
        for gaze_x, gaze_y in x_t:
            gaze_heatmap_i = th.zeros(64, 64)

            gaze_heatmap = get_label_map_1(
            gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma, pdf="Gaussian"
            )
            my_list.append(gaze_heatmap)
        
        x_t_final=th.stack(my_list,0)
        x_t_final=x_t_final.unsqueeze(1)
        x_t_final=x_t_final.to(x_start.device, non_blocking=True)
        ##
        # take care of normalization.
        mse_loss_weight = None
        alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
        sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
        snr = (alpha / sigma) ** 2
        if self.model_mean_type is not ModelMeanType.START_X or self.mse_loss_weight_type == 'constant':
            mse_loss_weight = th.ones_like(t)
            if self.mse_loss_weight_type.startswith("min_snr_"):
                k = float(self.mse_loss_weight_type.split('min_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0] / snr
            
            elif self.mse_loss_weight_type.startswith("max_snr_"):
                k = float(self.mse_loss_weight_type.split('max_snr_')[-1])
                # max{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0] / snr

        else:
            if self.mse_loss_weight_type == 'trunc_snr':
                # max{snr, 1}
                mse_loss_weight = th.stack([snr, th.ones_like(t)], dim=1).max(dim=1)[0]
            elif self.mse_loss_weight_type == 'snr':
                mse_loss_weight = snr

            elif self.mse_loss_weight_type == 'inv_snr':
                mse_loss_weight = 1. / snr

            elif self.mse_loss_weight_type.startswith("min_snr_"):
                k = float(self.mse_loss_weight_type.split('min_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0]

            elif self.mse_loss_weight_type.startswith("max_snr_"):
                k = float(self.mse_loss_weight_type.split('max_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0]
        if mse_loss_weight is None:
            raise ValueError(f'mse loss weight is not correctly set!')
        terms = {}
        # when we call the model 
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t_final,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # print(model)
            # print(type(model))
            # we come here next step
            model_kwargs['Flag_unetsampling']=False
            model_output,inout,_,_ = model(x=x_t_final, ts=self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t_final.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t_final.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                # frozen_out is 16,2,64,64
                '''
                *args: The *args syntax allows the lambda function to accept any number of arguments. 
                These arguments are collected into a tuple named args. 
                The * is used for unpacking the arguments when the lambda function is called.
                
                r=frozen_out: This syntax assigns a default value of frozen_out to the variable r. 
                If the lambda function is called without explicitly providing a value for r,
                it will default to the value of frozen_out.
                This specifies the expression that the lambda function will return. In this case, it simply returns the value of r.
                '''
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t_final,
                    t=t,
                    clip_denoised=False,
                )["output"]
                # we pass this dumpy model because this funciton will later be calling the model 
                # funciton calling this model we do not do a forward prob but this is how we define the model
                # no matter what you will pass here you will return r which is the frozen_out 
                # dummpy return of what we always computeted 
                # xtt the images that are noises
                # xstart are the images in the beinginngg 
                # we use t number of steps 
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t_final, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.VELOCITY: self._predict_v(x_point, t, noise),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            # make sure that network is within expected range
            # model_output = th.clamp(model_output,-1*self.normalization_value,self.normalization_value)
            copy_model = model_output.clone().squeeze(1)
            pred_location = softargmax2d(copy_model,device=copy_model.device)
            # terms["mse"] = mean_flat((target - model_output) ** 2)
            # terms['kl_new_term']=calculate_kl_divergence(target,copy_model)
            terms["inout"] = inout
            terms["location"] = pred_location
            terms["mse"] = mse_loss_weight * mean_flat((target - model_output) ** 2)
            terms["mse_raw"] = mean_flat((target - model_output) ** 2)

            # print("x-start min before",th.min(copy_model.view(16,-1),1)[0])
            # print("x-start max before ",th.max(copy_model.view(16,-1),1)[0])
            terms["output"] = self.unnormalize(copy_model,self.normalization_value)
            # print("x-start min after",th.min(terms["output"].view(16,-1),1)[0])
            # print("x-start max after ",th.max(terms["output"].view(16,-1),1)[0])
            # terms["output"] = copy_model
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
#### Normal sampling 
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        if t[0]!=self.num_timesteps - 1:
            Flag_unetsampling = True
        else:
            Flag_unetsampling = False
        x = x / x.std(axis=(1,2,3), keepdims=True) if self.normalizaiton_std_flag else x
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            Flag_unetsampling=Flag_unetsampling

        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],"scene_face_feat":out["scene_face_feat"],"conditioning":out["conditioning"],}
   


    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return self.unnormalize(final["sample"],self.normalization_value)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
                if t[0] == self.num_timesteps - 1:
                    del model_kwargs
                    model_kwargs= {'scene_face_feat':out['scene_face_feat'],
                                   'conditioning':out['conditioning'],
                      }
        
    ###############################################################################################
    ##TODO  Changes sampling directly in a different want
    def ddim_sample(
        self,
        model,
        x,
        t,
        sigma_hm,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        if t[0] ==self.num_timesteps - 1:
            x = th.randn((t.shape[0],2), device=t.device)
            noise = th.clamp(x, min=-1 * self.normalization_value, max=self.normalization_value)
            # noise = wrap_clamp_tensor(x,-1 * self.normalization_value,self.normalization_value)
            noise = self.unnormalize(noise,self.normalization_value)
            my_list = []
            for gaze_x, gaze_y in noise:
                gaze_heatmap_i = th.zeros(64, 64)

                gaze_heatmap = get_label_map_1(
                gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma_hm, pdf="Gaussian"
                )
                my_list.append(gaze_heatmap)            
            x_hm=th.stack(my_list,0)
            x_hm=x_hm.unsqueeze(1).to(t.device,non_blocking=True)
        else:
            # x = x / x.std(axis=(1,2,3), keepdims=True) if self.normalizaiton_std_flag else x
            x_point  = x / x.std(axis=(1), keepdims=True) if self.normalizaiton_std_flag else x

            x_point = th.clamp(x_point, min=-1 * self.normalization_value, max=self.normalization_value)
            # x_point = wrap_clamp_tensor(x_point,-1 * self.normalization_value,self.normalization_value)

            x_point = self.unnormalize(x_point,self.normalization_value)
            my_list = []
            for gaze_x, gaze_y in x_point:
                gaze_heatmap_i = th.zeros(64, 64)

                gaze_heatmap = get_label_map_1(
                gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma_hm, pdf="Gaussian"
                )
                my_list.append(gaze_heatmap)            
            x_hm=th.stack(my_list,0)
            x_hm=x_hm.unsqueeze(1).to(t.device,non_blocking=True)
        if t[0]!=self.num_timesteps - 1:
            Flag_unetsampling = True
        else:
            Flag_unetsampling = False
        out = self.p_mean_variance(
            model,
            x_hm,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            Flag_unetsampling=Flag_unetsampling
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        
        # new_x=  (batch_argmax(x.squeeze(),1)/64).to(x.device, non_blocking=True)
        new_out = (batch_argmax(out["pred_xstart"].squeeze(),1)/64).to(x.device, non_blocking=True)
        new_out = self.normalize(new_out,self.normalization_value)
        new_out = th.clamp(new_out, min=-1 * self.normalization_value, max=self.normalization_value)
        # new_out = wrap_clamp_tensor(new_out,-1 * self.normalization_value,self.normalization_value)

        # new_x = self.normalize(new_x,self.normalization_value)
        # new_x = th.clamp(new_x, min=-1 * self.normalization_value, max=self.normalization_value)   
        if t[0]<0:
            return {"sample": out["pred_xstart"], "pred_xstart": out["pred_xstart"],"inout":out["inout"]}
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        # extract the maxium over a batch of samples # you dividie by 64 as the return is between
        # 0 to 64
        # print(x.shape)
        # print(t.shape)
        # print(new_out.shape)

        eps = self._predict_eps_from_xstart(x, t, new_out)

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            new_out * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        sample = mean_pred + sigma * noise
        # sigma = 8

        # my_list = []
        # for gaze_x, gaze_y in sample:
        #     gaze_heatmap_i = th.zeros(64, 64)

        #     gaze_heatmap = get_label_map_1(
        #     gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma, pdf="Gaussian"
        #     )
        #     my_list.append(gaze_heatmap)
        #     # print(gaze_heatmap.shape)
        
        # x_t_final=th.stack(my_list,0)
        # x_t_final=x_t_final.unsqueeze(1)
        # x_t_final=x_t_final.to(x.device, non_blocking=True)
        return {"sample": sample, "pred_xstart": out["pred_xstart"],"inout":out["inout"],"scene_face_feat":out["scene_face_feat"],"conditioning":out["conditioning"]}

    ### DDIM SAMPLE ORGINAL
    # def ddim_sample(
    #     self,
    #     model,
    #     x,
    #     t,
    #     clip_denoised=True,
    #     denoised_fn=None,
    #     cond_fn=None,
    #     model_kwargs=None,
    #     eta=0.0,
        
    # ):
    #     """
    #     Sample x_{t-1} from the model using DDIM.

    #     Same usage as p_sample().
    #     """
    #     if t[0] ==999:
    #         # _,_,_,_,x= model(heat_map=x,time=self._scale_timesteps(t), **model_kwargs)
    #         # print(x.shape)
    #         # print(x.dtype)
    #         noise = th.randn((t.shape[0],2), device=t.device)
    #         noise = th.clamp(noise, min=-1 * self.normalization_value, max=self.normalization_value)
    #         noise = self.unnormalize(noise,self.normalization_value)
    #         sigma = 8
    #         my_list = []
    #         for gaze_x, gaze_y in noise:
    #             gaze_heatmap_i = th.zeros(64, 64)

    #             gaze_heatmap = get_label_map_1(
    #             gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma, pdf="Gaussian"
    #             )
    #             my_list.append(gaze_heatmap)            
    #         x=th.stack(my_list,0)
    #         x=x.unsqueeze(1).to(t.device,non_blocking=True)
    #     else:
    #         x = x / x.std(axis=(1,2,3), keepdims=True) if self.normalizaiton_std_flag else x
    #         x = th.clamp(x, min=-1 * self.normalization_value, max=self.normalization_value)
    #         x = self.unnormalize(x,self.normalization_value)
    #     if t[0]!=999:
    #         Flag_unetsampling = True
    #     else:
    #         Flag_unetsampling = False
    #     out = self.p_mean_variance(
    #         model,
    #         x,
    #         t,
    #         clip_denoised=clip_denoised,
    #         denoised_fn=denoised_fn,
    #         model_kwargs=model_kwargs,
    #         Flag_unetsampling=Flag_unetsampling
    #     )
    #     out["pred_xstart"] = self.normalize(out["pred_xstart"],self.normalization_value)
    #     out["pred_xstart"] = th.clamp(out["pred_xstart"], min=-1 * self.normalization_value, max=self.normalization_value)
    #     if cond_fn is not None:
    #         out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
    #     if t[0]<0:
    #         return {"sample": out["pred_xstart"], "pred_xstart": out["pred_xstart"],"inout":out["inout"]}
    #     # Usually our model outputs epsilon, but we re-derive it
    #     # in case we used x_start or x_prev prediction.
    #     if self.model_mean_type in [ModelMeanType.VELOCITY]:
    #         eps= self._predict_eps_for_v(out['model_output'], t, x)
    #     else:
    #         eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

    #     alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    #     alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
    #     sigma = (
    #         eta
    #         * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
    #         * th.sqrt(1 - alpha_bar / alpha_bar_prev)
    #     )
    #     # Equation 12.
    #     noise = th.randn_like(x)
    #     mean_pred = (
    #         out["pred_xstart"] * th.sqrt(alpha_bar_prev)
    #         + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    #     )
    #     # nonzero_mask = (
    #     #     (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    #     # )  
    #     # no noise when t == 0
    #     # if nonzero_mask.all() ==False:
    #     #     sample = out["pred_xstart"]
    #     # else:
    #     # sample = mean_pred + nonzero_mask * sigma * noise
    #     sample = mean_pred + sigma * noise

    #     return {"sample": sample, "pred_xstart": out["pred_xstart"],"inout":out["inout"],"scene_face_feat":out["scene_face_feat"],"conditioning":out["conditioning"]}
   

    def ddim_sample_loop(
        self,
        model,
        shape,
        sigma_hm,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            sigma_hm=sigma_hm
        ):
            final = sample
        return th.clamp(final["sample"],0,1),final["inout"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        sigma_hm,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn((shape[0],2), device=device)
        # I am sampling in here not till 1000 but till specific number of sample steps 
        # img = img.clamp(-1*self.normalization_value,1*self.normalization_value)
        # indices = list(range(self.sample_steps))[::-1]
        #modifcation to be done to make ddim work and start from last sample 
        indices =   list(reversed(th.linspace(-1, self.num_timesteps - 1, steps = self.sample_steps+1).int().tolist()))
        # print(indices)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    sigma_hm=sigma_hm
                )
                yield out
                img = out["sample"]
                if t[0] == self.num_timesteps - 1:
                    del model_kwargs
                    model_kwargs= {'scene_face_feat':out['scene_face_feat'],
                                   'conditioning':out['conditioning'],

                      }
    