import argparse
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(
    *,
    steps=1000,
    sample_steps=250,
    learn_sigma=False,
    sigma_small=True,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="ddim",
    auto_normalize=True,
    normalization_value=1,
    normalizaiton_std_flag=False,
    noise_changer=False,
    mse_loss_weight_type='constant',
    predict_v=False,
    enforce_snr=False,
    factor_multiplier=10
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps,factor_multiplier = factor_multiplier )
    if use_kl:
        # loss_type = gd.LossType.RESCALED_KL
        loss_type = gd.LossType.KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]#he set the number of time steps to list with steps 
    print("My loss type is ",str(loss_type))
    if learn_sigma:
        print("My var type is ",str(gd.ModelVarType.LEARNED_RANGE))
    elif sigma_small:
        print("My var type is ",str(gd.ModelVarType.FIXED_SMALL))
    else:
        print("My var type is ",str(gd.ModelVarType.FIXED_LARGE))
    if predict_xstart: 
        print("My mean type is ",str(gd.ModelMeanType.START_X))
        the_mean = gd.ModelMeanType.START_X
    elif predict_v:
        print("My mean type is ",str(gd.ModelMeanType.VELOCITY))
        the_mean = gd.ModelMeanType.VELOCITY
    else:
        print("My mean type is ",str(gd.ModelMeanType.EPSILON))
        the_mean = gd.ModelMeanType.EPSILON

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            the_mean
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        sample_steps=sample_steps,
        auto_normalize=auto_normalize,
        normalization_value=normalization_value,
        normalizaiton_std_flag=normalizaiton_std_flag,
        noise_changer=noise_changer,
        mse_loss_weight_type=mse_loss_weight_type,
        enforce_snr=enforce_snr,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
