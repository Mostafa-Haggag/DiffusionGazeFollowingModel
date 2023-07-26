import torch


from .SpatialDepthLateFusion_2 import SpatialDepthLateFusion_2


def get_model(config, device=torch.device("cuda")):
    # this has everything in here where we intialize the backbone 

    model = SpatialDepthLateFusion_2(
            resnet_scene_layers=config.list_resnet_scene_layers,
            resnet_face_layers=config.list_resnet_face_layers,
            resnet_scene_inplanes=config.resnet_scene_inplanes,
            resnet_face_inplanes=config.resnet_face_inplanes,
            unet_inout_channels=config.unet_inout_channels,
            unet_inplanes=config.unet_inplanes,
            unet_residual=config.unet_residual,
            unet_attention_levels=config.list_unet_attention_levels,
            unet_inplanes_multipliers=config.list_unet_inplanes_multipliers,
            unet_spatial_tf_heads=config.unet_spatial_tf_heads,
            unet_spatial_tf_layers=config.unet_spatial_tf_layers,
            unet_context_vector=config.unet_context_vector,
            learn_sigma=config.adm_learn_sigma,
            attention_module=config.adm_attention_module,
            dropout=config.unet_dropout,
            )
    """
    unused
            cond_drop_prob=config.cond_drop_prob,
            cond_scale=config.cond_scale,
            rescaled_phi=config.rescaled_phi
    """
    modules = []
    # all the frezing stuff is set to false 
    if config.freeze_scene:
        modules += [model.scene_backbone]

    if config.freeze_face:
        modules += [model.face_backbone]
    for module in modules:
        for layer in module.children():
            for param in layer.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")
    return model


def load_pretrained(model, pretrained_dict):
    if model is None:
        return model
    if pretrained_dict is None:
        print("Pretraining is None")
        return model
    model_dict = model.state_dict()# the weights that are already loaded
    model_dict.update(pretrained_dict)
    print(model.load_state_dict(model_dict, strict=False))
    return model

