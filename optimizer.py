import torch
def get_optimizer(model, lr=2.5e-4,weight_decay=0):
    diffusion_models_weights = []
    backbone_weights = []
    for name, param in model.named_parameters():
        if name.startswith('model.'):
                diffusion_models_weights.append(param)
        else:
            backbone_weights.append(param)
    return torch.optim.Adam([{'params':diffusion_models_weights},{'params':backbone_weights,'lr': lr}], lr=lr,weight_decay=weight_decay)