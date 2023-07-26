import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ResNet
from models.modules.new_ADM_final.unet import UNetModel
from einops import rearrange, repeat
from functools import partial
import math

class SpatialDepthLateFusion_2(nn.Module):
    def __init__(
        self,
        resnet_scene_layers=[3, 4, 6, 3, 2],
        resnet_face_layers=[3, 4, 6, 3, 2],
        resnet_scene_inplanes=64,
        resnet_face_inplanes=64,
        unet_inout_channels=1,
        unet_inplanes=32,
        unet_residual=1,
        unet_attention_levels=[4],
        unet_inplanes_multipliers=[1,2,4,4],
        unet_spatial_tf_heads=8,
        unet_spatial_tf_layers=1,
        unet_context_vector=1024,
        learn_sigma=False,
        use_fp16=False,
        attention_module=False,
        dropout=0.0,
        cond_drop_prob=0.,
        cond_scale=1.0,
        rescaled_phi=0,
    ):
        super(SpatialDepthLateFusion_2, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.scene_backbone = ResNet(in_channels=4, layers=resnet_scene_layers, inplanes=resnet_scene_inplanes)
        self.face_backbone = ResNet(in_channels=3, layers=resnet_face_layers, inplanes=resnet_face_inplanes)
        self.model = UNetModel(in_channels=unet_inout_channels,
                         out_channels=(unet_inout_channels if not learn_sigma else 2*unet_inout_channels),
                         model_channels=unet_inplanes,
                         num_res_blocks=unet_residual,
                         attention_resolutions=unet_attention_levels,
                         channel_mult=unet_inplanes_multipliers,
                         num_heads=unet_spatial_tf_heads,
                         tf_layers=unet_spatial_tf_layers,
                         context_dim=unet_context_vector,
                         use_fp16=use_fp16,
                         dropout=dropout)   
        self.channels=unet_inout_channels
        self.unet_context_vector=unet_context_vector
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.attn = nn.Linear(1808, 1 * 7 * 7)
        self.attention_module = attention_module
        self.cond_drop_prob=cond_drop_prob
        self.cond_scale=cond_scale
        self.rescaled_phi=rescaled_phi
        if self.cond_drop_prob>0:
            self.null_classes_emb = nn.Parameter(torch.randn([98,unet_context_vector]))
        self.sequential = nn.Sequential(    nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                            nn.BatchNorm2d(1),
                                            nn.ReLU(inplace=True)
                                        )
        self.fc_inout = nn.Linear(49, 1)
        for m in self.sequential.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,heat_map,time, images, face,masks):
        images = torch.cat((images, masks), dim=1)
        if not self.attention_module:
            scene_feat = self.scene_backbone(images)
            # scene_feat_reduced = self.avgpool(scene_feat).view(-1, 1,self.unet_context_vector)
            scene_feat_reduced   = scene_feat.view(-1, 49,self.unet_context_vector)
            face_feat = self.face_backbone(face)
            # important stuff to add in here 
            scene_face_feat = torch.cat((scene_feat, face_feat), 1)
            encoding_inout = self.sequential(scene_face_feat)
            encoding_inout = encoding_inout.view(-1, 49)
            encoding_inout = self.fc_inout(encoding_inout)
            ## end of important stuff 
            # face_feat_reduced = self.avgpool(face_feat).view(-1, 1,self.unet_context_vector)
            face_feat_reduced = face_feat.view(-1, 49,self.unet_context_vector)
            conditioning = torch.concat([scene_feat_reduced,face_feat_reduced],1)
            
        else:
            scene_feat = self.scene_backbone(images) # output is (batchsize ,1024,7,7)

            face_feat = self.face_backbone(face)# output is (batchsize ,1024,7,7)
            # Reduce feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)->(N,1024)
            face_feat_reduced = self.avgpool(face_feat).view(-1,self.unet_context_vector)
            
            # Reduce head channel size by max pooling: (B, 1, 224, 224) -> (B, 1, 28, 28)-->(B,784)
            mask_reduced = self.maxpool(self.maxpool(self.maxpool(masks))).view(-1, 784)
            #(8,1024+784=1808) pass by the attention you get 
            attn_weights = self.attn(torch.cat((mask_reduced, face_feat_reduced), 1))# after oncat you have size of 8,1808
            attn_weights = attn_weights.view(-1, 1, 49)# add 1 dimeinson to be 8,1,49
            attn_weights = F.softmax(attn_weights, dim=2)
            attn_weights = attn_weights.view(-1, 1, 7, 7)# (8,1,7,7)
            # attention weights has size of (batchsize,1,7,7)
            # you remember the scene features where equal to  (batchsize ,1024,7,7)
            attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)# 8,1024,7,7
            # you must average pool it to go down to the proper size 
            # scene_feat_attn_reduced = self.avgpool(attn_applied_scene_feat).view(-1,self.unet_context_vector)
            # important stuff to add in here 
            scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
            encoding_inout = self.sequential(scene_face_feat)
            encoding_inout = encoding_inout.view(-1, 49)
            encoding_inout = self.fc_inout(encoding_inout)
            ## end of important stuff 
            conditioning = torch.concat([attn_applied_scene_feat.view(-1, 49,self.unet_context_vector),face_feat.view(-1, 49,self.unet_context_vector)],1)
        if self.cond_drop_prob>0:
            batch =conditioning.shape[0]
            device = conditioning.get_device()
            # keep_mask = (torch.rand(batch)>self.cond_drop_prob).to(device) 
            keep_mask = self.prob_mask_like((batch,), 1 - self.cond_drop_prob, device = device)# you have size of batch size 
            null_classes_emb = repeat(self.null_classes_emb, 'd c -> b d c', b = batch)
            conditioning = torch.where(
                rearrange(keep_mask, 'b -> b 1 1'),
                conditioning,
                null_classes_emb
            )
        x = self.model(heat_map,time,conditioning)
        return x,encoding_inout
    def forward_with_cond_scale(
        self,
        *args,
        **kwargs
        ):
        orginal_value = self.cond_drop_prob
        self.cond_drop_prob = 0
        logits = self.forward(*args, **kwargs)
        self.cond_drop_prob = 1
        null_logits = self.forward(*args,**kwargs)
        scaled_logits = null_logits + (logits - null_logits).mul(self.cond_scale)
        if self.rescaled_phi == 0.:
            self.cond_drop_prob = orginal_value
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        self.cond_drop_prob = orginal_value

        return rescaled_logits.mul(self.rescaled_phi) + scaled_logits.mul(1. - self.rescaled_phi)
        
    def prob_mask_like(self,shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
