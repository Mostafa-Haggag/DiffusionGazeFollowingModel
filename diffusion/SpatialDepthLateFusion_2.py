import torch.nn as nn
import torch.nn.functional as F
from diffusion.modules.new_ADM_final.unet import UNetModel
import math

class SpatialDepthLateFusion_2(nn.Module):
    def __init__(
        self,
        gaze_model,
        unet_inout_channels=1,
        unet_inplanes=32,
        unet_residual=1,
        unet_attention_levels=[4],
        unet_inplanes_multipliers=[1,2,4,4],
        unet_spatial_tf_heads=8,
        unet_spatial_tf_layers=1,
        unet_context_vector=1024,
        learn_sigma=False,
        dropout=0.0,
        depth_flag=False,
    ):
        super(SpatialDepthLateFusion_2, self).__init__()
        self.gaze_model = gaze_model

        self.model = UNetModel(in_channels=unet_inout_channels,
                         out_channels=(unet_inout_channels if not learn_sigma else 2*unet_inout_channels),
                         model_channels=unet_inplanes,
                         num_res_blocks=unet_residual,
                         attention_resolutions=unet_attention_levels,
                         channel_mult=unet_inplanes_multipliers,
                         num_heads=unet_spatial_tf_heads,
                         tf_layers=unet_spatial_tf_layers,
                         context_dim=unet_context_vector,
                         dropout=dropout)   
        self.channels=unet_inout_channels
        self.unet_context_vector=unet_context_vector
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
        self.depth_flag=depth_flag

    def forward(self,heat_map,time, images, face,masks,depth = None):
        # This part is related to x_loss
        # Scene_face_feat should contain output of resnet of face and head
        # Both the condiitoning and scene face feat should be changed according 
        # To what gaze model you are using
        if self.depth_flag:
            scene_face_feat,conditioning,picture=self.gaze_model(images, face,masks,depth)
        else:
            scene_face_feat,conditioning,picture=self.gaze_model(images, face,masks)

        # this part is to be used during the X-loss only 
        encoding_inout = self.sequential(scene_face_feat)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)
        x = self.model(heat_map,time,conditioning)
        return x,encoding_inout,scene_face_feat,conditioning,picture
    def forward_shorter (self,heat_map,time,scene_face_feat,conditioning,picture):

        # this part is to be used during the X-loss only 
        encoding_inout = self.sequential(scene_face_feat)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)
        x = self.model(heat_map,time,conditioning)
        return x,encoding_inout,scene_face_feat,conditioning,picture