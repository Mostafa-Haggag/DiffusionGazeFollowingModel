# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from gaze.modules import ResNet,Encoder,Decoder
# class gaze_net_module(nn.Module):
#         def __init__(
#         self,
#         resnet_scene_layers=[3, 4, 6, 3, 2],
#         resnet_face_layers=[3, 4, 6, 3, 2],
#         resnet_depth_layers=[3, 4, 6, 3, 2],
#         resnet_scene_inplanes=64,
#         resnet_face_inplanes=64,
#         resnet_depth_inplanes=64,
#         attention_module=False,
#         unet_context_vector=1024,
#         depth_flag=False
#         ):
#             super(gaze_net_module, self).__init__()
#             self.avgpool = nn.AvgPool2d(7, stride=1)
#             self.scene_backbone = ResNet(in_channels=4, layers=resnet_scene_layers, inplanes=resnet_scene_inplanes)
#             self.face_backbone = ResNet(in_channels=3, layers=resnet_face_layers, inplanes=resnet_face_inplanes)
#             self.depth_flag=depth_flag
#             if self.depth_flag:
#                 self.depth_backbone = ResNet(in_channels=4, layers=resnet_depth_layers, inplanes=resnet_depth_inplanes)
#                 # Encoding for scene saliency
#                 self.scene_encoder = Encoder()
#                 # Encoding for depth saliency
#                 self.depth_encoder = Encoder()
#             self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#             self.attn = nn.Linear(1808, 1 * 7 * 7)
#             self.attention_module = attention_module
#             self.unet_context_vector=unet_context_vector
#         def forward(self, images, face,masks,depth=None):
#             if self.depth_flag:
#                     images = torch.cat((images, masks), dim=1)
#                     depth = torch.cat((depth, masks), dim=1)
#                     scene_feat = self.scene_backbone(images)
#                     face_feat = self.face_backbone(face)
#                     depth_feat = self.depth_backbone(depth)
#                     scene_face_feat = torch.cat((scene_feat, face_feat), 1)
#                     depth_face = torch.cat((depth_feat, face_feat), 1)
#                     # Scene encode
#                     scene_encoding = self.scene_encoder(scene_face_feat)

#                     # Depth encoding
#                     depth_encoding = self.depth_encoder(depth_face)
#                     conditioning = torch.concat([scene_encoding.view(-1, 49,self.unet_context_vector),depth_encoding.view(-1, 49,self.unet_context_vector)],1)
#             else:
#                     images = torch.cat((images, masks), dim=1)
#                     if not self.attention_module:
#                         scene_feat = self.scene_backbone(images)
#                         scene_feat_reduced   = scene_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)
#                         face_feat = self.face_backbone(face)
#                         # important stuff to add in here 
#                         scene_face_feat = torch.cat((scene_feat, face_feat), 1)

#                         ## end of important stuff 
#                         face_feat_reduced = face_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)
#                         conditioning = torch.concat([scene_feat_reduced,face_feat_reduced],1)
#                     else:
#                         scene_feat = self.scene_backbone(images) # output is (batchsize ,1024,7,7)
#                         face_feat = self.face_backbone(face)# output is (batchsize ,1024,7,7)
#                         # Reduce feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)->(N,1024)
#                         face_feat_reduced = self.avgpool(face_feat).view(-1,self.unet_context_vector)
#                         # Reduce head channel size by max pooling: (B, 1, 224, 224) -> (B, 1, 28, 28)-->(B,784)
#                         mask_reduced = self.maxpool(self.maxpool(self.maxpool(masks))).view(-1, 784)
#                         #(8,1024+784=1808) pass by the attention you get 
#                         attn_weights = self.attn(torch.cat((mask_reduced, face_feat_reduced), 1))# after oncat you have size of 8,1808
#                         attn_weights = attn_weights.view(-1, 1, 49)# add 1 dimeinson to be 8,1,49
#                         attn_weights = F.softmax(attn_weights, dim=2)
#                         attn_weights = attn_weights.view(-1, 1, 7, 7)# (8,1,7,7)
#                         # attention weights has size of (batchsize,1,7,7)
#                         # you remember the scene features where equal to  (batchsize ,1024,7,7)
#                         attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)# 8,1024,7,7
#                         scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
#                         conditioning = torch.concat([attn_applied_scene_feat.view(-1, 49,self.unet_context_vector),face_feat.view(-1, 49,self.unet_context_vector)],1)
#             return scene_face_feat,conditioning
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gaze.modules import ResNet,Encoder
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        # tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc

    @property
    def org_channels(self):
        return self.penc.org_channels

class gaze_net_module(nn.Module):
        def __init__(
        self,
        resnet_scene_layers=[3, 4, 6, 3, 2],
        resnet_face_layers=[3, 4, 6, 3, 2],
        resnet_depth_layers=[3, 4, 6, 3, 2],
        resnet_scene_inplanes=64,
        resnet_face_inplanes=64,
        resnet_depth_inplanes=64,
        attention_module=False,
        unet_context_vector=1024,
        depth_flag=False,
        head_flag= True
        ):
            super(gaze_net_module, self).__init__()
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.scene_backbone = ResNet(in_channels=4, layers=resnet_scene_layers, inplanes=resnet_scene_inplanes)
            self.face_backbone = ResNet(in_channels=3, layers=resnet_face_layers, inplanes=resnet_face_inplanes)
            self.depth_flag=depth_flag
            self.head_flag = head_flag
            if self.depth_flag:
                self.depth_backbone = ResNet(in_channels=4, layers=resnet_depth_layers, inplanes=resnet_depth_inplanes)
                # Encoding for scene saliency
                self.scene_encoder = Encoder()
                # Encoding for depth saliency
                self.depth_encoder = Encoder()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.attn = nn.Linear(1808, 1 * 7 * 7)
            self.attention_module = attention_module
            self.unet_context_vector=unet_context_vector
            self.postional_embedding_1 = PositionalEncodingPermute2D(self.unet_context_vector)
            self.postional_embedding_2 = PositionalEncodingPermute2D(self.unet_context_vector)
        def forward(self, images, face,masks,depth=None):
            if self.depth_flag:
                    images = torch.cat((images, masks), dim=1)
                    depth = torch.cat((depth, masks), dim=1)
                    scene_feat = self.scene_backbone(images)
                    face_feat = self.face_backbone(face)
                    depth_feat = self.depth_backbone(depth)
                    scene_face_feat = torch.cat((scene_feat, face_feat), 1)
                    scene_depth_feat = torch.cat((scene_feat, depth_feat), 1)
                    # Scene encode
                    scene_encoding = self.scene_encoder(scene_depth_feat)

                    # Depth encoding
                    conditioning = torch.concat([scene_encoding.permute(0,2,3,1).view(-1, 49,self.unet_context_vector),face_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)],1)
            elif self.head_flag == False:
                    images = torch.cat((images, masks), dim=1)
                    scene_feat = self.scene_backbone(images)
                    scene_feat_reduced   = scene_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)
                    # important stuff to add in here 
                    scene_face_feat = scene_feat
                    conditioning = scene_feat_reduced
            else:
                    images = torch.cat((images, masks), dim=1)
                    if not self.attention_module:
                        scene_feat = self.scene_backbone(images)
                        scene_feat_reduced   = (self.postional_embedding_1(scene_feat.permute(0,2,3,1))+scene_feat.permute(0,2,3,1)).view(-1, 49,self.unet_context_vector)
                        face_feat = self.face_backbone(face)
                        scene_face_feat = torch.cat((scene_feat, face_feat), 1)
                        face_feat_reduced   = (self.postional_embedding_2(face_feat.permute(0,2,3,1))+face_feat.permute(0,2,3,1)).view(-1, 49,self.unet_context_vector)
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
                        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
                        conditioning = torch.concat([attn_applied_scene_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector),face_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)],1)
            return scene_face_feat,conditioning