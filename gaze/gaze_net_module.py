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
import torch.nn as nn
import torch.nn.functional as F
from gaze.modules import ResNet,Encoder
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
        depth_flag=False
        ):
            super(gaze_net_module, self).__init__()
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.scene_backbone = ResNet(in_channels=4, layers=resnet_scene_layers, inplanes=resnet_scene_inplanes)
            self.face_backbone = ResNet(in_channels=3, layers=resnet_face_layers, inplanes=resnet_face_inplanes)
            self.depth_flag=depth_flag
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
            else:
                    images = torch.cat((images, masks), dim=1)
                    if not self.attention_module:
                        scene_feat = self.scene_backbone(images)
                        scene_feat_reduced   = scene_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)
                        face_feat = self.face_backbone(face)
                        # important stuff to add in here 
                        scene_face_feat = torch.cat((scene_feat, face_feat), 1)
                        ## end of important stuff 
                        face_feat_reduced = face_feat.permute(0,2,3,1).view(-1, 49,self.unet_context_vector)
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