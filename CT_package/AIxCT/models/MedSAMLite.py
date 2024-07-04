
from .modeling_SAMLite import MaskDecoder, PromptEncoder, TwoWayTransformer, TinyViT
import torch
from torch import nn
from tiny_vit_sam import TinyViT
import numpy as np



class MedSAMLite(nn.Module):
    def __init__(self, output_channels, use_boxes = False):
        super().__init__()
        
        self.output_channels = output_channels
        if self.output_channels > 1:
            self.multimask_output = True
        else:
            self.multimask_output = False
            
        self.use_boxes = use_boxes
        
        self.image_encoder = TinyViT(img_size=256, 
                                     in_chans=3, 
                                     embed_dims=[64, 128, 160, 320 ], 
                                     depths=[2, 2, 6, 2], 
                                     num_heads=[2, 4, 5, 10], 
                                     window_sizes=[7, 7, 14, 7], 
                                     mlp_ratio=4., 
                                     drop_rate=0., 
                                     drop_path_rate=0.0, 
                                     use_checkpoint=False, 
                                     mbconv_expand_ratio=4.0, 
                                     local_conv_size=3, 
                                     layer_lr_decay=0.8)
        
        self.mask_decoder = MaskDecoder(num_multimask_outputs=self.output_channels, 
                                        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,), 
                                        transformer_dim=256, 
                                        iou_head_depth=3, 
                                        iou_head_hidden_dim=256,)
        
        self.prompt_encoder = PromptEncoder(embed_dim=256, 
                                            image_embedding_size=(64, 64), 
                                            input_image_size=(256, 256), 
                                            mask_in_chans=16)
        
        self.load_weights("CT_package/AIxCT/weights_finetune/lite_medsam.pth")
        self.freeze_encoder()
        
    def forward(self, image, boxes):
        
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        #print("image embedding size", image_embedding.size())

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        #print("sparse embeddings size", sparse_embeddings.size())
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=self.multimask_output,
          ) # (B, 1, 256, 256)
        #print("low_res_masks", low_res_masks.size())

        return low_res_masks
    
    def load_weights(self, weight_path):
        
        state_dict = torch.load(weight_path)
        image_encoder_dict = {k: v for k, v in state_dict.items() if 'image_encoder' in k}
        prompt_encoder_dict = {k: v for k, v in state_dict.items() if 'prompt_encoder' in k}
        mask_decoder_dict = {k: v for k, v in state_dict.items() if 'mask_decoder' in k}
        
        self.image_encoder.load_state_dict(image_encoder_dict, strict=False)
        self.prompt_encoder.load_state_dict(prompt_encoder_dict, strict=False)
        self.mask_decoder.load_state_dict(mask_decoder_dict, strict=False)
        
    def freeze_encoder(self):
        
        # make sure we only compute gradients for mask decoder (encoder weights are frozen)
        for _, param in self.image_encoder.named_parameters():
            param.requires_grad_(False)  
            
        for _, param in self.prompt_encoder.named_parameters():
            param.requires_grad_(False)  
            