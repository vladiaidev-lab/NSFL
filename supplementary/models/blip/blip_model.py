import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import normalize
from transformers import BlipPreTrainedModel, BlipConfig, BlipVisionModel, BlipTextModel

class BlipForImageTextRetrievalVision(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        # vision projection layer
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self,  
                pixel_values: torch.FloatTensor,        
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                interpolate_pos_encoding: bool = False):
      
      vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
      image_embeds = vision_outputs[0]
      image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
      return image_feat
    
class BlipForImageTextRetrievalText(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # text projection layer
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self,  
                input_ids: torch.LongTensor,        
                attention_mask: Optional[torch.LongTensor] = None,                
                return_dict: Optional[bool] = None):
      
      question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
      question_embeds = question_embeds[0] 
      text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
      return text_feat    
  