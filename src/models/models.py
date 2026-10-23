### MODELS.PY

import os
#os.environ['HF_HOME'] = '<PATH_TO_HF_CACHE>'

import torch
from models.blip.blip_model import BlipForImageTextRetrievalVision, BlipForImageTextRetrievalText
from transformers import BlipProcessor    
from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch.nn.functional as F
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel, ClapModel
from PIL import Image
from abc import ABC, abstractmethod
from classifiers.nsfl_base import IImageTextEncoder
import numpy as np
import numpy.typing as npt

class ModelWrapper(IImageTextEncoder):
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.name = model_name

        if 'blip' in model_name:
            # load blip model

            # image_preprocess
            #model_name_1 = "Salesforce/blip-image-captioning-base"
            model_name_1 = 'Salesforce/blip-itm-base-coco'
            if ('large' in model_name):
                model_name_1 = 'Salesforce/blip-itm-large-coco'
            
            self.image_preprocess = BlipProcessor.from_pretrained(model_name_1, use_fast=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_1, trust_remote_code=True)

            self.visual_model = BlipForImageTextRetrievalVision.from_pretrained(model_name_1).eval().half().to(self.device)
            self.text_model = BlipForImageTextRetrievalText.from_pretrained(model_name_1).eval().half().to(self.device)

            
        elif 'clip' in model_name:
            model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name).eval().half().to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        elif 'clap' in model_name:
            model_name = "laion/clap-htsat-unfused"
            self.model = ClapModel.from_pretrained(model_name).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name)
              
        elif 'siglip' in model_name:
            # device = 'cuda'
            if 'siglip2' in model_name:
                model_name_1 = "google/siglip2-so400m-patch14-384"
                self.model = SiglipModel.from_pretrained(model_name_1).to(device)
                self.processor = AutoProcessor.from_pretrained(model_name_1, use_fast=False)
            else:
                model_name_1 = "google/siglip-so400m-patch14-384"
                self.model = SiglipModel.from_pretrained(model_name_1).to(device)
                self.processor = AutoProcessor.from_pretrained(model_name_1, use_fast=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_1)

            self.model = SiglipModel.from_pretrained(model_name_1).to(device)

            
        elif 'e5-v' in model_name:
            # load e5-v model
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.half).to(device)
            self.img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
            self.text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
            self.text_img_prompt = 'Given <image> of <sent>\n, Summary <sent> in one word: '        

        else:
            print("unsupported model: " + model_name)

    
    def encode_images(self, images):
        if 'blip' in self.model_name:
            # Original PyTorch path (CPU / CUDA)
            inputs = self.image_preprocess(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeds = self.visual_model(inputs['pixel_values'].half()).detach().cpu()
            
        elif 'clip' in self.model_name:
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                visual_embs = self.model.get_image_features(**inputs)
            embeds = (visual_embs / visual_embs.norm(dim=-1, keepdim=True)).detach().cpu()
        
        elif 'clap' in self.model_name:
            inputs = self.processor(audio=images, return_tensors="pt", 
                                   padding=True, sampling_rate=48000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                audio_embeddings = self.model.get_audio_features(**inputs)
            embeds = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1).detach().cpu()
            
        elif 'siglip' in self.model_name:
            #images_rgb = [image.convert("RGB") for image in images]
            inputs = {}
            pixel_values = []
            for i, image in enumerate(images):
                im_shape = np.array(image).shape
                print(im_shape)
                print(i)
                if len(im_shape) != 3:
                    image= image.convert('L')
                    image= image.convert('RGB')
                input_1 = self.processor(images=image, return_tensors="pt").to(self.device)
                pixel_values.append(input_1['pixel_values'])
            pixel_values = torch.cat(pixel_values, dim=0)
            inputs['pixel_values'] = pixel_values
            # inputs = [torch.Tensor(self.processor(images=image, return_tensors="pt")).to(self.device) for image in images]
            # inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                visual_embs = self.model.get_image_features(**inputs)
            embeds = (visual_embs / visual_embs.norm(dim=-1, keepdim=True)).detach().cpu()
            
        elif 'e5-v' in self.model_name:
            inputs = self.processor([self.img_prompt]*len(images), images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():                
                embeds = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            embeds = F.normalize(embeds, dim=-1)
        return embeds


    def encode_texts(self, texts):
        if 'blip' in self.model_name:
            # Original PyTorch path (CPU / CUDA)
            tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            # embeds = self.text_model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            with torch.no_grad():
                embeds = self.text_model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            
        elif 'clip' in self.model_name:
            text_token = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_embs = self.model.get_text_features(**text_token)
            embeds = (text_embs / text_embs.norm(dim=-1, keepdim=True)).detach().cpu()
            
        elif 'clap' in self.model_name:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                text_embeddings = self.model.get_text_features(**inputs) 
            embeds = torch.nn.functional.normalize(text_embeddings, p=2, dim=1).detach().cpu()
            
        elif 'siglip' in self.model_name:
            text_token = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_embs = self.model.get_text_features(**text_token)
            embeds = (text_embs / text_embs.norm(dim=-1, keepdim=True)).detach().cpu()
            # embeds = embeds.cpu().numpy()
            
        elif 'e5-v' in self.model_name:            
            inputs = self.processor([self.text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():                
                embeds = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            embeds = F.normalize(embeds, dim=-1)
        return embeds
    
    
    def encode_text_image(self, texts, image):
        if 'blip' in self.model_name:
            print("not supported")
        elif 'clip' in self.model_name:
            assert("not supported")            
        elif 'e5-v' in self.model_name:            
            input_texts = self.processor([self.text_img_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():                
                embeds = self.processor(input_texts, image, return_tensors="pt", padding=True).to(self.device)        
            embeds = F.normalize(embeds, dim=-1)
        return embeds

