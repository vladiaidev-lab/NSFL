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
        self.use_openvino = (device == torch.device("xpu"))

        if 'blip' in model_name:
            # load blip model

            # image_preprocess
            #model_name_1 = "Salesforce/blip-image-captioning-base"
            model_name_1 = 'Salesforce/blip-itm-base-coco'
            if ('large' in model_name):
                model_name_1 = 'Salesforce/blip-itm-large-coco'
            
            self.image_preprocess = BlipProcessor.from_pretrained(model_name_1, use_fast=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_1, trust_remote_code=True)

            if self.use_openvino:
                self._init_openvino_blip(model_name_1)
            else:
                # Original path for CPU / CUDA
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

            if self.use_openvino:
                self._init_openvino_siglip(model_name_1)
            else:
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

    
    # ------------------------------------------------------------------ #
    #  OpenVINO setup (only called when device == xpu)
    # ------------------------------------------------------------------ #
    def _init_openvino_blip(self, model_name_1):
        import openvino as ov
        import os

        core = ov.Core()
        available = core.available_devices
        self.ov_device = "GPU" if "GPU" in available else "CPU"
        print(f"OpenVINO available devices: {available}, using: {self.ov_device}")

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ov_blip")
        os.makedirs(cache_dir, exist_ok=True)

        safe_name = model_name_1.replace("/", "_")
        visual_xml = os.path.join(cache_dir, f"{safe_name}_visual.xml")
        text_onnx = os.path.join(cache_dir, f"{safe_name}_text.onnx")

        # --- Visual encoder (ov.convert_model works fine) ---
        if os.path.exists(visual_xml):
            print(f"Loading cached visual model from {visual_xml}")
            ov_visual = core.read_model(visual_xml)
        else:
            print("Converting visual model to OpenVINO IR (one-time)...")
            pt_visual = BlipForImageTextRetrievalVision.from_pretrained(model_name_1).eval()
            dummy_image = torch.randn(1, 3, 384, 384)
            ov_visual = ov.convert_model(pt_visual, example_input=dummy_image)
            ov.save_model(ov_visual, visual_xml)
            del pt_visual
            print(f"Visual model saved to {visual_xml}")

        self.ov_visual_model = core.compile_model(ov_visual, self.ov_device)
        self.ov_visual_input = self.ov_visual_model.input(0)
        self.ov_visual_output = self.ov_visual_model.output(0)

        # --- Text encoder (use ONNX export to avoid tracing issues) ---
        if os.path.exists(text_onnx):
            print(f"Loading cached text model from {text_onnx}")
        else:
            print("Exporting text model to ONNX (one-time)...")
            pt_text = BlipForImageTextRetrievalText.from_pretrained(model_name_1).eval()

            dummy_ids = torch.randint(0, 1000, (1, 32))
            dummy_mask = torch.ones(1, 32, dtype=torch.long)

            # Wrap forward to force return_dict=False and unpack the tuple
            class TextWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids, attention_mask):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=False,
                    )
                    # return_dict=False returns a tuple; first element is the embedding
                    if isinstance(out, tuple):
                        return out[0]
                    return out

            wrapped = TextWrapper(pt_text).eval()

            torch.onnx.export(
                wrapped,
                (dummy_ids, dummy_mask),
                text_onnx,
                input_names=["input_ids", "attention_mask"],
                output_names=["embeddings"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "embeddings": {0: "batch"},
                },
                opset_version=14,
            )
            del pt_text, wrapped
            print(f"Text model saved to {text_onnx}")

        ov_text = core.read_model(text_onnx)
        self.ov_text_model = core.compile_model(ov_text, self.ov_device)


    def _encode_images_openvino(self, images):
        inputs = self.image_preprocess(images=images, return_tensors="np")  # numpy for OV
        pixel_values = inputs['pixel_values'].astype(np.float32)

        result = self.ov_visual_model({self.ov_visual_input: pixel_values})
        embeds = torch.tensor(result[self.ov_visual_output])
        return embeds
    

    def _encode_texts_openvino(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="np", padding=True)
        input_ids = tokenized['input_ids'].astype(np.int64)
        attention_mask = tokenized['attention_mask'].astype(np.int64)

        result = self.ov_text_model({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        embeds = torch.tensor(result[self.ov_text_model.output(0)])
        return embeds


    def _init_openvino_siglip(self, model_name_1):
        import openvino as ov
        import os

        core = ov.Core()
        available = core.available_devices
        self.ov_device = "GPU" if "GPU" in available else "CPU"
        print(f"OpenVINO (SigLIP) available devices: {available}, using: {self.ov_device}")

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ov_siglip")
        os.makedirs(cache_dir, exist_ok=True)

        safe_name = model_name_1.replace("/", "_")
        visual_onnx = os.path.join(cache_dir, f"{safe_name}_visual.onnx")
        text_onnx = os.path.join(cache_dir, f"{safe_name}_text.onnx")

        # Load full model temporarily for export
        pt_model = SiglipModel.from_pretrained(model_name_1).eval()

        # --- Visual encoder ---
        if os.path.exists(visual_onnx):
            print(f"Loading cached SigLIP visual model from {visual_onnx}")
        else:
            print("Exporting SigLIP visual model to ONNX (one-time)...")

            class SiglipVisualWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.vision_model = model.vision_model

                def forward(self, pixel_values):
                    outputs = self.vision_model(pixel_values=pixel_values, return_dict=False)
                    # SigLIP vision model returns (last_hidden_state, pooler_output)
                    pooled = outputs[1]
                    return pooled

            wrapped_visual = SiglipVisualWrapper(pt_model).eval()
            dummy_pixel = torch.randn(1, 3, 384, 384)

            torch.onnx.export(
                wrapped_visual,
                dummy_pixel,
                visual_onnx,
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "image_features": {0: "batch"},
                },
                opset_version=14,
            )
            print(f"SigLIP visual model saved to {visual_onnx}")

        # --- Text encoder ---
        if os.path.exists(text_onnx):
            print(f"Loading cached SigLIP text model from {text_onnx}")
        else:
            print("Exporting SigLIP text model to ONNX (one-time)...")

            class SiglipTextWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.text_model = model.text_model

                def forward(self, input_ids, attention_mask):
                    outputs = self.text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=False,
                    )
                    pooled = outputs[1]
                    return pooled

            wrapped_text = SiglipTextWrapper(pt_model).eval()
            dummy_ids = torch.randint(0, 1000, (1, 64))
            dummy_mask = torch.ones(1, 64, dtype=torch.long)

            torch.onnx.export(
                wrapped_text,
                (dummy_ids, dummy_mask),
                text_onnx,
                input_names=["input_ids", "attention_mask"],
                output_names=["text_features"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "text_features": {0: "batch"},
                },
                opset_version=14,
            )
            print(f"SigLIP text model saved to {text_onnx}")

        del pt_model

        # Compile both
        ov_visual = core.read_model(visual_onnx)
        self.ov_visual_model = core.compile_model(ov_visual, self.ov_device)

        ov_text = core.read_model(text_onnx)
        self.ov_text_model = core.compile_model(ov_text, self.ov_device)


    def _encode_images_openvino_siglip(self, images):
        pixel_values = []
        for image in images:
            im_shape = np.array(image).shape
            if len(im_shape) != 3:
                image = image.convert('L').convert('RGB')
            input_1 = self.processor(images=image, return_tensors="np")
            pixel_values.append(input_1['pixel_values'])
        pixel_values = np.concatenate(pixel_values, axis=0).astype(np.float32)

        result = self.ov_visual_model({"pixel_values": pixel_values})
        visual_embs = torch.tensor(result[self.ov_visual_model.output(0)])
        embeds = visual_embs / visual_embs.norm(dim=-1, keepdim=True)
        return embeds
    
    def _encode_texts_openvino_siglip(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="np", padding=True)
        input_ids = tokenized['input_ids'].astype(np.int64)
        
        # SigLIP tokenizer may not return attention_mask
        if 'attention_mask' in tokenized:
            attention_mask = tokenized['attention_mask'].astype(np.int64)
        else:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

        result = self.ov_text_model({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        text_embs = torch.tensor(result[self.ov_text_model.output(0)])
        embeds = text_embs / text_embs.norm(dim=-1, keepdim=True)
        return embeds

        
    def encode_images(self, images):
        if 'blip' in self.model_name:
            if self.use_openvino:
                return self._encode_images_openvino(images)

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
            if self.use_openvino:
                return self._encode_images_openvino_siglip(images)
            
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
            if self.use_openvino:
                return self._encode_texts_openvino(texts)

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
            if self.use_openvino:
                return self._encode_texts_openvino_siglip(texts)
            
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

