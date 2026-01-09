import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler

class DiffusionRestorer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", controlnet_id="lllyasviel/control_v11f1e_sd15_tile", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"DiffusionRestorer using device: {self.device}")
        
        # Load ControlNet
        print(f"Loading ControlNet: {controlnet_id}...")
        self.controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32)
        
        # Load Pipeline
        print(f"Loading Pipeline: {model_id}...")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, 
            controlnet=self.controlnet, 
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            safety_checker=None
        )
        
        # Use DDIM scheduler for faster inference
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        
        # Optimization for Mac
        if self.device.type == "mps":
            # Some versions of diffusers/mps benefit from this
            pass

    @torch.no_grad()
    def enhance(self, img, prompt=None, negative_prompt=None, strength=0.5, num_inference_steps=20, guidance_scale=7.5):
        """
        Enhance a single image using Diffusion + ControlNet Tile.
        img: numpy array (BGR)
        """
        if prompt is None:
            prompt = "high quality jiu jitsu athletes grappling, sharp details, gi fabric texture, realistic, anatomical correctness, 8k"
        if negative_prompt is None:
            negative_prompt = "blur, low quality, artifact, distorted limbs, extra fingers, cartoon, drawing, painting"

        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Resize to multiple of 8 (requirement for SD)
        w, h = pil_img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # ControlNet Tile uses the image itself as the control
        # It's an img2img pipeline with ControlNet
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_img,
            control_image=pil_img,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Convert back to numpy BGR
        output_rgb = np.array(output)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        
        return output_bgr

