import torch
import cv2
import numpy as np
from spandrel import ModelLoader
from spandrel.__helpers.model_descriptor import UnsupportedDtypeError

class RealESRGANRestorer:
    def __init__(self, model_path, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load model using spandrel
        loader = ModelLoader()
        self.model = loader.load_from_file(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization: Convert to half precision if supported
        if self.device.type in ["mps", "cuda"]:
            try:
                self.model.half()
                print("Model converted to half-precision (FP16)")
            except UnsupportedDtypeError:
                print("Model does not support half-precision (FP16). Keeping in FP32.")
            except Exception as e:
                print(f"Warning: Could not convert to half precision: {e}")
        
        self.scale = getattr(self.model, 'scale', 4)

    @torch.no_grad()
    def enhance(self, img, target_size=1024):
        """
        Enhance a single image.
        img: numpy array (BGR)
        target_size: max dimension for the final output
        """
        # Convert BGR to RGB and then to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Check if model is in half precision
        try:
            is_half = next(self.model.model.parameters()).dtype == torch.float16
        except:
            is_half = False
            
        if is_half:
            img_tensor = img_tensor.half()
        
        # Inference
        output = self.model(img_tensor)

        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).float().cpu().clamp(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Optimization: Resize to target size if necessary
        h, w = output_bgr.shape[:2]
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            output_bgr = cv2.resize(output_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return output_bgr
