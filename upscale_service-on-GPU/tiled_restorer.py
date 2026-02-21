import torch
import cv2
import numpy as np
import math
from spandrel import ModelLoader
from spandrel.__helpers.model_descriptor import UnsupportedDtypeError
from tqdm import tqdm

class TiledRealESRGANRestorer:
    def __init__(self, model_path, device=None, tile_size=400, tile_pad=10, pre_pad=0, half=True):
        """
        Args:
            model_path (str): Path to the model file.
            device (str): 'cuda', 'mps', or 'cpu'. Auto-detects if None.
            tile_size (int): Size of the tile for processing (e.g., 400). 0 to disable tiling.
            tile_pad (int): Padding around the tile to avoid edge artifacts.
            pre_pad (int): Padding for the whole image (rarely needed).
            half (bool): Use half precision (FP16) if possible.
        """
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
        
        # Optimization: Convert to half precision if supported and requested
        self.half = half
        if self.half and self.device.type in ["mps", "cuda"]:
            try:
                self.model.half()
                print("Model converted to half-precision (FP16)")
            except UnsupportedDtypeError:
                print("Model does not support half-precision (FP16). Keeping in FP32.")
                self.half = False
            except Exception as e:
                print(f"Warning: Could not convert to half precision: {e}")
                self.half = False
        else:
            self.half = False
        
        self.scale = getattr(self.model, 'scale', 4)
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad

    @torch.no_grad()
    def enhance(self, img, target_size=None):
        """
        Enhance a single image with tiling support.
        img: numpy array (BGR)
        target_size: Optional max dimension for the final output (resizes result if exceeded).
        """
        # Convert BGR to RGB and then to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        if self.half:
            img_tensor = img_tensor.half()
            
        # Check if we should use tiling
        # If tile_size is 0, disable tiling. 
        # Also disable if image is smaller than tile_size.
        batch, channel, height, width = img_tensor.shape
        use_tiling = self.tile_size > 0 and (height > self.tile_size or width > self.tile_size)
        
        if use_tiling:
            output_tensor = self.tile_process(img_tensor)
        else:
            output_tensor = self.model(img_tensor)

        # Convert back to numpy
        output = output_tensor.squeeze(0).permute(1, 2, 0).float().cpu().clamp(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Resize to target size if necessary
        if target_size is not None:
            h, w = output_bgr.shape[:2]
            if max(h, w) > target_size:
                scale_factor = target_size / max(h, w)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                output_bgr = cv2.resize(output_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return output_bgr

    def tile_process(self, img):
        """
        Process image using tiling to save memory.
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # Start with a zeroed output tensor on the same device
        output = img.new_zeros(output_shape)
        
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # Loop over tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate input tile coordinates
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                
                # Input tile boundaries (without padding)
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)
                
                # Input tile boundaries (with padding)
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)
                
                # Calculate padding offsets relative to the extracted tile
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                
                # Crop input tile
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                
                # Run inference
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                
                # Calculate output boundaries
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                
                # Calculate output padding to crop
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
                
                # Place output tile into result tensor
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
                    
        return output
