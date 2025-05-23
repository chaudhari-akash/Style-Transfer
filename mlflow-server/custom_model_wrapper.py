import mlflow.pyfunc
import os
import torch
import json
from PIL import Image
from model import Model, VGGEncoder, RC, Decoder
from model import calc_mean_std, adain
from torchvision.transforms.functional import to_pil_image # type: ignore
import numpy as np
import io
import logging 
import base64
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomStyleTransferWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CustomStyleTransferWrapper initialized. Using device: {self.device}")
        
    def load_context(self, context):
        
        logger.info(f"DEBUG: In load_context, sys.path: {sys.path}")

        model_state_path = context.artifacts["model_state"]
        logger.info(f"DEBUG: In load_context, received model_state_path: {model_state_path}")
        logger.info(f"DEBUG: Current working directory: {os.getcwd()}")
        
        if not os.path.exists(model_state_path):
            logger.error(f"ERROR: Model state file does not exist at: {model_state_path}")
            parent_dir = os.path.dirname(model_state_path)
            if os.path.exists(parent_dir):
                logger.error(f"DEBUG: Contents of parent directory {parent_dir}: {os.listdir(parent_dir)}")
            else:
                logger.error(f"ERROR: Parent directory {parent_dir} does not exist.")
            raise FileNotFoundError(f"Model state file not found at {model_state_path}")
        
        if not os.path.isfile(model_state_path):
            logger.error(f"ERROR: Model state path is not a file: {model_state_path}")
            raise IsADirectoryError(f"Expected a file but found a directory at {model_state_path}")
        
        logger.info(f"Loading model state from: {model_state_path}")
        try:
            self.model = Model()
        except Exception as e:
            logger.error(f"ERROR: Failed to instantiate Model class. Check 'model.py' and its dependencies: {e}")
            raise

        try:
            self.model.load_state_dict(torch.load(model_state_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("DEBUG: Model state loaded and moved to device successfully in load_context.")
        except Exception as e:
            logger.error(f"ERROR: Failed to load model state from {model_state_path}: {e}")
            raise

        logger.info("Model loaded successfully.")
            
    def predict(self, context, model_input):
    
        if self.model is None:
            raise RuntimeError("Model not loaded. load_context must be called first.")
            
        if not isinstance(model_input, dict):
            raise TypeError("model_input must be a dictionary with 'content_image' and 'style_image'.")

        content_image_data = model_input.get("content_image")
        style_image_data = model_input.get("style_image")

        if not content_image_data:
            raise ValueError("content_image is missing from model_input.")
        if not style_image_data:
            raise ValueError("style_image is missing from model_input.")

        if isinstance(content_image_data, str):
            try:
                content_image = Image.open(io.BytesIO(base64.b64decode(content_image_data))).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to decode content_image (base64 string): {e}")
        elif isinstance(content_image_data, (bytes, bytearray)):
            content_image = Image.open(io.BytesIO(content_image_data)).convert("RGB")
        else:
            raise TypeError("content_image must be a base64 string or raw bytes.")
        
        if isinstance(style_image_data, str):
            try:
                style_image = Image.open(io.BytesIO(base64.b64decode(style_image_data))).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to decode style_image (base64 string): {e}")
        elif isinstance(style_image_data, (bytes, bytearray)):
            style_image = Image.open(io.BytesIO(style_image_data)).convert("RGB")
        else:
            raise TypeError("style_image must be a base64 string or raw bytes.")

        logger.info("Converting images to tensors.")
        content_tensor = self.pillow_to_tensor(content_image, self.device)
        style_tensor = self.pillow_to_tensor(style_image, self.device)

        logger.info("Generating styled image.")
        styled_tensor_normalized = self.generate_styled_image_from_tensors(
            content_tensor, style_tensor, self.model
        )

        logger.info("Denormalizing styled image.")
        styled_tensor_denorm = self.denorm(styled_tensor_normalized, self.device)

        logger.info("Converting tensor to PIL Image.")
        processed_img_pil = self.tensor_to_pil(styled_tensor_denorm)

        return processed_img_pil

        
    def pillow_to_tensor(self,pil_image: Image.Image, device: torch.device) -> torch.Tensor:
        if pil_image.mode != 'RGB':
            logger.warning(f"Image mode was {pil_image.mode}, converting to RGB.")
            pil_image = pil_image.convert('RGB')

        img_array = np.array(pil_image, dtype=np.float32)
        img_array = img_array / 255.0 
        img_array = img_array.transpose((2, 0, 1))

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
        img_array = (img_array - mean) / std

        tensor = torch.from_numpy(img_array).unsqueeze(0)
        return tensor.to(device)

    def denorm(self,tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
        res = torch.clamp(tensor * std + mean, 0, 1)
        return res

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.squeeze(0).cpu()
        img = to_pil_image(tensor)
        return img

    def generate_styled_image_from_tensors(
        self,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        model: Model,
        alpha: float = 1.0
    ) -> torch.Tensor:
        with torch.inference_mode():
            stylized_tensor = model.generate(content_tensor, style_tensor, alpha)
        return stylized_tensor

