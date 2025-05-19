from fastapi import FastAPI, HTTPException, Request # type: ignore
from pydantic import BaseModel # type: ignore
import requests
import cloudinary # type: ignore
import cloudinary.uploader # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import io
from PIL import Image
from model import Model, VGGEncoder, RC
import os
import torch # type: ignore
from torchvision.utils import save_image  # type: ignore
from torchvision.transforms.functional import to_pil_image # type: ignore
import numpy as np
from dotenv import load_dotenv # type: ignore
from fastapi.exceptions import RequestValidationError # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY # type: ignore
import logging 




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

if not all([cloud_name, api_key, api_secret]):
    logger.error("Cloudinary credentials not found in environment variables.")
else:
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True
    )
    logger.info("✅ Cloudinary configured successfully.")


# --- Load Model ---

model = Model() 
# model_state_dict_path = 'models/model_state.pth'
model_state_dict_path = os.environ.get("MODEL_PATH")

try:
    if not os.path.exists(model_state_dict_path):
         raise FileNotFoundError(f"Model state dictionary not found at {model_state_dict_path}")
    model.load_state_dict(torch.load(model_state_dict_path, map_location=lambda storage, loc: storage, weights_only=True))
    logger.info(f"✅ Model state dictionary loaded successfully from {model_state_dict_path}.")
except FileNotFoundError as e:
    logger.error(f"❌ Failed to load Model State Dictionary: {e}")
except Exception as e:
    logger.error(f"❌ Failed to load Model State Dictionary: {e}")
    
    



# --- Setup device ---

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
logger.info(f"Using device: {device}")
model.eval()

# --- Image Preprocessing & Conversion Functions ---

def pillow_to_tensor(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
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

def denorm(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).cpu()
    img = to_pil_image(tensor)
    return img


# --- Generate Styled image (using tensors) ---

def generate_styled_image_from_tensors(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    model: Model,
    alpha: float = 1.0
) -> torch.Tensor:
    with torch.inference_mode():
        stylized_tensor = model.generate(content_tensor, style_tensor, alpha)
    return stylized_tensor


# --- FastAPI App Setup ---

app = FastAPI(
    title="Style Transfer API",
    description="API for performing neural style transfer on images.",
    version="1.0.0",
)

# --- CORS Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Model for Request Body ---

class ImageUrls(BaseModel):
    image1_url: str 
    image2_url: str 
    alpha: float = 1.0


# --- Exception Handler for Pydantic Validation Errors ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors for cleaner responses."""
    logger.error(f"Validation error: {exc.errors()} for request {request.url}")
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "message": "Validation error processing request body. Please check the format of image URLs."
        },
    )

# --- Health Check Endpoint ---

@app.get("/health")

async def health_check():
    status = {"status": "ok", "device": str(device)}
    if not all([cloud_name, api_key, api_secret]):
         status["cloudinary_config"] = "missing"
    else:
         status["cloudinary_config"] = "configured"

    try:
        test_tensor = torch.randn(1, 3, 64, 64).to(device)
        with torch.inference_mode():
            _ = model.generate(test_tensor, test_tensor, 0.5)
        status["model_status"] = "loaded and runnable"
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        status["model_status"] = f"loaded but error during test run: {e}"
    logger.info("Health check performed.")
    return status


# --- Main Endpoint to Process Images ---

@app.post("/process-images-from-urls/")

async def process_images_from_urls(urls: ImageUrls):
    image1_url = urls.image1_url
    image2_url = urls.image2_url
    alpha = urls.alpha

    logger.info(f"Received request to process images. Content: {image1_url}, Style: {image2_url}, Alpha: {alpha}")

    # --- 1. Download Images from URLs ---
    try:
        logger.info(f"Downloading content image from {image1_url}...")
        response1 = requests.get(image1_url, timeout=20)
        response1.raise_for_status()
        image1_content = response1.content
        logger.info("Content image downloaded successfully.")

        logger.info(f"Downloading style image from {image2_url}...")
        response2 = requests.get(image2_url, timeout=20)
        response2.raise_for_status()
        image2_content = response2.content
        logger.info("Style image downloaded successfully.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download images: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download images from URLs. Please check the URLs. Error: {e}")
    except Exception as e:
         logger.error(f"An unexpected error occurred during image download: {e}")
         raise HTTPException(status_code=500, detail=f"An unexpected error occurred during image download: {e}")


    # --- 2. Open Images using Pillow and Convert to RGB ---
    try:
        img1_pil = Image.open(io.BytesIO(image1_content)).convert("RGB")
        img2_pil = Image.open(io.BytesIO(image2_content)).convert("RGB")
        logger.info("Images opened and converted to RGB using Pillow.")

    except Exception as e:
        logger.error(f"Failed to open or convert image files with Pillow: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to open or process image files. Ensure they are valid image formats. Error: {e}")


    # --- 3. Convert Pillow Images to Normalized Tensors ---
    try:
        logger.info("Converting Pillow images to PyTorch tensors...")
        content_tensor = pillow_to_tensor(img1_pil, device)
        style_tensor = pillow_to_tensor(img2_pil, device)
        logger.info("Images converted to tensors successfully.")
    except Exception as e:
        logger.error(f"Failed to convert images to tensors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare images for the model. Error: {e}")


    # --- 4. Generate Styled image Tensor ---
    try:
        logger.info("Generating styled image tensor using the model...")
        styled_tensor_normalized = generate_styled_image_from_tensors(
            content_tensor, style_tensor, model, alpha=alpha
        )
        logger.info("Styled tensor generated successfully.")
        styled_tensor_denorm = denorm(styled_tensor_normalized, device)
        logger.info("Styled tensor denormalized.")

    except Exception as e:
        logger.error(f"Failed during style transfer model execution: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the style transfer process. Error: {e}")

    # --- 5. Convert Styled Tensor back to Pillow Image ---
    try:
        logger.info("Converting denormalized tensor back to Pillow image...")
        processed_img_pil = tensor_to_pil(styled_tensor_denorm)
        logger.info("Tensor converted back to Pillow image.")

    except Exception as e:
        logger.error(f"Failed to convert tensor back to Pillow image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to finalize the processed image. Error: {e}")


    # --- 6. Save Processed Pillow Image to BytesIO Buffer ---
    try:
        logger.info("Saving Pillow image to BytesIO buffer...")
        buf = io.BytesIO()
        processed_img_pil.save(buf, format='JPEG')
        buf.seek(0)
        logger.info("Image saved to BytesIO buffer.")

    except Exception as e:
        logger.error(f"Failed to save processed image to BytesIO: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare the final image for upload. Error: {e}")


    # --- 7. Upload Processed Image from BytesIO to Cloudinary ---
    if not all([cloud_name, api_key, api_secret]):
         logger.error("Cloudinary credentials not configured. Cannot upload.")
         raise HTTPException(status_code=500, detail="Cloudinary is not configured. Cannot upload result.")

    try:
        logger.info("Uploading image to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            buf,
            resource_type="image",
            folder="processed_results",
            format="jpg"
        )
        processed_image_url = upload_result.get("secure_url")

        if not processed_image_url:
             logger.error("Cloudinary upload response missing secure_url.")
             logger.error(f"Cloudinary upload result: {upload_result}")
             raise Exception("Cloudinary upload response did not contain a secure_url.")

        logger.info(f"Image uploaded to Cloudinary: {processed_image_url}")

    except Exception as e:
        logger.error(f"Failed to upload processed image to Cloudinary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload the result to Cloudinary. Error: {e}")


    # --- 8. Return Processed Image URL to Frontend ---
    logger.info("Successfully processed request. Returning result URL.")
    return {
        "processed_image_url": processed_image_url,
        "message": "Images processed and styled successfully!"
    }

# --- Root Endpoint ---
@app.get("/")

async def read_root():
    return {"message": "Style Transfer API is running. Use /process-images-from-urls/ to process images."}
