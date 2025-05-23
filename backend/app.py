from fastapi import FastAPI, HTTPException, Request # type: ignore
from pydantic import BaseModel # type: ignore
import requests
import cloudinary # type: ignore
import cloudinary.uploader # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import io
from PIL import Image
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
from model_loader import ModelLoader
import time
from prometheus_client import Counter, Histogram, start_http_server

start_http_server(4000)

predictions_total = Counter('mlflow_predictions_total', 'Total ML predictions')
prediction_time = Histogram('mlflow_prediction_seconds', 'Time spent on predictions')
model_load_time = Histogram('mlflow_model_load_seconds', 'Time spent loading models')

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

model = ModelLoader()


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

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


@app.on_event("startup")
async def startup_event():
    with model_load_time.time():
        model.load_model()

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
    predictions_total.inc()
    image1_url = urls.image1_url
    image2_url = urls.image2_url
    alpha = urls.alpha

    logger.info(f"Received request to process images. Content: {image1_url}, Style: {image2_url}, Alpha: {alpha}")

    # ---  Download Images from URLs ---
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


    try:
        start_time = time.time()
        with prediction_time.time():
            processed_img_pil = model.apply_style_transfer(image1_content, image2_content)
        process_time = time.time() - start_time
        logger.info(f"Style transfer completed successfully in {process_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Style transfer failed: {str(e)}")

    # ---  Save Processed Pillow Image to BytesIO Buffer ---
    try:
        logger.info("Saving Pillow image to BytesIO buffer...")
        buf = io.BytesIO()
        processed_img_pil.save(buf, format='JPEG')
        buf.seek(0)
        logger.info("Image saved to BytesIO buffer.")

    except Exception as e:
        logger.error(f"Failed to save processed image to BytesIO: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare the final image for upload. Error: {e}")


    # ---  Upload Processed Image from BytesIO to Cloudinary ---
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


    # ---  Return Processed Image URL to Frontend ---
    logger.info("Successfully processed request. Returning result URL.")
    return {
        "processed_image_url": processed_image_url,
        "message": "Images processed and styled successfully!"
    }

# --- Root Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Style Transfer API is running. Use /process-images-from-urls/ to process images."}
