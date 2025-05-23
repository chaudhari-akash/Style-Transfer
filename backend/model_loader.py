import os
import logging
import time
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_name = os.environ.get("MODEL_NAME", "nst")
        self.model_version = os.environ.get("MODEL_VERSION", "1")
        self.mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-tracking-server:7000")
        logger.info(f"ModelLoader initialized: {self.model_name} ({self.model_version})")
    
    def load_model(self):
        try:
            start_time = time.time()
            logger.info(f"Loading model {self.model_name} (stage: {self.model_version}) from MLflow")
        
            mlflow.set_tracking_uri(self.mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            
            latest_versions = client.get_latest_versions(self.model_name)
            if not latest_versions:
                raise Exception(f"No versions found for model {self.model_name}")
            
            latest_version = latest_versions[0].version
            logger.info(f"Latest version found: {latest_version}")
            
            self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{latest_version}")
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
        
    
    def apply_style_transfer(self, content_image, style_image=None):
        if self.model is None:
            success = self.load_model()
            if not success:
                raise Exception("Failed to load model")
        
        try:
            start_time = time.time()
            model_input = {"content_image": content_image}
            if style_image is not None:
                model_input["style_image"] = style_image
                
            result = self.model.predict(model_input)
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise