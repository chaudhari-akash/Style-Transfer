import mlflow
import mlflow.pyfunc
import argparse
import tempfile
import os
from datetime import datetime
import requests
from custom_model_wrapper import CustomStyleTransferWrapper

def download_file(repo_id, filename, token=None):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    print(f"Downloading {filename} from {url}")
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download {filename}, status code: {response.status_code}")
        return None


def download_model_files(repo_id, output_dir, token=None):
    files_to_download = [
        "model_state.pth"
    ]

    os.makedirs(output_dir, exist_ok=True)
    
    for filename in files_to_download:
        content = download_file(repo_id, filename, token)
        if content:
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as f:
                f.write(content)
            print(f"Saved {filename} to {file_path}")
    
    downloaded_files = os.listdir(output_dir)
    with open(os.path.join(output_dir, "files.txt"), "w") as f:
        f.write("\n".join(downloaded_files))
    
    return downloaded_files

def import_custom_model(repo_id, model_name, version=1,description=None, token=None):
    
    print(f"Importing custom model {repo_id} as {model_name}...")
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-tracking-server:7000")
    print(f"Using MLflow tracking URI: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    artifact_location = "file:///mlflow/artifacts"
    print(f"Setting artifact location: {artifact_location}")
    
    experiment_name = "NST"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating new experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(
            experiment_name, 
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Downloading model to {tmp_dir}...")
        
        try:
            downloaded_files = download_model_files(repo_id, tmp_dir, token)
            print(f"Downloaded {len(downloaded_files)} files")
            
            if not downloaded_files:
                raise Exception("No model files were downloaded")
            
            print("Starting MLflow run...")
            
            for filename in downloaded_files:
                file_path = os.path.join(tmp_dir, filename)
                print(f"Raw model file path: {file_path}")
                
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"import_{repo_id.replace('/', '_')}"):
                run_id = mlflow.active_run().info.run_id
                print(f"Started run with ID: {run_id}")
                
                mlflow.log_param("model_id", repo_id)
                mlflow.log_param("model_type", "custom_neural_style_transfer")
                mlflow.log_param("import_date", datetime.now().isoformat())
                mlflow.log_param("files", ", ".join(downloaded_files))
                
                if description:
                    mlflow.set_tag("description", description)
                
                print("Logging model to MLflow...")
                
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=CustomStyleTransferWrapper(repo_id),
                    artifacts={"model_state": os.path.join(tmp_dir, "model_state.pth")},
                    code_path=["custom_model_wrapper.py", "model.py"],
                    pip_requirements=[
                        "torch>=2.0.0",
                        "Pillow>=9.0.0",
                        "torchvision>=0.15.0"
                    ],
                )
                
                print(f"Model logged successfully. Registering with name: {model_name}")
                
                registered_model = mlflow.register_model(
                    model_uri=model_info.model_uri,
                    name=model_name
                )
                
                artifact_uri = mlflow.get_artifact_uri()
                print(f"Artifact URI: {artifact_uri}")
                print(f"Model URI: {model_info.model_uri}")
                print(f"Model registered as {model_name} version {registered_model.version}")
                print(f'Model stage {registered_model.current_stage}')
                
                return registered_model.version
                
        except Exception as e:
            print(f"Error importing model: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import a custom model from Hugging Face to MLflow")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID (e.g., username/model-name)")
    parser.add_argument("--model_name", required=True, help="Name for MLflow registry")
    parser.add_argument("--version", help="Model Version")
    parser.add_argument("--description", help="Description of the model")
    parser.add_argument("--token", help="Hugging Face API token (if needed for private repos)")
    
    args = parser.parse_args()
    import_custom_model(args.repo_id, args.model_name, args.version, args.description, args.token)