from flask import Flask, render_template, request, redirect, url_for, flash, session # type: ignore
import requests
import os
import cloudinary # type: ignore
import cloudinary.uploader  # type: ignore
import json
from werkzeug.utils import secure_filename # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_replace_with_a_strong_random_string'

# --- Cloudinary Configuration for Flask ---
cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

if not all([cloud_name, api_key, api_secret]):
    print("WARNING: Cloudinary credentials not fully set in Flask .env!")


cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret,
    secure=True
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

BACKEND_SERVICE_URL = os.environ.get('BACKEND_URL', 'http://localhost:5000')
BACKEND_PROCESS_URL = f"{BACKEND_SERVICE_URL}/process-images-from-urls/"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    session.pop('processed_image_url', None)
    session.pop('message', None)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        flash('Both images are required.')
        return redirect(url_for('index'))

    image1_file = request.files['image1']
    image2_file = request.files['image2']

    if image1_file.filename == '' or image2_file.filename == '':
        flash('Both image files must be selected.')
        return redirect(url_for('index'))

    if not (allowed_file(image1_file.filename) and allowed_file(image2_file.filename)):
        flash('Invalid file format. Please use png, jpg, jpeg, or gif.')
        return redirect(url_for('index'))

    image1_url = None
    image2_url = None

    try:
        print(f"DEBUG: Attempting to POST to URL: {BACKEND_PROCESS_URL}")
        upload_result_1 = cloudinary.uploader.upload(
            image1_file.stream,
            resource_type="image",
            folder="uploaded_from_flask", 
        )
        image1_url = upload_result_1.get("secure_url")

        if not image1_url:
             raise Exception("Cloudinary upload for image 1 failed or returned no URL.")

        upload_result_2 = cloudinary.uploader.upload(
            image2_file.stream,
            resource_type="image",
            folder="uploaded_from_flask",
        )
        image2_url = upload_result_2.get("secure_url")

        if not image2_url:
             raise Exception("Cloudinary upload for image 2 failed or returned no URL.")

    except Exception as e:
        flash(f'Error uploading images to Cloudinary: {e}')
        return redirect(url_for('index'))

    backend_data = {
        'image1_url': image1_url,
        'image2_url': image2_url
    }

    try:
        print(f"DEBUG: Attempting to POST to URL: {BACKEND_PROCESS_URL}")
        response = requests.post(
            BACKEND_PROCESS_URL,
            json=backend_data,
            timeout=60
        )

        if response.status_code == 200:
            processed_data = response.json()
            processed_image_url = processed_data.get('processed_image_url')
            message = processed_data.get('message', 'Processing complete.')

            if not processed_image_url:
                 flash('Backend processed successfully, but no processed image URL was returned.')
                 return redirect(url_for('index'))

            session['processed_image_url'] = processed_image_url
            session['message'] = message

            return redirect(url_for('result'))

        else:
            try:
                error_data = response.json()
                error_message = error_data.get('detail')
                if isinstance(error_message, list):
                    error_message = json.dumps(error_message)
                elif not error_message:
                    error_message = error_data.get('message', f'Backend returned status {response.status_code}')

            except requests.exceptions.JSONDecodeError:
                error_message = f'Backend returned status {response.status_code} with non-JSON response.'
            except Exception as e:
                 error_message = f'An unexpected error occurred while parsing backend error: {e}'
                 
            flash(f'Error from backend: {error_message}')
            return redirect(url_for('index'))

    except requests.exceptions.RequestException as e:
        flash(f'Error communicating with backend: {e}')
        return redirect(url_for('index'))


@app.route('/result')
def result():
    processed_image_url = session.pop('processed_image_url', None)
    message = session.pop('message', 'Images processed successfully')
    
    if processed_image_url is None:
        flash('Could not retrieve processed image result URL. Please try again.')
        return redirect(url_for('index'))

    return render_template('result.html',
                           processed_image_url=processed_image_url,
                           message=message)