from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from src.gradio_demo import SadTalker
from huggingface_hub import snapshot_download

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

def download_model():
    REPO_ID = 'vinthony/SadTalker-V002rc'
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', use_auth_token=True)

download_model()  # Download model once when server starts

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file parts
        if 'source_image' not in request.files or 'audio_path' not in request.files:
            return 'Missing files', 400
        source_image = request.files['source_image']
        audio_path = request.files['audio_path']
        ref_video_path = request.files.get('ref_video_path')  # Optional

        if source_image.filename == '' or audio_path.filename == '':
            return 'No selected file', 400

        source_image_filename = secure_filename(source_image.filename)
        audio_path_filename = secure_filename(audio_path.filename)

        source_image_path = os.path.join(app.config['UPLOAD_FOLDER'], source_image_filename)
        audio_path_file = os.path.join(app.config['UPLOAD_FOLDER'], audio_path_filename)

        source_image.save(source_image_path)
        audio_path.save(audio_path_file)

        ref_video_file = None
        if ref_video_path and ref_video_path.filename != '':
            ref_video_filename = secure_filename(ref_video_path.filename)
            ref_video_file = os.path.join(app.config['UPLOAD_FOLDER'], ref_video_filename)
            ref_video_path.save(ref_video_file)

        # Process the files
        generated_video_path = process_files(source_image_path, audio_path_file, ref_video_file)
        
        return render_template('display_video.html', video_file=generated_video_path)
    
    return render_template('upload_form.html')

def process_files(source_image_path, audio_path, ref_video_path=None):
    sad_talker = SadTalker(lazy_load=True)

    params = {
        'preprocess': 'crop',
        'still_mode': True,
        'use_enhancer': False,
        'batch_size': 1,
        'size': 256,
        'pose_style': 0,
        'facerender': 'facevid2vid',
        'exp_scale': 1.0,
        'use_ref_video': ref_video_path is not None,
        'ref_info': 'pose',
        'use_idle_mode': False,
        'length_of_audio': 5,
        'result_dir': app.config['RESULT_FOLDER'],
    }

    generated_video_path = sad_talker.generate(source_image_path=source_image_path,
                                               driven_audio=audio_path,
                                               ref_video=ref_video_path,
                                               **params)

    return generated_video_path

@app.route('/results/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True,host=0.0.0.0,port=7860)
