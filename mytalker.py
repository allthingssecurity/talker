from flask import Flask, request, render_template, send_from_directory,jsonify
from werkzeug.utils import secure_filename
import os
from src.gradio_demo import SadTalker
from huggingface_hub import snapshot_download
from upload import upload_to_do

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
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', local_dir_use_symlinks=True)

download_model()  # Download model once when server starts

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
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
            print(f"source file={source_image_filename}")
            audio_path_filename = secure_filename(audio_path.filename)
            print(f"audio file={audio_path_filename}")
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
            print("before calling video gen")
            generated_video_path = process_files(source_image_path, audio_path_file, ref_video_file)
            print("after video gen")
            renamed_video_path=rename_video_to_audio_filename(generated_video_path,audio_path)
            upload_to_do(renamed_video_path)
            
            print("uploaded to Digital Ocean space")
            return jsonify(message="Files processed and uploaded successfully"), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify(error=str(e)), 500


def rename_video_to_audio_filename(generated_video_path, audio_path):
    # Extract just the filenames
    video_filename = os.path.basename(generated_video_path)
    audio_filename = os.path.basename(audio_path)

    # Get the file extension of the video file
    video_extension = os.path.splitext(video_filename)[1]

    # Construct the new video filename using the audio filename (but keep its original extension)
    new_video_filename = os.path.splitext(audio_filename)[0] + video_extension

    # Construct the full new path for the video if necessary
    # Here we assume the video is in the same directory as the generated_video_path
    new_video_path = os.path.join(os.path.dirname(generated_video_path), new_video_filename)

    # Rename the video file
    os.rename(generated_video_path, new_video_path)

    return new_video_path
        
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

    generated_video_path = sad_talker.test(source_image=source_image_path,
                                               driven_audio=audio_path,
                                               ref_video=ref_video_path,
                                               **params)
    
    
    return generated_video_path

@app.route('/results/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
