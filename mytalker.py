from flask import Flask, request, render_template, send_from_directory,jsonify
from werkzeug.utils import secure_filename
import os
from src.gradio_demo import SadTalker
from huggingface_hub import snapshot_download
from upload import upload_to_do
import subprocess
from multiprocessing import Process, SimpleQueue, set_start_method,get_context
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
video_generation_in_progress = False
split_model="htdemucs"
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

def download_model():
    REPO_ID = 'vinthony/SadTalker-V002rc'
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', local_dir_use_symlinks=True)

download_model()  # Download model once when server starts

@app.route('/generate_video', methods=['POST'])
def generate_video():
    global video_generation_in_progress
    try:
        if video_generation_in_progress:
            return jsonify(message="Video generation is already in progress. Please try again later."), 429

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
            print(source_image_path)
            
            audio_path_filename_without_ext=os.path.splitext(audio_path_filename)[0]


            ref_video_file = 'examples/ref_video/1.mp4'
            # if ref_video_path and ref_video_path.filename != '':
                # ref_video_filename = secure_filename(ref_video_path.filename)
                # ref_video_file = os.path.join(app.config['UPLOAD_FOLDER'], ref_video_filename)
                # ref_video_path.save(ref_video_file)

            # Process the files
            print("cut audio ")
            cut_vocal_and_inst(audio_path_file)
            
            vocal_path = f"output/{split_model}/{audio_path_filename_without_ext}/vocals.wav"
            inst = f"output/{split_model}/{audio_path_filename_without_ext}/no_vocals.wav"
            
            print(inst)
            print(vocal_path)
            
            print("before calling video gen")
            video_generation_in_progress=True
            
            
            
            generated_video_path = process_files(source_image_path, vocal_path, ref_video_file)
            print("after video gen")
            
            print("combine the video with instrument")
            output_video_path=combine_video_and_audio(generated_video_path,audio_path_file,audio_path_filename_without_ext)
            #def combine_video_and_audio(video_path, audio_path, output_path):
            print("combine the video with instrument finished")
            #renamed_video_path=rename_video_to_audio_filename(generated_video_path,audio_path)
            upload_to_do(output_video_path)
            #upload_to_do(generated_video_path)
            print("uploaded to Digital Ocean space")
            video_generation_in_progress = False
            return jsonify(message="Files processed and uploaded successfully"), 200
    except Exception as e:
        print(f"An error occurred: str{e}")
        return jsonify(error=str(e)), 500

def combine_video_and_audio(video_path, audio_path, output_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path).without_audio()
    
    # Load the audio clip
    audio_clip = AudioFileClip(audio_path)
    
    # Set the audio of the video clip as the audio clip
    final_clip = video_clip.set_audio(audio_clip)
    
    # Ensure the output file has the correct .mp4 extension
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    output_path1 = f"output/result/{output_path}"
    # Write the result to a file with the correct codecs
    final_clip.write_videofile(output_path1, codec='libx264', audio_codec='aac')

    
    return output_path1
        
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
33

def cut_vocal_and_inst(audio_path):
    
    
    os.makedirs("output/result", exist_ok=True)
    
    print("before executing splitter")
    command = f"demucs --two-stems=vocals -n {split_model} {audio_path} -o output"
    env = os.environ.copy()

# Add or modify the environment variable for this subprocess
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
   
    
    #result = subprocess.Popen(command.split(), stdout=subprocess.PIPE, text=True)
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Demucs process failed:", result.stderr)
    else:
        print("Demucs process completed successfully.")
    print("after executing splitter")
    #for line in result.stdout:
    #    logs.append(line)
    #    yield "\n".join(logs), None, None
    
    print(result.stdout)
    
    file_name=os.path.splitext(os.path.basename(audio_path))[0]


    vocal = f"output/{split_model}/{file_name}/vocals.wav"
    inst = f"output/{split_model}/{file_name}/no_vocals.wav"
    #logs.append("Audio splitting complete.")


# def combine_video_and_audio(video_path, audio_path, output_path):
    # os.makedirs("output/result", exist_ok=True)
    # output_path = f"output/result/{output_path}.mp4"
    # command = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{output_path}"'
    
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # return output_path










@app.route('/results/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
