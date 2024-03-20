import argparse
from src.gradio_demo import SadTalker
from huggingface_hub import snapshot_download

def download_model():
    REPO_ID = 'vinthony/SadTalker-V002rc'
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', use_auth_token=True)

def main(source_image_path, audio_path, ref_video_path=None):
    download_model()

    # Initialize SadTalker
    sad_talker = SadTalker(lazy_load=True)

    # Hardcoded parameters as they are in Gradio UI
    params = {
        'preprocess_type': 'crop',
        'is_still_mode': True,
        'enhancer': False,
        'batch_size': 1,
        'size_of_image': 256,
        'pose_style': 0,
        'facerender': 'facevid2vid',
        'exp_weight': 1.0,
        'use_ref_video': ref_video_path is not None,
        'ref_info': 'pose',
        'use_idle_mode': False,
        'length_of_audio': 5,
        'blink_every': True,
    }

    # Assuming SadTalker has a method `generate` to process these inputs
    # Adapt this part according to your actual implementation
    generated_video_path = sad_talker.generate(source_image_path=source_image_path,
                                               audio_path=audio_path,
                                               ref_video_path=ref_video_path,
                                               **params)

    print(f"Generated video saved to: {generated_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate talking face animation")
    parser.add_argument("source_image_path", type=str, help="Path to the source image")
    parser.add_argument("audio_path", type=str, help="Path to the input audio")
    parser.add_argument("--ref_video_path", type=str, default=None, help="Optional path to the reference video")

    args = parser.parse_args()

    main(args.source_image_path, args.audio_path, args.ref_video_path)
