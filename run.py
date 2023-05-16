
import subprocess
import argparse
import os

parser = argparse.ArgumentParser('This script will run the complete text to video generation pipeline for deepfake '
                                 'videos')
parser.add_argument('--text', '-t', required=True,
                    help='Input text to generate the audio')
parser.add_argument('--video', '-v', required=True,
                    help='Path to the short video clip that will be the driving video for generated final video')
parser.add_argument('--output', '-o', required=True,
                    help='Path to the output file with .mp4 extension')

args = parser.parse_args()

if os.path.exists('temp_phrase.txt'):
    os.remove('temp_phrase.txt')
if os.path.exists('audio_01.wav'):
    os.remove('audio_01.wav')

with open('temp_phrase.txt', 'w') as file:
    file.writelines(args.text)
tts_command = ["python", "inference-tts.py",
               "--tacotron2", "checkpoints/tacotron2",
               "--waveglow", "checkpoints/waveglow",
               "--wn-channels", "256",
               "--cpu",
               "--suffix", "1",
               "-i", "temp_phrase.txt",
               "-o", "./"]
shell_process = subprocess.call(tts_command)
os.remove('temp_phrase.txt')
assert shell_process == 0, "Failed while generating audio"


lipgan_command = ["python", "inference.py",
                  "--face", args.video,
                  "--audio", "audio_01.wav"]
shell_process = subprocess.call(lipgan_command)
os.remove('audio_01.wav')
assert shell_process == 0, "Failed while generating video"

os.replace('results/result_voice.mp4', args.output)

print("Generated Video Saved at:", args.output)
print("Finish")









