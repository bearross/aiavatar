
import sys

import time
from os import listdir, path
import numpy as np
import scipy, cv2, os, argparse
from wav2lip import audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from detectors import face_detection
from wav2lip import Wav2Lip
import platform

from detectors.ultralight_facedetector import UltraLightFaceDetecion

from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, default='wav2lip/checkpoints/wav2lip_gan.pth',
					help='Path to Wav2Lip model saved checkpoint to load weights from')

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

# def face_detect(images):
# 	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
# 											flip_input=False, device=device)
#
# 	batch_size = args.face_det_batch_size
#
# 	while 1:
# 		predictions = []
# 		try:
# 			for i in tqdm(range(0, len(images), batch_size)):
# 				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
# 		except RuntimeError:
# 			if batch_size == 1:
# 				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
# 			batch_size //= 2
# 			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
# 			continue
# 		break
#
# 	results = []
# 	pady1, pady2, padx1, padx2 = args.pads
# 	for rect, image in zip(predictions, images):
# 		if rect is None:
# 			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
# 			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
#
# 		y1 = max(0, rect[1] - pady1)
# 		y2 = min(image.shape[0], rect[3] + pady2)
# 		x1 = max(0, rect[0] - padx1)
# 		x2 = min(image.shape[1], rect[2] + padx2)
#
# 		results.append([x1, y1, x2, y2])
#
# 	boxes = np.array(results)
# 	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
# 	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
#
# 	del detector
# 	return results

def face_detect(images):
	fd = UltraLightFaceDetecion('detectors/ultralight_facedetector/RFB-320.tflite',
								conf_threshold=0.6)
	results = list()
	for img in images:
		boxes, scores = fd.inference(img)
		x1, y1, x2, y2 = boxes[0].round().astype(int)
		half = ((x2 - x1) / 2) + x1
		y_margin = int((y2 - y1)/15)
		x_margin = int((x2 - x1)/15)
		if x1 - x_margin > 0:
			half -= x_margin
		y1 = max(0, y1 - y_margin)
		y2 += y_margin
		x1 = max(0, x1 - x_margin)
		x2 += x_margin
		results.append([img[y1:y2, x1:x2], (y1, y2, x1, x2, half)])
	return results

def resize_with_padding(img, size):
    """
    Resizes an image to a given size while maintaining the aspect ratio.
    If the aspect ratio of the image is not the same as the target aspect ratio,
    the function pads the image with black pixels to achieve the desired aspect ratio.
    Args:
        img (numpy.ndarray): The input image to resize with shape (height, width, channels).
        size (tuple): The desired output size as a tuple of (width, height).
    Returns:
        numpy.ndarray: The resized image with shape (new_height, new_width, channels).
    """
    h, w = img.shape[:2]
    target_w, target_h = size

    # Determine scaling factors to fit the image to the target size while maintaining the aspect ratio
    scale_factor = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)

    # Resize the image with the determined scaling factors
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad the image with black pixels to achieve the desired aspect ratio
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_img

import random
def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	half_points = list()
	random_choice = False

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			if os.path.isdir(args.face):
				face_det_results = face_detect([frames])
			else:
				face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	reverse = False
	reverse_point = np.random.randint(1, len(frames))
	idx = np.random.randint(0, reverse_point)
	for i, m in enumerate(mels):
		if idx == reverse_point:
			reverse = not reverse
			if reverse:
				reverse_point = np.random.randint(0, idx)
			else:
				reverse_point = np.random.randint(idx, len(frames))
		if reverse:
			idx -= 1
		else:
			idx += 1
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()
		half_point = coords[4]
		coords = coords[:4]
		half_point *= args.img_size/face.shape[1]
		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)
		half_points.append(int(round(half_point)))


		if len(img_batch) >= args.wav2lip_batch_size:

			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0
			# img_masked = list()
			# for img, p in zip(img_batch, half_points):
			# 	let_img = img.copy()
			# 	let_img[:, p:] = 0
			# 	img_masked.append(let_img)
			# img_masked = np.asarray(img_masked)

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:

		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0
		# img_masked = list()
		# for img, p in zip(img_batch, half_points):
		# 	let_img = img.copy()
		# 	let_img[:, p:] = 0
		# 	img_masked.append(let_img)
		# img_masked = np.asarray(img_masked)

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def get_face_enhancer(model_name='RealESRGAN_x4plus'):

	cuda_available = torch.cuda.is_available()

	esrgan_dir = '../Real-ESRGAN'

	if model_name == 'RealESRGAN_x4plus':

		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
		netscale = 4
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

	elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
		netscale = 2
		file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']

	model_path = os.path.join(esrgan_dir, 'weights', model_name + '.pth')
	if not os.path.isfile(model_path):
		for url in file_url:
			model_path = load_file_from_url(
				url=url, model_dir=os.path.join(esrgan_dir, 'weights'), progress=True, file_name=None)

	upsampler = RealESRGANer(
		scale=netscale,
		model_path=model_path,
		dni_weight=None, # For realesr-general-x4v3 model only
		model=model,
		tile=0,
		tile_pad=10,
		pre_pad=0,
		half=cuda_available,
		gpu_id=None)

	face_enhancer = GFPGANer(
		model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
		upscale=4,
		arch='clean',
		channel_multiplier=2,
		bg_upsampler=upsampler)

	return face_enhancer

from PIL import Image

def main():

	face_enhancer = get_face_enhancer()

	if not (os.path.isfile(args.face) | os.path.isdir(args.face)):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif os.path.isdir(args.face):
		faces_names = sorted(os.listdir(args.face), key=lambda x: int(x.split('.')[0]))
		full_frames = [cv2.imread(os.path.join(args.face, _)) for _ in faces_names]
		fps = args.fps

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	##### Resize All frames [start] #####
	f = full_frames[0]
	_, c = face_detect([full_frames[0]])[0]
	h, w, _ = f.shape
	y1, y2, x1, x2, _ = c
	cw, ch = x2 - x1, y2 - y1

	ratio = args.img_size / max(y2 - y1, x2 - x1)
	dh, dw = int(round(ratio * (y2 - y1))), int(round(ratio * (x2 - x1)))

	ratio_w, ratio_h = dw / cw, dh / ch
	nw, nh = int(round(ratio_w * w)), int(round(ratio_h * h))

	full_frames = [np.array(Image.fromarray(_).resize((nw, nh))) for _ in full_frames]

	##### Resize All frames [end] #####

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	synced_frames = list()
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):

		# if i == 2:
		# 	break
		if i == 0:
			print()
			model = load_model(args.checkpoint_path)
			print("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c

			pu = p.astype(np.uint8)
			ratio = args.img_size/max(y2-y1, x2-x1)
			h, w = int(round(ratio*(y2-y1))), int(round(ratio*(x2-x1)))
			pu = np.array(Image.fromarray(pu).resize((w, h)))

			# i_initials = i < 2
			# if i_initials:
			# 	t = time.perf_counter()
			# _, _, pu = face_enhancer.enhance(pu, has_aligned=False, only_center_face=False, paste_back=True)
			# if i_initials:
			# 	print('face-enhancer time:', time.perf_counter() - t)

			pu = np.array(Image.fromarray(pu).resize((x2 - x1, y2 - y1)))
			f[y1:y2, x1:x2] = pu
			t = time.perf_counter()
			_, _, fu = face_enhancer.enhance(f.copy(), has_aligned=False, only_center_face=False, paste_back=True)
			print('face-enhancer time:', time.perf_counter() - t)
			try:
				out
			except NameError:
				h, w, _ = fu.shape
				out = cv2.VideoWriter('wav2lip/temp.avi',
									  cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
			out.write(fu)

	out.release()
	os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'wav2lip/temp.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
