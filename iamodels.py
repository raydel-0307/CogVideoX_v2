import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import os
import pickle
from minio_db import *
from metrics import get_time
import time
import shutil

def TrainModel(ruta,model_name):

	init_time = time.perf_counter()
	
	save_path = f"{ruta}/model"
	
	if not os.path.exists(save_path):os.mkdir(save_path)
	
	pipe = CogVideoXPipeline.from_pretrained(
		model_name,
		cache_dir=save_path,
		torch_dtype=torch.float16
	)

	model_path = f'{ruta}/model.pkl'

	with open(model_path, 'wb') as f:
		pickle.dump(pipe, f)

	shutil.rmtree(save_path)

	print(f"Modelo Exportdo: {ruta}/")

	get_time(init_time)

def MainModel(ruta,prompt,settings,model_name):

	init_time = time.perf_counter()

	if not os.path.exists(f"{ruta}/model.pkl"):
		print("Descargue el modelo primeramente: 'python3 download_model.py'")
		return

	with open(f"{ruta}/model.pkl", 'rb') as f:
		pipe = pickle.load(f)

	if settings["slow_memory"]:
		pipe.enable_model_cpu_offload()
		pipe.vae.enable_tiling()

	print("Generando Video")

	try:
		video = pipe(
		    prompt=prompt,
		    num_videos_per_prompt=settings["num_videos_per_prompt"],
		    num_inference_steps=settings["num_inference_steps"],
		    num_frames=settings["num_frames"],
		    guidance_scale=settings["guidance_scale"],
		    generator=torch.Generator(device="cuda").manual_seed(42),
		).frames[0]
	except:
		print("No se pudo generar el video, por favor, cambie la configuracion")
		return
	
	export_to_video(video, f"{ruta}/output.mp4", fps=settings["fps"])

	print(f"Video Exportado {ruta}/")

	get_time(init_time)

