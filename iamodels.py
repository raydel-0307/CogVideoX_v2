import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import gc
import os

def TrainModel(ruta,model_name):
	
	save_path = f"{ruta}/model"
	
	if not os.path.exists(save_path):os.mkdir(save_path)
	
	pipe = CogVideoXPipeline.from_pretrained(
		model_name,
		cache_dir=save_path,
		torch_dtype=torch.float16
	)

	gc.collect()
	
	torch.cuda.empty_cache()
	gc.collect()

	print(f"Modelo Exportdo: {ruta}/")

def MainModel(ruta,prompt,settings,model_name):
	
	model_path = f"{ruta}/model"

	pipe = CogVideoXPipeline.from_pretrained(
		model_name,
		cache_dir=model_path,
		torch_dtype=torch.float16
	)

	if settings["slow_memory"]:
		pipe.enable_model_cpu_offload()
		pipe.enable_sequential_cpu_offload()
		pipe.vae.enable_slicing()
		pipe.vae.enable_tiling()

	print("Generando Video")

	video = pipe(
	    prompt=prompt,
	    num_videos_per_prompt=settings["num_videos_per_prompt"],
	    num_inference_steps=settings["num_inference_steps"],
	    num_frames=settings["num_frames"],
	    guidance_scale=settings["guidance_scale"],
	    generator=torch.Generator(device="cuda").manual_seed(42),
	).frames[0]
	
	export_to_video(video, f"{ruta}/output.mp4", fps=settings["fps"])
	
	torch.cuda.empty_cache()

	gc.collect()

	del pipe
	torch.cuda.empty_cache()
	gc.collect()

	print(f"Video Exportado {ruta}/")

