from flask import Flask, request, send_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import os
from io import BytesIO
from PIL import Image
import settings

app = Flask(__name__)

# setup pipes
pipe = StableDiffusionPipeline.from_pretrained(settings.model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

if settings.use_xformers:
    pipe.enable_xformers_memory_efficient_attention()

if settings.compile_unet:
    pipe.unet = torch.compile(pipe.unet)

if settings.use_2pass:
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(settings.model_id, torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to("cuda")
    if settings.use_xformers:
        pipe_img2img.enable_xformers_memory_efficient_attention()

    if settings.compile_unet:
        pipe_img2img.unet = torch.compile(pipe_img2img.unet)


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')

    if not prompt:
        return "No prompt provided.", 400
    else:
        print(prompt)
    
    #  first img pass:
    if settings.prev_init_img_strength >= 0.0 and os.path.exists("result.jpg"):
        print("Using init image from previous run.")
        init = Image.open("result.jpg").resize(settings.first_stage_res)
        image = pipe_img2img(
            prompt,
            image=init,
            strength=1.0 - settings.prev_init_img_strength,
            num_inference_steps=settings.steps,
            negative_prompt=settings.neg_prompt,
        ).images[0]
    else:
        image = pipe(
            prompt,
            width=settings.first_stage_res[0],
            height=settings.first_stage_res[1],
            num_inference_steps=settings.steps,
            negative_prompt=settings.neg_prompt,
        ).images[0]

    # 2pass for more detail:
    if settings.use_2pass:
        image = image.resize(settings.second_stage_res)
        image = pipe_img2img(
            prompt,
            image=image,
            strength=1.0-settings.upscale_init_img_strength,
            num_inference_steps=settings.steps,
            negative_prompt=settings.neg_prompt,
        ).images[0]

    img_io = BytesIO()
    image.save(img_io, format='JPEG', quality=95)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()