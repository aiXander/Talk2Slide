
# Prompt engineering settings:
transcribe_every_n_seconds = 7
section_length = 350     # only send the last 350 characters of the transcript to chatgpt
prompt_mode = "chat_gpt" # ['chatgpt', 'moving_buffer', 'last_line']

system_description = '''
You're a conversation assistant that creates descriptions of thought provoking images that are shown alongside ongoing conversations and presentations.
Based on the transcript of the conversation, you create a visual description of an image relevant to the current topic of conversation.
'''

task_description = '''
Create a visual description of an image that is relevant to the conversation transcript that follows. The image description should be specific: dont describe what the image envokes, just describe whats in it.
The description will be used by a generative AI model (StableDiffusion) to generate an image that will be displayed live to the audience during the presentation. At the end of the image description you can optionally add (comma separated) style modifiers to make the generated image look better and more realistic.
The image description should be maximum 60 words long and it should be obvious how the image is related to the conversation.
Here's the transcript of the most recent section of the conversation (pay most attention to the last two sentences since those are the most recent):
'''


# Image Rendering settings:
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
negative_prompt = "oversaturated colors, nude, naked, poorly drawn face, ugly, tiling, out of frame, disfigured, deformed, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
steps = 15

prev_init_img_strength    = -1.0 # Use previous image as init for more consistency (set to -1 to turn off)
upscale_init_img_strength = 0.65
first_stage_res = (1024, 576)
#second_stage_res = (1920, 1080)
second_stage_res = (1728, 960)

use_xformers = True   # highly recommended: faster + smaller memory footprint
use_2pass = True      # use 2 passes (low-res + upscale) to get more detail
compile_unet = False  # recompile the unet for slightly faster inference (takes a full extra minute on the first forward pass)

