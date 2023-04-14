# Talk2Slide (Whisper --> ChatGPT --> StableDiffusion)

## Main Idea:
1. Use Whisper to auto-transcribe an ongoing conversation / presentation
2. Send the transcript to ChatGPT and ask it to generate a visual prompt relevant to the conversation
3. Generate an image from that prompt using StableDiffusion

## provenance:
- Initial idea of hooking up whisper to StableDiffusion to do live voice prompting was implemented by [@Huemin](https://twitter.com/huemin_art)
- I then added async speech-to-text to enable capturing the full conversation transcript + ChatGPT support to turn that transcript into a relevant prompt

## Setup:
```
conda create -n t2s python=3.10 -y
conda activate t2s
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu117
python -m pip install -r requirements.txt
```

## Usage:
```
conda activate t2s
python diffusers-server-local.py
python whisper-real-time.py
```

If you want to use ChatGPT to auto-generate prompts from the transcript you'll also need to 
`export OPENAI_API_KEY= ...` in your terminal. 


## Settings:
`settings.py` contains the most prominant tweakable parameters of the pipeline, including the chatgpt task description and SD render settings.

