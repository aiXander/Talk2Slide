# voice-to-image (with optional chatgpt for prompt generation)

```
conda create -n v2i python=3.10 -y
conda activate v2i
```

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu117
pip3 install -r requirements_diffusers.txt
pip3 install -r requirements_whisper.txt
```

## Main pipeline flow:
	- Initial idea of hooking up whisper to StableDiffusion to do live voice prompting was implemented by (@Huemin)[https://twitter.com/huemin_art]
	- I then added async speech-to-text to enable capturing the full conversation transcript + ChatGPT support to turn that transcript into a relevant prompt
	
## Settings:
`settings.py` contains the most prominant tweakable parameters of the pipeline, including the chatgpt task description and SD render settings.

