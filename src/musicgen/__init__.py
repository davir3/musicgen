import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "A high-energy 2000s nu-metal and rap-rock track. Heavy distorted electric guitar riffs, punchy aggressive drums, and a sarcastic, witty male vocal delivery. Fast-paced, cinematic action movie soundtrack vibes, rebellious and comedic."

print("Cargando MusicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(
    text=[PROMPT],
    padding=True,
    return_tensors="pt"
).to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1500
    )

sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)

print("Musica generada correctamente: musica.wav")