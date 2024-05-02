'''
Understanding the implementation of stable diffusion in Hugging Face diffusers
Reference. https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline
'''
# Deconstruct the stable diffusion pipeline
# The Stable Diffusion model has three separate pretrained models in latent diffusion
# 1. An autoencoder (VAE). Convert the image into a low dimensional latent representation.
# 2. A U-Net. Output predicts the noise residual which can be used to compute the predicted denoised image representation.
# 3. A text-encoder, e.g. CLIP's Text Encoder. Transform the input prompt into an embedding space.
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import DDIMScheduler
from tqdm.auto import tqdm
from pathlib import Path

from rewards import aesthetic_score

prefix_path = '/tmp2/lupoy/L4HF/model_assets'
model_name = 'runwayml/stable-diffusion-v1-5' 
torch_device = "cuda"
prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 10  # Number of denoising steps
is_random_steps = False  # random steps ['exp', True, False (default)]
guidance_scale = 5 # 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(1126)  # Seed generator to create the initial latent noise

vae = AutoencoderKL.from_pretrained(f'{prefix_path}/{model_name}', subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(f'{prefix_path}/{model_name}', subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(f'{prefix_path}/{model_name}', subfolder="text_encoder", use_safetensors=True)
unet = UNet2DConditionModel.from_pretrained(f'{prefix_path}/{model_name}', subfolder='unet', use_safetensors=True)
# scheduler
# Hugging Face official suggests: PNDM (used by default), DDIM, K-LMS
scheduler = DDIMScheduler.from_pretrained(f'{prefix_path}/{model_name}', subfolder='scheduler')

# cuda
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# reward model
reward_fn = aesthetic_score

# 1. Create text embeddings
batch_size = len(prompt)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# 2. Create random noise
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
).to(torch_device)
# 3. Denoise the image
latents = latents * scheduler.init_noise_sigma
# 3.1 Set the scheduler’s timesteps to use during denoising.
scheduler.set_timesteps(num_inference_steps)

# save noisy images in a directory
if is_random_steps == 'exp':
    noisyImgs_dir = f"expNoisyImgs-timesteps_{num_inference_steps}"
elif is_random_steps:
    noisyImgs_dir = f"randNoisyImgs-timesteps_{num_inference_steps}"
else:
    noisyImgs_dir = f"noisyImgs-timesteps_{num_inference_steps}"

Path(noisyImgs_dir).mkdir(parents=True, exist_ok=True)

# 3.2 iterative denoising
if is_random_steps == 'exp':
    # Calculate the base and the values for each of the 10 steps (x from 0 to 9)
    base = scheduler.config['num_train_timesteps'] ** (1/(num_inference_steps - 1))  # Calculating the base as the ninth root of 1000

    # Compute the values for each step using the base
    timesteps = torch.as_tensor([int(base) ** x for x in range(num_inference_steps)][::-1])
elif is_random_steps:
    timesteps, _ = torch.sort(torch.randperm(scheduler.config['num_train_timesteps'])[:num_inference_steps-1], descending=True)
    timesteps = torch.as_tensor(timesteps.tolist() + [1])
else:
    timesteps = scheduler.timesteps

for t in tqdm(timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 3.3 Decode the noisy image
    noisy_latents = 1 / 0.18215 * latents
    with torch.no_grad():
        noisy_image = vae.decode(noisy_latents).sample
    noisy_image = (noisy_image / 2 + 0.5).clamp(0, 1).squeeze()
    noisy_image = (noisy_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    noisy_image = Image.fromarray(noisy_image)
    noisy_image.save(f'{noisyImgs_dir}/step_{t}.png')

# 4. Decode the final image
# scale and decode the final image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
if is_random_steps == 'exp':
    image.save(f'exptimesteps_{num_inference_steps}.png')
elif is_random_steps:
    image.save(f'randtimesteps_{num_inference_steps}.png')
else:
    image.save(f'timesteps_{num_inference_steps}.png')

breakpoint()