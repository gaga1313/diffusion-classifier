from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
root_dir = '/content/drive/MyDrive/Gg_Fossils_data_shared_copy/SD'

def get_sd_model():
  scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
  pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
  unet = pipe.unet
  vae = pipe.vae
  return vae, unet, scheduler

def give_image(dir):
  image_list = os.listdir(dir)
  image = Image.open(os.path.join(dir, image_list[0])).convert('RGB').resize(size=(512, 512))
  return image

def pil_to_latents(image):
  init_image = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
  init_image = init_image.to(device="cuda", dtype=torch.float16)
  init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
  return init_latent_dist

def latents_to_pil(latents):
  latents = (1 / 0.18215) * latents
  with torch.no_grad():
      image = vae.decode(latents).sample

  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  return pil_images

vae, unet, scheduler = get_sd_model()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype = torch.float16)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype = torch.float16).cuda()

prompt = [""]
tok =tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
emb = text_encoder(tok.input_ids.to("cuda"))[0].half()
print(f"Shape of embedding : {emb.shape}")

vae = vae.to("cuda")
image = give_image(root_dir)
latents = pil_to_latents(image)
print(latents.shape)


## Using U-Net to predict noise    
noise = torch.randn_like(latents)
ts = [1000, 281]
encoded_and_noised = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[t] for t in ts]))

print(encoded_and_noised.shape)
# latent_model_input = torch.cat([encoded_and_noised.to("cuda").float()]).half()
# with torch.no_grad():
#     noise_pred = unet(
#         latent_model_input,40,encoder_hidden_states=text_embeddings
#     )["sample"]
# ## Visualize after subtracting noise 
# latents_to_pil(encoded_and_noised- noise_pred)[0]