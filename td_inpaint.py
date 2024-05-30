# Source: 
# https://huggingface.co/spaces/JingyeChen22/TextDiffuser-2-Text-Inpainting

import torch
import gradio as gr

import numpy as np
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler,UNet2DConditionModel
from tqdm import tqdm
from PIL import Image, ImageDraw
import string
from inpaint_functions import format_prompt, to_tensor, add_tokens


#### import diffusion models
text_encoder = CLIPTextModel.from_pretrained('JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="text_encoder").cuda().half()

tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")

vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae").half().cuda()

unet = UNet2DConditionModel.from_pretrained(    'JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="unet").half().cuda()

scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler") 


def inpaint(orig_i, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict):

    # print(type(i))
    # exit(0)
    add_tokens(tokenizer, text_encoder)

    print(f'[info] Prompt: {prompt} | Keywords: {keywords} | Radio: {radio} | Steps: {slider_step} | Guidance: {slider_guidance} | Natural: {slider_natural}')
    print(f'Global Stack: {global_dict["stack"]}')

    # global stack
    # global state

    if len(positive_prompt.strip()) != 0:
        prompt += positive_prompt

    with torch.no_grad():
        image_mask = Image.new('L', (512,512), 0)
        draw = ImageDraw.Draw(image_mask)


        ### CLIP Tokenizer
        if slider_natural:
            user_prompt = f'{prompt}'
            composed_prompt = user_prompt
            prompt = tokenizer.encode(user_prompt)
            
        else:
            user_prompt = format_prompt(draw, prompt, global_dict['stack'])
                
            prompt = tokenizer.encode(user_prompt)
            
            composed_prompt = tokenizer.decode(prompt)

            print("Composed Prompt:",composed_prompt)
        
         
        prompt = prompt[:77]
        while len(prompt) < 77: 
            prompt.append(tokenizer.pad_token_id) 

        prompts_cond = prompt
        prompts_nocond = [tokenizer.pad_token_id]*77
        
        ### CLIP Encoder
        
        prompts_cond = [prompts_cond] * slider_batch
        prompts_nocond = [prompts_nocond] * slider_batch

        prompts_cond = torch.Tensor(prompts_cond).long().cuda()
        prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()
        
        encoder_hidden_states_cond = text_encoder(prompts_cond)[0].half()
        encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0].half()

        ### Apply mask
                
        # image_mask converted to a float16 tensor
        image_mask = torch.Tensor(np.array(image_mask)).float().half().cuda()
        
        # (H, W) -> (1, H, W) -> (1, 1, H, W) * (B, 1, 1, 1) = (B, 1, H, W)
        # The result is the image mask tensor repeated B times along the batch dimension.
        image_mask = image_mask.unsqueeze(0).unsqueeze(0).repeat(slider_batch, 1, 1, 1)

        # Resize og to 512x512
        image = orig_i.resize((512,512))
        
        # convert image tensor vals to distribution from [-1,1]
        image_tensor = to_tensor(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)   # (1, 3, 512, 512)
        
        # set masked area in tensor to 0.
        masked_image = image_tensor * (1-image_mask)
        
        ### VAE Encoder
        
        # sampled latent feature vector from VAE's encoded distribution.
        masked_feature = vae.encode(masked_image.half()).latent_dist.sample()
        
        # Scale the sampled latent feature by a scaling factor specified in the VAE configuration.
        masked_feature = (masked_feature * vae.config.scaling_factor).half() # (4, 4, 64, 64)
        
        print(f'masked_feature.shape {masked_feature.shape}')

        ## DDPM Sampler
        
        # Resize the image mask to 64x64 using nearest-neighbor interpolation.
        feature_mask = torch.nn.functional.interpolate(image_mask, size=(64,64), mode='nearest').cuda()
        noise = torch.randn((slider_batch, 4, 64, 64)).to("cuda").half()
        scheduler.set_timesteps(slider_step) 
        
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():  # classifier free guidance

                noise_pred_cond = unet(sample=noise, timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:slider_batch],feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
                noise_pred_uncond = unet(sample=noise, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:slider_batch],feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
                noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
                noise = scheduler.step(noisy_residual, t, noise).prev_sample
                del noise_pred_cond
                del noise_pred_uncond

                torch.cuda.empty_cache()

        ## VAE decode
        
        noise = 1 / vae.config.scaling_factor * noise 
        images = vae.decode(noise, return_dict=False)[0] 
        width, height = 512, 512
        results = []
        new_image = Image.new('RGB', (2*width, 2*height))
        for index, image in enumerate(images.cpu().float()):
            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
            results.append(image)
            row = index // 2
            col = index % 2
            new_image.paste(image, (col*width, row*height))
            
        # os.system('nvidia-smi')
        torch.cuda.empty_cache()
        # os.system('nvidia-smi')
        return tuple(results), composed_prompt
