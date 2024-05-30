import os
import easyocr
import numpy as np
import ocr
from stable_diffusion_pytorch import model_loader, pipeline

if not os.path.exists('stable-diffusion-pytorch'):
    os.system('git clone https://github.com/kjsman/stable-diffusion-pytorch.git')
    
models = model_loader.preload_models(device="cpu")

prompt = "a man holding a sign that says 'Hello CS281B'"
replace = ["Hello CS281B"]



# encapsulate diffusion params
def diffusion(prompt):
    
    prompts = [prompt]

    uncond_prompt = ""  #@param { type: "string" }
    uncond_prompts = [uncond_prompt] if uncond_prompt else None

    input_images = None

    strength = 0.8  #@param { type:"slider", min: 0, max: 1, step: 0.01 }

    do_cfg = True  #@param { type: "boolean" }
    cfg_scale = 7.5  #@param { type:"slider", min: 1, max: 14, step: 0.5 }
    height = 512  #@param { type: "integer" }
    width = 512  #@param { type: "integer" }
    sampler = "k_lms"  #@param ["k_lms", "k_euler", "k_euler_ancestral"]
    n_inference_steps = 35  #@param { type: "integer" }

    use_seed = False  #@param { type: "boolean" }
    if use_seed:
        seed = 42  #@param { type: "integer" }
    else:
        seed = None

    image = pipeline.generate(prompts=prompts, uncond_prompts=uncond_prompts,
                    input_images=input_images, strength=strength,
                    do_cfg=do_cfg, cfg_scale=cfg_scale,
                    height=height, width=width, sampler=sampler,
                    n_inference_steps=n_inference_steps, seed=seed,
                    models=models, device='cuda', idle_device='cpu')[0]
    
    
    return image

def simple_inpaint(image, bounds, word):
    from td_inpaint import inpaint
    from inpaint_functions import parse_bounds
    
    global_dict = {}
    global_dict["stack"] = parse_bounds(bounds, word)
    print(global_dict["stack"])
    #image = "./hat.jpg"
    prompt = ""
    keywords = ""
    positive_prompt = ""
    radio = 8
    slider_step = 25
    slider_guidance= 7
    slider_batch= 4
    slider_natural= False
    return inpaint(image, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict)


models = model_loader.preload_models(device="cpu")

# Basic Stable Diffusion
image = diffusion(prompt)
ocr.display(image)
model_loader.delete_models(models)

image = image.resize((512,512))
images = [image]

image_arr = np.array(image)

# OCR Mask Gen
reader = easyocr.Reader(['en'])
bounds = reader.readtext(image_arr)

# TD2 Inpaint
result = simple_inpaint(image, bounds, replace)
stitched = ocr.stitch(result[0])
ocr.display(stitched)