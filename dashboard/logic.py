import torch
import gc
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from PIL import Image, ImageFilter

# MODEL LOADER (JANGAN DIUBAH)
def load_models_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models to {device}")

    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(device)

    return pipe_txt2img, pipe_inpaint

# Ini mencegah error "Function not found" jika hanya mengerjakan Basic
def flush_memory(): pass
def set_scheduler(pipe, name): return pipe
def run_inpainting(pipe, img, mask, prompt, strength): return None
def prepare_outpainting(img, expand=128): return img, None


def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg, num_images=1, scheduler_name="Euler A"):

    ### MULAI CODE ###

    # Setup Generator (Seed)
    generator = torch.Generator("cuda").manual_seed(seed)

    # Generate gambar standard
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator
    ).images[0]

    ### SELESAI CODE ###

    # Kembalikan dalam bentuk List agar kompatibel dengan UI (List isi 1 gambar)
    return [image]

# Implementasi pembersihan RAM GPU
def flush_memory():

    ### MULAI CODE ###

    gc.collect()
    torch.cuda.empty_cache()

    ### SELESAI CODE ###

    print("Memory Flushed!")

# Ganti scheduler sesuai input
def set_scheduler(pipe, scheduler_name):

    ### MULAI CODE ###

    if scheduler_name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    ### SELESAI CODE ###

    return pipe

# Definisikan ulang fungsi generate_image dan tambahkan parameter untuk batch inference
def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg, num_images=1, scheduler_name="Euler A"):

    ### MULAI CODE ###

    # Set Scheduler
    pipe = set_scheduler(pipe, scheduler_name)

    generator = torch.Generator("cuda").manual_seed(seed)

    # Generate Batch
    result = pipe(
        prompt=[prompt] * num_images,
        negative_prompt=[neg_prompt] * num_images,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator
    ).images

    ### SELESAI CODE ###

    return result

def run_inpainting(pipe, image, mask, prompt, strength):
    # Pastikan konversi RGB/L dan Resize Mask (Nearest)
    if image.mode != "RGB": image = image.convert("RGB")
    if mask.mode != "L": mask = mask.convert("L")

    # Resize Mask agar tajam
    if image.size != mask.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        strength=strength
    ).images[0]

    return result

def prepare_outpainting(image, expand_pixels=128):

    ### MULAI CODE ###
    w, h = image.size
    new_w = w + (2 * expand_pixels)
    new_h = h + (2 * expand_pixels)

    # Safety: Resolusi kelipatan 8
    new_w -= (new_w % 8)
    new_h -= (new_h % 8)

    ### SELESAI CODE ###

    # Background Blur
    bg = image.resize((new_w, new_h), resample=Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=50))

    canvas = bg.copy()
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(image, (paste_x, paste_y))

    # Masker (Putih = Edit, Hitam = Keep)
    mask = Image.new("L", (new_w, new_h), 255)
    inner_box = Image.new("L", (w, h), 0)

    ### MULAI CODE ###

    mask.paste(inner_box, (paste_x, paste_y))

    ### SELESAI CODE ###

    return canvas, mask
