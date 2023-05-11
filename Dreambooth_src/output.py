import diffusers
from PIL import Image
# from torch import autocast
from diffusers import StableDiffusionPipeline
import torch
import os
import csv

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def output(model_file,steps,prompt_text):
    name = f'{steps}_steps'
    model_id = os.path.join("./model_database/"+model_file+"/"+name)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    prompt = prompt_text

    num_samples = 5
    num_rows = 5

    all_images = [] 
    for _ in range(num_rows):
        images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)

    grid = image_grid(all_images, num_samples, num_rows)
    grid.save(f"../{model_file}_{steps}.jpg")

def output_for_dataset(model_file,steps,prompt_text):
    name = f'{steps}_steps'
    model_id = os.path.join("./model_database/"+model_file+"/"+name)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    prompt = prompt_text
    folder_name = prompt.replace("a photo of chest x-ray, ","")

    if not os.path.exists(f"../output_images_dataset/{folder_name}"):
        os.mkdir(f"../output_images_dataset/{folder_name}")

    total_number = 10000
    image_num_now = 1

    while image_num_now <= total_number:
        image_for_onetime = 10
        image = pipe(prompt, num_images_per_prompt=image_for_onetime, num_inference_steps=50, guidance_scale=7.5).images
        for i in image:
            if i.getbbox(): # Check whether the image output is totally black
                i.save(f"../output_images_dataset/{folder_name}/{image_num_now}.jpg")
                image_num_now += 1


def check_for_dataset(prompt):
    content = 0 
    index = []
    prompt_text = "a photo of chest x-ray"
    file = open('../CheXpert-v1.0-small/train.csv')
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    csv_index = ""

    # Counting the diseases in the prompt
    for i,n in enumerate(header):  
        if n in prompt:
            prompt_text += f", {header[i]}" # Real input prompt
            csv_index += f"{header[i]} "
            content += 1
            index.append(i)

    if content != 0:
        # Check whether the model exists
        model_file = ""
        for i in index:
            model_file += str(f"{i}")
        if os.path.exists(f"./model_database/{model_file}/2500_steps"):
            print("How many steps do you want to use?")
            steps = input()
            output_for_dataset(model_file,steps,prompt_text)
            return
        else:
            print("This hasn't been trained, use train.py to train the model.")

    else:
        print("There is no disease in the prompt.")

def output_for_contrast(model_file,steps,prompt_text):
    name = f'{steps}_steps'
    model_id = os.path.join("./model_database/"+model_file+"/"+name)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    prompt = prompt_text

    num_samples = 1
    num_rows = 10
    i = 1

    all_images = [] 
    while i <= num_rows:
        images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
        if images[0].getbbox(): # Check whether the image output is totally black
            i += 1
            all_images.extend(images)


    grid = image_grid(all_images, num_samples, num_rows)
    grid.save(f"../{model_file}_{steps}.jpg")
    
if __name__ == '__main__':
    print("Enter your prompt:   (This one is only for generating datasets!!)")
    prompt = input()
    check_for_dataset(prompt)