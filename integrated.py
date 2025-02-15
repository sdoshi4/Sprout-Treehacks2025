from lumaai import LumaAI
import os
import time
import requests
import yaml
from google import genai
from PIL import Image
from google.genai import types
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor


#api keys
with open('keys.yaml', 'r') as file:
    keys = yaml.safe_load(file)

luma_client = LumaAI(auth_token=keys['lumaai_api_key'])
gemini_client = genai.Client(api_key=keys['gemini_api_key'])

class StoryOutput(BaseModel):
    story: str
    image_prompt: str
    options: list[str]

# this just generates an image from str promtp
def generate_image(prompt):
    generation = luma_client.generations.image.create(prompt=prompt)
    
    while generation.state != "completed":
        generation = luma_client.generations.get(id=generation.id)
        if generation.state == "failed":
            raise RuntimeError(f"Image generation failed: {generation.failure_reason}")
        time.sleep(2)

    image_url = generation.assets.image
    response = requests.get(image_url, stream=True)
    filename = f'{generation.id}.jpg'
    with open(filename, 'wb') as file:
        file.write(response.content)
    return filename

# this tkaes in an image and generates the start of the story via gemini multimodeal input
def generate_story_from_image(image_path):
    image = Image.open(image_path)
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, "This is an image of a child's drawing. Generate the first chapter of a children's book (around 100 words) where the main character has to make a choice. Return 'story', 'image_prompt', and 'options' as described previously."],
        config={
            'response_mime_type': 'application/json',
            'response_schema': StoryOutput,
        },
    )
    return response.parsed

# genereates the next chapter based on previous choices
def generate_next_story(story, choice):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"Chapter: {story}", f"Choice: {choice}", "Generate the next chapter based on this choice."],
        config={
            'response_mime_type': 'application/json',
            'response_schema': StoryOutput,
        },
    )
    return response.parsed

# main loop for storytelling
def storytelling_loop(start_image_path, num_panels=4):
    current_image_path = start_image_path
    for panel in range(num_panels):
        if panel == 0:
            story_output = generate_story_from_image(current_image_path)
        else:
            choice = input(f"Choose option 1 or 2: {story_output.options[0]} / {story_output.options[1]}: ")
            choice_text = story_output.options[int(choice) - 1]
            story_output = generate_next_story(story_output.story, choice_text)

        print(f"Panel {panel + 1} Story: {story_output.story}")
        print(f"Options: 1. {story_output.options[0]} 2. {story_output.options[1]}")

        current_image_path = generate_image(story_output.image_prompt)

        # get the 2 images as the child is reading
        choice1_prompt = f"Make an illustration for the story if the character chooses: {story_output.options[0]}"
        choice2_prompt = f"Make an illustration for the story if the character chooses: {story_output.options[1]}"

        # trying to do concurrent image generation via python threads (not sure if this will work when we do full integrate)
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(generate_image, choice1_prompt)
            future2 = executor.submit(generate_image, choice2_prompt)

        choice1_image = future1.result()
        choice2_image = future2.result()

        print(f"generated images for choices saved as: {choice1_image}, {choice2_image}")

# main running
storytelling_loop('kids_drawing.jpg', num_panels=4)
