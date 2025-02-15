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

from fastapi import FastAPI
from fastapi.responses import FileResponse
from typing import Optional


app = FastAPI()


with open('keys.yaml', 'r') as file:
    keys = yaml.safe_load(file)

luma_client = LumaAI(auth_token=keys['lumaai_api_key'])
gemini_client = genai.Client(api_key=keys['gemini_api_key'])

class StoryOutput(BaseModel):
    story: str
    image_prompt: str
    options: list[str]

class StoryRequest(BaseModel):
    image_path: Optional[str] = None
    story: Optional[str] = None
    choice: Optional[str] = None
    panel: int

class StoryResponse(BaseModel):
    story: str
    image_prompt: str
    options: list[str]
    image_path: str


def generate_image(prompt):
    generation = luma_client.generations.image.create(prompt=prompt)
    while generation.state != "completed":
        generation = luma_client.generations.get(id=generation.id)
        if generation.state == "failed":
            raise RuntimeError(f"Image generation failed: {generation.failure_reason}")
        time.sleep(2)

    image_url = generation.assets.image
    response = requests.get(image_url, stream=True)
    os.makedirs('images', exist_ok=True)
    filename = f'images/{generation.id}.jpg'

    with open(filename, 'wb') as file:
        file.write(response.content)

    return filename


def generate_story_from_image(image_path):
    image = Image.open(image_path)
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, "Generate the first chapter of a children's book (~100 words) with a choice. Return 'story', 'image_prompt', 'options'."],
        config={'response_mime_type': 'application/json', 
                'response_schema': StoryOutput,
                'safety_settings': [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            },
    )
    return response.parsed


def generate_next_story(story, choice):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"Chapter: {story}", f"Choice: {choice}", "Generate the next chapter."],
        config={'response_mime_type': 'application/json', 
                'response_schema': StoryOutput,
                'safety_settings': [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            },
    )
    return response.parsed


@app.post("/generate_first_panel", response_model=StoryResponse)
def generate_first_panel(request: StoryRequest):
    story_output = generate_story_from_image(request.image_path)
    image_path = generate_image(story_output.image_prompt)

    return StoryResponse(
        story=story_output.story,
        image_prompt=story_output.image_prompt,
        options=story_output.options,
        image_path=image_path
    )


@app.post("/generate_next_panel", response_model=StoryResponse)
def generate_next_panel(request: StoryRequest):
    story_output = generate_next_story(request.story, request.choice)
    image_path = generate_image(story_output.image_prompt)

    return StoryResponse(
        story=story_output.story,
        image_prompt=story_output.image_prompt,
        options=story_output.options,
        image_path=image_path
    )


@app.get("/images/{image_name}")
def get_image(image_name: str):
    return FileResponse(f"images/{image_name}")
