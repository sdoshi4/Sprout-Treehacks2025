from lumaai import LumaAI
import os
import time
import random 
import requests
import yaml
from google import genai
from PIL import Image
from google.genai import types
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from fastapi import FastAPI, Body, Request, Form
import requests
from io import BytesIO
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import math

# from PyDictionary import PyDictionary

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


app = FastAPI()

comprehension_vocab = {
    "grade_3": [
        "abbreviation", "adverb", "biography", "character", "chronological", "conjunction",
        "context", "declarative", "encyclopedia", "fact", "glossary", "index", "inference",
        "interrogative", "main", "modern", "opinion", "persuasion", "possessive", "revise",
        "subject", "theme", "attentive", "destination", "emerge", "fragrance", "habitation",
        "illuminate", "jargon", "knoll", "misgiving", "pertinent", "serenity", "stamina", "vigilant", "waft"
    ],
    "grade_4": [
        "almanac", "analyze", "audience", "compare", "contrast", "evaluate", "genre",
        "legend", "metaphor", "outline", "paraphrase", "publish", "research", "sentence",
        "simile", "supporting", "text", "transition", "adversary", "banish", "bluff", "cunning",
        "defiance", "dispel", "egregious", "falter", "grueling", "headway", "imperious",
        "malleable", "momentum", "obscure", "precipice", "replenish", "scarcity", "somber", "swagger", "tactic"
    ],
    "grade_5": [
        "caption", "conflict", "figurative", "idiom", "interjection", "minor", "onomatopoeia",
        "outline", "poetic", "reference", "resolution", "stereotypical", "superlative",
        "supporting", "word", "barricade", "circumference", "concoction", "contortion",
        "disdain", "exasperation", "foresight", "gusto", "ignite", "jut", "meander",
        "monotonous", "muster", "outlandish", "pristine", "quell", "recuperate",
        "restitution", "subside", "translucent", "unsightly", "weather"
    ],
    "grade_6": [
        "affix", "analogy", "allusion", "appositive", "clause", "dialect", "literal",
        "mythology", "narrative", "phrases", "plagiarism", "predicate", "propaganda",
        "relevant", "sentence", "synthesize", "aplomb", "apprehensive", "brackish",
        "debris", "deft", "diminish", "dismal", "ember", "engross", "exhilarate",
        "furtive", "hasten", "impending", "jabber", "jostle", "kindle", "luminous",
        "materialize", "meticulous", "multitude", "narrate", "ominous", "persistent",
        "potential", "repugnant", "sabotage", "scurry", "sociable", "specimen",
        "swarm", "terse", "uncanny", "versatile", "vulnerable", "waver", "zeal"
    ],
    "grade_7": [
        "assumption", "clause", "convention", "description", "exposition", "flashback",
        "fluency", "foreshadowing", "imagery", "interpretation", "irony", "nominative",
        "prose", "types", "viewpoint", "brandish", "commotion", "conspicuous", "counter",
        "eavesdrop", "engross", "exasperation", "falter", "fragrance", "grueling", "headway",
        "ignite", "imperious", "jargon", "jut", "luminous", "meander", "momentum", "muster",
        "obscure", "pertinent", "potential", "pristine", "recuperate", "repugnant", "scarcity",
        "serenity", "swagger", "tactic", "translucent", "unsightly", "vigilant", "waft"
    ],
    "grade_8": [
        "agreement", "argument", "bias", "coherence", "debate", "derivation", "dramatization",
        "elaboration", "gerund", "inference", "infinitive", "parallel", "persuasive",
        "sensory", "thesis", "adversary", "apprehensive", "barricade", "banish", "bluff",
        "circumference", "concoction", "contortion", "cunning", "defiance", "destination",
        "disdain", "dispel", "egregious", "emerge", "exasperation", "falter", "foresight",
        "gusto", "habitation", "headway", "illuminate", "imperious", "jargon", "knoll",
        "malleable", "meander", "misgiving", "momentum", "muster", "obscure", "outlandish",
        "pertinent", "precipice", "pristine", "recluse", "replenish", "restitution",
        "scarcity", "serenity", "somber", "stamina", "swagger", "tactic", "translucent",
        "unsightly", "vigilant", "waft", "zeal"
    ]
}

chosen_vocab = []
chosen_grade_level_key = "grade_4"
used_word2 = False



with open('keys.yaml', 'r') as file:
    keys = yaml.safe_load(file)

luma_client = LumaAI(auth_token=keys['lumaai_api_key'])
gemini_client = genai.Client(api_key=keys['gemini_api_key'])

class ImageData(BaseModel):
    name: str
    bytes: List[int]  # Expecting a list of integers representing image bytes

class RequestModel(BaseModel):
    data: ImageData

class StoryOutput(BaseModel):
    story: str
    title: str
    image_prompt: str
    options: list[str]

class StoryRequest(BaseModel): # This is asking for the 
    image_path: Optional[str] = None
    story: Optional[str] = None
    title: Optional[str] = None
    choice: Optional[str] = None

class StoryResponse(BaseModel): # This is the output of gemini
    story: str
    title: str
    image_prompt: str
    options: list[str]
    image_path: str

def get_definition(word):
    synsets = wn.synsets(word)
    if not synsets:
        return None
    # Get the first definition as a simple approach
    return synsets[0].definition()


def upload_to_imgur(image_path: str) -> str:
    """
    Uploads an image to Imgur and returns the image URL.
    """
    client_id = "40c6711dcccaf03" # TODO : fix bc this is hardcoded
    url = "https://api.imgur.com/3/upload"
    headers = {"Authorization": f"Client-ID {client_id}"}
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            url,
            headers=headers,
            files={"image": image_file}
        )
    if response.status_code != 200:
        raise RuntimeError(f"Imgur upload failed: {response.json()}")
    return response.json()['data']['link']



def generate_image(prompt, image_url: Optional[str] = None):
    """
    Generate an image using Luma with an optional reference image URL as context.
    """
    if image_url:
        # Pass the image URL as conditioning input if supported by Luma API
        generation = luma_client.generations.image.create(
            prompt=prompt + " Make sure the style is similar to a children's book or more cartoonish, similar to their drawing.",
            image_ref=[
                {
                    "url": image_url,
                    "weight": 0.85
                }
            ], 
            style_ref=[
                {
                    "url": image_url,
                    "weight": 0.85
                }
            ]
        )
    else:
        generation = luma_client.generations.image.create(
            prompt=prompt
        )

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

    # Upload the generated image to Imgur
    imgur_url = upload_to_imgur(filename)

    return imgur_url  # Return Imgur URL instead of local path

# USED FOR THE FIRST ITERATION
def generate_story_from_image(image: Image.Image, grade_level_key="grade_4"):
    chosen_grade_level_key = grade_level_key
    i, j = 0
    while (i==j):
        i, j = math.random(0, len(comprehension_vocab[grade_level_key]))
    chosen_vocab.append(comprehension_vocab[grade_level_key][i])
    chosen_vocab.append(comprehension_vocab[grade_level_key][j])
    
    


    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, '''Analyze the provided image of a child's drawing and create the following for the first chapter of a children's storybook:

1. Image description (image_prompt): 
   - Provide a detailed description of the key elements in the drawing.
   - Include specific details about characters, settings, and objects that should be consistent in future chapters.

2. Title: 
   - Create a title in the format "Chapter 1: {title name here}"

3. Story:
   - Write a 300-word chapter based on the image.
   - Introduce and describe the main characters and setting in detail.
   - Develop a clear plot point or conflict.
   - End the chapter with a situation that leads to two distinct choices.
   

4. Options:
   - Provide two specific, mutually exclusive choices for how the story could continue.
   - Keep each option brief, ideally 10 words or less.

Return the results in the following JSON format:
{
  "image_prompt": "[Detailed image description for consistency]",
  "title": "Chapter 1: [Your Title Here]",
  "story": "[300-word chapter content]",
  "options": ["[Brief Option 1]", "[Brief Option 2]"]
}

Note: Ensure the title strictly follows the format "Chapter 1: {title name here}".
''' + f"Finally, in the story, ensure that you incorporate the chosen vocabulary word: {chosen_vocab[0]}; this is very important"],
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

# USED FOR ALL AFTER THE FIRST
def generate_next_story(title, story, choice):
    
    if (not used_word2):
        vocab_prompt_engineering = f"Finally, in the story, ensure that you incorporate the chosen vocabulary word: {chosen_vocab[1]}; this is very important"
        used_word2 = True
    else:
        vocab_prompt_engineering = f"Finally, in the story, if it makes sense, you should incorporate the chosen vocabulary words: {chosen_vocab[0]} and/or {chosen_vocab[1]};"
        
    
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"Chapter title: {title}", f"Chapter: {story}", f"Choice: {choice}", '''
                  Previous chapter information:
Previous chapter information:
Title: {title}
Content: {story}
User's choice: {choice}

Based on the previous chapter and the user's choice, generate the next chapter of the children's storybook:

1. Image description (image_prompt):
   - Provide a detailed description for a new image that reflects the user's choice and continues the story.
   - Ensure consistency with previously established characters and settings.

2. Title:
   - Create a title in the format "Chapter X: {title name here}", where X is the next chapter number.

3. Story:
   - Write a 300-word chapter that directly follows from the user's choice.
   - Continue developing the main characters and setting.
   - Introduce new elements or conflicts as appropriate.
   - End the chapter with a new situation that leads to two distinct choices.

4. Options:
   - Provide two specific, mutually exclusive choices for how the story could continue.
   - Keep each option brief, ideally 10 words or less.

Return the results in the following JSON format:
{
  "image_prompt": "[Detailed image description for the new chapter]",
  "title": "Chapter X: [Your Title Here]",
  "story": "[300-word new chapter content]",
  "options": ["[Brief Option 1]", "[Brief Option 2]"]
}

Note: Maintain consistency with the characters, settings, and plot developments from the previous chapter. The new chapter should be a direct continuation based on the user's choice. Ensure the title strictly follows the format "Chapter X: {title name here}".
                  ''' + vocab_prompt_engineering],
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

# USED FOR ALL AFTER THE FIRST
def generate_final_story(title, story, choice):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"Chapter title: {title}", f"Chapter: {story}", f"Choice: {choice}", '''
                  Previous chapter information:
Title: {title}
Content: {story}
User's choice: {choice}

Based on the previous chapter and the user's choice, generate the final chapter of the children's storybook:

1. Image description (image_prompt):
   - Provide a detailed description for a concluding image that reflects the story's resolution.
   - Ensure consistency with previously established characters and settings.

2. Title:
   - Use the format "Title: Chapter {number}: The End!"

3. Story:
   - Write a 300-word concluding chapter that wraps up the story.
   - Resolve the main conflicts or challenges introduced in previous chapters.
   - Provide a satisfying ending that ties together the story elements.
   - Include a clear moral or lesson learned by the main character(s).
   - End with a sense of closure, but hint at potential future adventures if appropriate.

4. Options:
   - Both options should be identical and say "Go to Quiz".

Return the results in the following JSON format:
{
  "image_prompt": "[Detailed image description for the final chapter]",
  "title": "Title: Chapter {number}: The End!",
  "story": "[300-word concluding chapter content]",
  "options": ["Go to Quiz", "Go to Quiz"]
}

Note: Ensure the story comes to a satisfying conclusion while maintaining consistency with previous chapters' characters, settings, and plot developments. The ending should feel complete and rewarding for the reader.
                  '''],
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


# @app.post("/generate_first_panel", response_model=StoryResponse)
# def generate_first_panel(request: StoryRequest):
#     story_output = generate_story_from_image(request.image_path)

#     # image_path = generate_image(story_output.image_prompt)
#     image_path = generate_image(story_output.image_prompt, image_url=request.image_path)


#     return StoryResponse(
#         story=story_output.story,
#         title=story_output.title,
#         image_prompt=story_output.image_prompt,
#         options=story_output.options,
#         image_path= image_path # this is a url now
#     )




# testing dictionary code
# if __name__ == "__main__":
#     # test generate_quiz_from_vcab:
#     vocab_words = ["abbreviation", "adverb"]
#     chosen_grade_level_key = "grade_3"
#     quiz_questions = generate_quiz_from_vocab(vocab_words)
#     print(quiz_questions)



def generate_quiz_from_vocab():
    ''' inputs: list of vocab words
        outputs: list of quiz questions '''
    # return [{'word': 'abbreviation', 'correct_answer': 'a shortened form of a word or phrase', 'options': ['any very large body of (salt) water', 'a shortened form of a word or phrase', 'carefully observant or attentive; on the lookout for possible danger', 'the act of persuading (or attempting to persuade); communication intended to induce belief or action']}, {'word': 'adverb', 'correct_answer': 'the word class that qualifies verbs or clauses', 'options': ["an account of the series of events making up a person's life", 'the word class that qualifies verbs or clauses', 'uneasiness about the fitness of an action', 'the reasoning involved in drawing a conclusion or making a logical judgment on the basis of circumstantial evidence and prior conclusions rather than on the basis of direct observation']}]
    quiz_questions = []
    chosen_vocab = ["abbreviation", "adverb"] #TODO change
    chosen_grade_level_key = "grade_3" #TODO change

    for word in chosen_vocab:
        # 1. Get the correct definition
        
        correct_definition = get_definition(word)
        

        # 2. Get 3 wrong answers from the same grade level (excluding the word itself)
        grade_words = comprehension_vocab[chosen_grade_level_key]
        other_words = [w for w in grade_words if w != word]

        wrong_answers = []
        while len(wrong_answers) < 2:
            random_word = random.choice(other_words)
            wrong_def = get_definition(random_word)

            if wrong_def and wrong_def != correct_definition and wrong_def not in wrong_answers:
                wrong_answers.append(wrong_def)

        # 3. Prepare options
        options = [correct_definition] + wrong_answers
        random.shuffle(options)

        # 4. Store the quiz question
        quiz_questions.append({
            "word": word,
            "correct_answer": correct_definition,
            "options": options
        })

    return quiz_questions  


###### ----------------- --------------------------- ENDPOINTS ----------------- --------------------------- ######
  

@app.post("/generate_quiz", response_model=List[Dict[str, Any]])
def generate_quiz(request: Any):
    quiz_questions = generate_quiz_from_vocab()
    return quiz_questions


@app.post("/generate_next_panel", response_model=StoryResponse)
def generate_next_panel(request: StoryRequest):
    story_output = generate_next_story(request.title, request.story, request.choice)
    image_path = generate_image(story_output.image_prompt)

    return StoryResponse(
        story=story_output.story,
        title=story_output.title,
        image_prompt=story_output.image_prompt,
        options=story_output.options,
        image_path=image_path # this is a url now
    )
    
    


@app.post("/generate_final_panel", response_model=StoryResponse)
def generate_final_panel(request: StoryRequest):
    story_output = generate_final_story(request.title, request.story, request.choice)
    image_path = generate_image(story_output.image_prompt)

    return StoryResponse(
        story=story_output.story,
        title=story_output.title,
        image_prompt=story_output.image_prompt,
        options=story_output.options,
        image_path=image_path # this is a url now
    )
# THIS IS JUST FOR TESTING
# @app.get("/generate_story")
# def generate_next_panel():
#     image = Image.open("kids_drawing.jpg")
#     story_output = generate_story_from_image(image)
#     image_path = generate_image(story_output.image_prompt)

#     return StoryResponse(
#         story=story_output.story,
#         title=story_output.title,
#         image_prompt=story_output.image_prompt,
#         options=story_output.options,
#         image_path=image_path
#     )


@app.get("/images/{image_name}")
def get_image(image_name: str):
    return FileResponse(f"images/{image_name}")


@app.get("/")
def read_root():
    return {"message": "Storyboard Backend is Running!"}


# @app.get("/get_first_panel")
# def get_first_panel(request: StoryRequest):
#     story_output = generate_story_from_image(request.image_path)
#     image_path = generate_image(story_output.image_prompt)

#     return StoryResponse(
#         story=story_output.story,
#         title=story_output.title,
#         image_prompt=story_output.image_prompt,
#         options=story_output.options,
#         image_path=image_path
#     )

#THIS WORKS LOCALLY
# @app.post("/upload_image/")
# async def upload_image(image_bytes: bytes = Body(..., media_type="application/octet-stream")):
#     image = Image.open(BytesIO(image_bytes))
#     story_output = generate_story_from_image(image)
#     image_path = generate_image(story_output.image_prompt)
    
#     return StoryResponse(
#         story=story_output.story,
#         title=story_output.title,
#         image_prompt=story_output.image_prompt,
#         options=story_output.options,
#         image_path=image_path
#     )
    
# THIS WORKS LOCALLY
# @app.post("/upload_image_flutterflow/")
# async def upload_image(request: Dict[str, Any] = Body(...)):
#     try:
#         print("Received Request:", request)

#         # Step 1: Extract the JSON string from "json" key
#         if "json" not in request:
#             raise HTTPException(status_code=400, detail="Missing 'json' key in request body")

#         json_str = request["json"]

#         # Step 2: Parse JSON string into a dictionary
#         try:
#             image_data_dict = json.loads(json_str)
#         except json.JSONDecodeError:
#             raise HTTPException(status_code=400, detail="Invalid JSON format inside 'json' key")

#         # Step 3: Validate that the parsed object contains expected fields
#         if not isinstance(image_data_dict, dict) or "bytes" not in image_data_dict:
#             raise HTTPException(status_code=400, detail="Missing 'bytes' key inside parsed JSON")

#         # Step 4: Convert list of integers into bytes
#         image_bytes = bytes(image_data_dict["bytes"])
#         image = Image.open(BytesIO(image_bytes))

#         # Process the image (replace with actual processing functions)
#         story_output = generate_story_from_image(image)
#         image_path = generate_image(story_output.image_prompt)

#         return {
#             "story": story_output.story,
#             "image_prompt": story_output.image_prompt,
#             "options": story_output.options,
#             "image_path": image_path
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/upload_image_flutterflow/")
async def upload_image(data: UploadFile = File(None), json_data: str = Form(None)):  
    try:
        if data:
            print("Received File:", data.filename)
            image_bytes = await data.read()
        elif json_data:
            print("Received JSON Data")
            json_dict = json.loads(json_data)
            if "bytes" not in json_dict:
                raise HTTPException(status_code=400, detail="Missing 'bytes' key in JSON")
            image_bytes = bytes(json_dict["bytes"])
        else:
            raise HTTPException(status_code=400, detail="No valid image data received")

        # Convert bytes to image
        image = Image.open(BytesIO(image_bytes))

        # Process image (replace with actual logic)
        story_output = generate_story_from_image(image)
        image_path = generate_image(story_output.image_prompt)

        return {
            "story": story_output.story,
            "title": story_output.title,
            "image_prompt": story_output.image_prompt,
            "options": story_output.options,
            "image_path": image_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")