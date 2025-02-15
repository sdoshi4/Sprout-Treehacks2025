# AIzaSyDq-age16qrYQOssIusF87S4zGU1I9N17A
from google import genai
from PIL import Image
from google.genai import types
from pydantic import BaseModel, TypeAdapter

class StoryOutput(BaseModel):
  story: str
  image_prompt: str
  options: list[str]


# sys_instruct="You are an author of children's stories"
client = genai.Client(api_key="AIzaSyDq-age16qrYQOssIusF87S4zGU1I9N17A")
image = Image.open("kids_drawing.jpg")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    # config=types.GenerateContentConfig(system_instruction=sys_instruct),
    contents=[image, '''This is an image of a child's drawing. Please generate the first chapter of a children's book (around 500 words) from this image, leaving it where the
                main character has to make one of two choices. Store this story as the story parameter, and the two choices in the options list. 
                Next, generate a description of an image that would best describe this chapter of the story. Make this description descriptive enough such that one could draw an
                accurate image of the story without ever reading it. Store this as the image_prompt.'''],
    config={
        'response_mime_type': 'application/json',
        'response_schema': StoryOutput,
    },
)

output: StoryOutput = response.parsed
print(response.text)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    # config=types.GenerateContentConfig(system_instruction=sys_instruct),
    contents=['''Generate the next chapter of the story given the first choice occurs.''', f"Chapter: {output.story}", f"Option: {output.options[0]}"],
    # config={
    #     'response_mime_type': 'application/json',
    #     'response_schema': StoryOutput,
    # },
)
print(response.text)
# output: StoryOutput = response.parsed
# print(response.text)
# print()
# print(output.story)
# print(output.image_prompt)
# print(output.options)



# luma-014b98b0-cb5f-428e-bc1a-45e32698a433-191e7d4c-c751-40a5-bd90-f4c045659a0e