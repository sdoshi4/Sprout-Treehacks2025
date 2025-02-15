from lumaai import LumaAI
import os
import time 
import requests

client = LumaAI(
    auth_token=os.environ.get("LUMAAI_API_KEY"),
)

generation = client.generations.image.create(
    prompt="this is a happy family show the happy family ",
    image_ref=[
      {
        "url": "https://yourelc.com.au/wp-content/uploads/2022/07/Blog.png",
        "weight": 0.85
      }
    ]
)



completed = False
while not completed:
  generation = client.generations.get(id=generation.id)
  if generation.state == "completed":
    completed = True
  elif generation.state == "failed":
    raise RuntimeError(f"Generation failed: {generation.failure_reason}")
  print("Dreaming")
  time.sleep(2)

image_url = generation.assets.image

# download the image
response = requests.get(image_url, stream=True)
with open(f'{generation.id}.jpg', 'wb') as file:
    file.write(response.content)
print(f"File downloaded as {generation.id}.jpg")
