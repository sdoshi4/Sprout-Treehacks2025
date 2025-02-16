import requests
import json

# Read image as bytes
# with open("kids_drawing.jpg", "rb") as f:
#     image_bytes = f.read()
# with open("bytes.txt", 'wb') as f2:
#     f2.write(image_bytes)
# print(image_bytes)


with open('image.json', 'r') as file:
    image_bytes = json.load(file)
    

# print(data["data"]["bytes"])
# image_bytes = bytes(data["data"]["bytes"])
# Send the request
response = requests.post(
    # "https://385a-68-65-164-29.ngrok-free.app/upload_image/",
    
    "https://a771-68-65-164-139.ngrok-free.app/upload_image_flutterflow",
    data=image_bytes,
    headers={"Content-Type": "application/octet-stream"}
)

# response = requests.get("https://385a-68-65-164-29.ngrok-free.app/generate_story")

# Print response
print(response.json())