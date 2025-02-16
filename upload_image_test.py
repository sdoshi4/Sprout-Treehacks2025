import requests

# Read image as bytes
with open("kids_drawing.jpg", "rb") as f:
    image_bytes = f.read()
# print(image_bytes)

# Send the request
response = requests.post(
    "https://385a-68-65-164-29.ngrok-free.app/upload_image/",
    data=image_bytes,
    headers={"Content-Type": "application/octet-stream"}
)

# Print response
print(response.json())