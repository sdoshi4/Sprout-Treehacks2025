import requests

# Read image as bytes
with open("kids_drawing.jpg", "rb") as f:
    image_bytes = f.read()
with open("bytes.txt", 'wb') as f2:
    f2.write(image_bytes)
# print(image_bytes)

# Send the request
response = requests.post(
    "https://385a-68-65-164-29.ngrok-free.app/upload_image/",
    data=image_bytes,
    headers={"Content-Type": "application/octet-stream"}
)

# Print response
print(response.json())