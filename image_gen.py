from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# Use CPU (for your laptop)
pipe = pipe.to("cpu")

# Define your text prompt
prompt = "A futuristic city floating in the sky, digital art"

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("output.png")

print("✅ Image saved as output.png")
