from flask import Flask, send_file
import torch
from diffusers import StableDiffusionPipeline
import io

app = Flask(__name__)

# Load the stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    dimensions = (400, 400) # (width, height) tuple
    image = pipe(prompt= prompt,   
              height = dimensions[0],
              width = dimensions[1],  
             ).images[0]
    
    # Save image to a BytesIO object
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return img_io

@app.route('/generate_image', methods=['GET'])
def generate_image_route():
    prompt = " "
    img_io = generate_image(prompt)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
