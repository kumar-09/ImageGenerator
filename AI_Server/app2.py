from flask import Flask, send_file
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Generator model
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

nz = 100
device = torch.device("cpu")
netG = Generator(nz).to(device)
netG.load_state_dict(torch.load('generator.pth', map_location=device))
netG.eval()

def generate_image():
    noise = torch.randn(1, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_image = netG(noise).squeeze(0)
    
    transform = transforms.ToPILImage()
    image = transform(fake_image)
    
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return img_io

@app.route('/generate_image', methods=['GET'])
def generate_image_route():
    img_io = generate_image()
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
