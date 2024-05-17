import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

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

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

def train_gan():
    batch_size = 64
    nz = 100
    epochs = 5
    lr = 0.0002
    beta1 = 0.5

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create models
    device = torch.device("cpu")
    netG = Generator(nz).to(device)
    netD = Discriminator().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    for epoch in range(epochs):
        for i, (data, _) in enumerate(dataloader):
            netD.zero_grad()
            real = data.to(device)
            batch_size = real.size(0)
            label = torch.full((batch_size,), 1.0, device=device)  # Ensure label is float

            # Forward pass real batch through Discriminator
            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)  # Fake labels are 0

            # Forward pass fake batch through Discriminator
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(1.0)  # Fake labels are real for generator cost

            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Loss D: {errD_real + errD_fake}, Loss G: {errG}')

        save_image(fake.data[:64], f'output_{epoch}.png', normalize=True)

    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')

if __name__ == '__main__':
    train_gan()
