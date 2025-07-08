import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.utils as vutils
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
IMAGE_SIZE = 96
UPSCALE_FACTOR = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "DIV2K/DIV2K_train_HR"
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)
class Generator(nn.Module):
    def __init__(self, scale_factor = 4, num_residuals = 16):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residuals)]
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        upsample_layers = []
        for _ in range(int(scale_factor / 2)):
            upsample_layers += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.block3 = nn.Conv2d(64, 3, 9, 1, 4)
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.residual_blocks(x1)
        x3 = self.block2(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.block3(x)
class SRGANDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.hr_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE), # Use CenterCrop to ensure consistent size
            transforms.ToTensor(),
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE // UPSCALE_FACTOR, IMAGE_SIZE // UPSCALE_FACTOR), interpolation=Image.BICUBIC), # Resize to a fixed size
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        hr_image = Image.open(self.files[idx]).convert("RGB")
        lr_image = self.lr_transform(hr_image)
        hr_image = self.hr_transform(hr_image)
        return lr_image, hr_image
pixel_criterion = nn.MSELoss()
generator = Generator(scale_factor=UPSCALE_FACTOR).to(DEVICE)
optimizer = optim.Adam(generator.parameters(), lr=LR)
def train():
    dataset = SRGANDataset(DATA_DIR)
    if len(dataset) == 0:
        print(f"No image files (.png or .jpg) found in {DATA_DIR}. Please make sure the dataset is downloaded and extracted correctly.")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        generator.train()
        total_loss = 0
        for batch, (lr, hr) in enumerate(dataloader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            sr = generator(lr)
            loss = pixel_criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch % 100 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Batch [{batch}], Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss: {total_loss / len(dataloader):.4f}")
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
            vutils.save_image(sr, f"sample_epoch_{epoch}.png")
    torch.save(generator.state_dict(), "generator.pth")
    print("Training complete.")
if __name__ == "__main__":
    train()