import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# DATASET
# -------------------------

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

        self.pairs = []

        for img_name in os.listdir(image_dir):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(image_dir, img_name)
                mask_name = img_name.replace(".jpg", "_segmentation.png")
                mask_path = os.path.join(mask_dir, mask_name)

                if os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))

        print(f"✅ Total valid pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

# -------------------------
# U-NET MODEL (SAFE TO IMPORT)
# -------------------------

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = block(3,64)
        self.enc2 = block(64,128)
        self.enc3 = block(128,256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = block(256,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.dec3 = block(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = block(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = block(128,64)

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),e3],1))
        d2 = self.dec2(torch.cat([self.up2(d3),e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2),e1],1))

        return torch.sigmoid(self.final(d1))

# -------------------------
# TRAINING BLOCK (RUNS ONLY WHEN EXECUTED DIRECTLY)
# -------------------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISICDataset(
        "datasets/isic2018/images",
        "datasets/isic2018/masks"
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    # -------------------------
    # RESUME LOGIC
    # -------------------------

    start_epoch = 0
    checkpoint_path = "models/unet_checkpoint.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1

        print(f"🔁 Resuming from epoch {start_epoch}")

    print("🚀 Training Started...")

    # -------------------------
    # TRAIN LOOP
    # -------------------------

    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss = 0

        print(f"\n🔵 Epoch {epoch+1}/{epochs}")

        for i, (images, masks) in enumerate(loader):

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}/{len(loader)} Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

        os.makedirs("models", exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

        print(f"💾 Saved checkpoint at epoch {epoch+1}")

    # -------------------------
    # SAVE FINAL MODEL
    # -------------------------

    torch.save(model.state_dict(), "models/unet_model.pth")

    print("🎉 Final model saved!")