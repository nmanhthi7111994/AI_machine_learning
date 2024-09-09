import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
path = "/content/drive/MyDrive/data2/runs/detect/train/weights/runs/detect/train/weights/runs/detect/train/weights/runs/detect/train/weights/runs/detect/train/weights/runs/detect/train/weights/last.pt"
model = YOLO(path)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Using learning rate decay with StepLR
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Decreasing learning rate after 100 epochs
epochs = 100
for epoch in range(epochs):
    # Training model
    optimizer.zero_grad()
    results = model.train(data="/content/drive/MyDrive/data2/mydataset.yaml")

    # Update learning rate
    scheduler.step()

    # Print info
    print('Epoch:', epoch, 'Loss:', results['loss'], 'Learning Rate:', scheduler.get_last_lr()[0])
