from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import timm   # pip install timm
from fastapi.middleware.cors import CORSMiddleware
import os, requests

# Hugging Face model URL (after you upload it there)
MODEL_PATH = "best_swin_model.pth"
MODEL_URL = "https://huggingface.co/kirankumar274/best_swin_model.pth/resolve/main/best_swin_model.pth"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model architecture (same one used during training)
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=2)

# Load trained weights
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output = model(img_tensor)
        prob = F.softmax(output, dim=1)[0, 1].item()
        pred = "Cancer" if output.argmax(dim=1).item() == 1 else "Normal"
    return {"prediction": pred, "probability": prob}
