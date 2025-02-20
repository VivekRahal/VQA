# app.py
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch
import torchvision.transforms as transforms

# Import our dataset, model, and config definitions
from dataset import VQADataset
from model import VQAModel
from config import Config

app = FastAPI(title="VQA FastAPI App")

# --- Configuration and Model Setup ---

# Paths to your dataset files (adjust these paths as needed)
TRAIN_CSV = "data/data_train.csv"
IMG_DIR = "data/images"
ANSWER_SPACE_FILE = "data/answer_space.txt"  # if using answer mapping
IMG_LIST_FILE = "data/train_images_list.txt"  # optional if you filter images

# Load the training dataset to rebuild vocabulary and answer mapping
train_dataset = VQADataset(TRAIN_CSV, IMG_DIR, ANSWER_SPACE_FILE, IMG_LIST_FILE)
vocab_size = len(train_dataset.word2idx)
config = Config()
if train_dataset.answer2idx is not None:
    config.num_classes = len(train_dataset.answer2idx)

# Initialize the model and load the trained weights
model = VQAModel(vocab_size, config).to(config.device)
model.load_state_dict(torch.load("best_model.pth", map_location=config.device))
model.eval()

# Define the image transformation (should match the one used during training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# --- Helper Function ---

def tokenize_question(question: str, word2idx: dict):
    """
    Tokenizes the question into integers using the word2idx mapping.
    """
    tokens = []
    for word in question.lower().split():
        tokens.append(word2idx.get(word, word2idx["<UNK>"]))
    return tokens

# --- FastAPI Endpoints ---

@app.post("/predict")
async def predict(image: UploadFile = File(...), question: str = Form(...)):
    """
    Endpoint to predict the answer to a question based on an uploaded image.
    Expects an image file and a question (as form data).
    """
    try:
        # Read and process the uploaded image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(pil_image).unsqueeze(0)  # add batch dimension

        # Tokenize and pad the question
        tokens = tokenize_question(question, train_dataset.word2idx)
        max_len = 20  # you can adjust this fixed length based on your training settings
        if len(tokens) < max_len:
            tokens = tokens + [train_dataset.word2idx["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        tokens_tensor = torch.tensor([tokens]).to(config.device)  # shape: (1, max_len)

        # Run the model forward pass
        with torch.no_grad():
            outputs = model(img_tensor.to(config.device), tokens_tensor)
            _, predicted = torch.max(outputs, 1)

        # Map the predicted index back to an answer string
        if train_dataset.answer2idx is not None:
            # Invert the answer mapping
            idx_to_answer = {v: k for k, v in train_dataset.answer2idx.items()}
            answer_text = idx_to_answer.get(predicted.item(), "Unknown")
        else:
            answer_text = str(predicted.item())

        return {"question": question, "answer": answer_text}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Main Block to Run the App ---

if __name__ == "__main__":
    # Run with: uvicorn app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
