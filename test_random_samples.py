# test_random_samples.py
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import difflib  # For similarity calculation
import pickle   # >>> CHANGES FOR RANDOM SAMPLES: Import pickle to load saved vocabulary

from dataset import VQADataset
from model import VQAModel
from config import Config

# --- Configuration and Paths ---
EVAL_CSV = "data/data_eval.csv"     # Evaluation CSV file
IMG_DIR = "data/images"              # Directory with images
ANSWER_SPACE_FILE = "data/answer_space.txt"  # Answer mapping file (if used)
IMG_LIST_FILE = None  # Use None or a test image list file if available

# --- Load the Evaluation Dataset ---
eval_dataset = VQADataset(EVAL_CSV, IMG_DIR, ANSWER_SPACE_FILE, IMG_LIST_FILE)
config = Config()
vocab_size = len(eval_dataset.word2idx)
if eval_dataset.answer2idx is not None:
    config.num_classes = len(eval_dataset.answer2idx)

#  CHANGES FOR RANDOM SAMPLES: Load saved vocabulary mapping and assign it to eval_dataset
with open("vocab.pkl", "rb") as f:
    saved_word2idx = pickle.load(f)
print(f"[DEBUG] Loaded vocabulary with size: {len(saved_word2idx)}")
eval_dataset.word2idx = saved_word2idx
vocab_size = len(saved_word2idx)

# --- Initialize and Load the Model ---
model = VQAModel(vocab_size, config).to(config.device)
model.load_state_dict(torch.load("best_model.pth", map_location=config.device))
model.eval()

# --- Define Transformation (should match your training transform) ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# --- Helper Function: Tokenize a question ---
def tokenize_question(question: str, word2idx: dict, max_len: int = 20):
    tokens = []
    for word in question.lower().split():
        tokens.append(word2idx.get(word, word2idx["<UNK>"]))
    # Pad or truncate to fixed length
    if len(tokens) < max_len:
        tokens += [word2idx["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

# --- Helper Function: Compute similarity score ---
def similarity_score(predicted: str, actual: str) -> float:
    """Return similarity score between two strings (0.0 to 1.0)."""
    return difflib.SequenceMatcher(None, predicted, actual).ratio()

# --- Pick 10 Random Samples from the Evaluation Dataset ---
num_samples = 10
indices = random.sample(range(len(eval_dataset)), num_samples)

# --- Prepare to Collect Results ---
results = []  # List of tuples: (PIL image, question, predicted answer, actual answer, similarity)

# Invert answer mapping (if available)
if eval_dataset.answer2idx is not None:
    idx_to_answer = {v: k for k, v in eval_dataset.answer2idx.items()}
else:
    idx_to_answer = None

# --- Process Each Sample ---
for idx in indices:
    # Retrieve sample from dataset
    img_tensor, question_tokens, actual_answer = eval_dataset[idx]
    
    # Get original question text from the DataFrame
    question_text = eval_dataset.data.iloc[idx]["question"]
    
    # Convert image tensor back to PIL image for display:
    pil_image = transforms.ToPILImage()(img_tensor.cpu())
    
    # Prepare the question tensor for the model:
    # We use the original question text to tokenize using our helper
    tokens = tokenize_question(question_text, eval_dataset.word2idx, max_len=20)
    question_tensor = torch.tensor([tokens]).to(config.device)
    
    # Prepare the image tensor (if not already batched)
    img_input = img_tensor.unsqueeze(0).to(config.device)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(img_input, question_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_idx = predicted.item()
    
    # Map predicted and actual answer indices to text
    if idx_to_answer is not None:
        predicted_answer = idx_to_answer.get(predicted_idx, "Unknown")
        actual_answer_text = idx_to_answer.get(actual_answer, "Unknown")
    else:
        predicted_answer = str(predicted_idx)
        actual_answer_text = str(actual_answer)
    
    # Compute similarity score between predicted and actual answer strings
    sim_score = similarity_score(predicted_answer, actual_answer_text)
    
    results.append((pil_image, question_text, predicted_answer, actual_answer_text, sim_score))

# --- Display the Results ---
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

for i, (img, question, pred_ans, actual_ans, sim) in enumerate(results):
    axes[i].imshow(np.array(img))
    axes[i].axis("off")
    axes[i].set_title(f"Q: {question}\nPred: {pred_ans}\nActual: {actual_ans}\nSim: {sim*100:.1f}%", fontsize=10)
plt.tight_layout()
plt.show()
