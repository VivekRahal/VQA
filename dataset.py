# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VQADataset(Dataset):
    def __init__(self, csv_file, img_dir, answer_space_file=None, img_list_file=None, transform=None):
        # Load CSV and print debug information.
        self.data = pd.read_csv(csv_file)
        print(f"[DEBUG] Loaded CSV '{csv_file}' with {len(self.data)} rows.")
        self.img_dir = img_dir

        # Filter by image list file if provided.
        if img_list_file is not None:
            with open(img_list_file, 'r') as f:
                valid_images = set(line.strip() for line in f.readlines())
            before_filter = len(self.data)
            self.data = self.data[self.data['image_id'].isin(valid_images)]
            after_filter = len(self.data)
            print(f"[DEBUG] After filtering with image list '{img_list_file}', rows reduced from {before_filter} to {after_filter}.")

        # Setup transformation.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        # Build vocabulary for questions.
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab()

        # Build answer mapping if an answer space file is provided.
        self.answer2idx = None
        if answer_space_file is not None:
            self.build_answer_mapping(answer_space_file)

    def build_vocab(self):
        idx = 2  # Start indexing new words from 2.
        for question in self.data['question']:
            for word in question.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    idx += 1

    def build_answer_mapping(self, answer_space_file):
        # Load and normalize answers from the mapping file.
        with open(answer_space_file, 'r') as f:
            answers = [line.strip().lower() for line in f.readlines()]
        self.answer2idx = {ans: i for i, ans in enumerate(answers)}
        print(f"[DEBUG] Built answer mapping with {len(self.answer2idx)} entries.")

    def tokenize(self, question):
        tokens = []
        for word in question.lower().split():
            tokens.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_id']
        full_img_path = os.path.join(self.img_dir, img_name)

        # Check if file exists; if not, try appending common extensions.
        if not os.path.exists(full_img_path):
            found = False
            for ext in [".jpg", ".png"]:
                candidate_path = full_img_path + ext
                if os.path.exists(candidate_path):
                    full_img_path = candidate_path
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Image file not found for {full_img_path} (tried .jpg and .png)")

        # Load the image.
        image = Image.open(full_img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Process question.
        question = row['question']
        tokens = self.tokenize(question)

        # Process answer with normalization.
        answer_text = str(row['answer']).strip().lower()
        # If the answer is composite (contains a comma), take the first part.
        if ',' in answer_text:
            answer_text = answer_text.split(',')[0].strip()

        if self.answer2idx is not None:
            if answer_text in self.answer2idx:
                answer = self.answer2idx[answer_text]
            else:
                print(f"[WARNING] Answer '{answer_text}' not found in mapping. Setting it to 0.")
                answer = 0
        else:
            answer = int(row['answer'])

        return image, torch.tensor(tokens), answer
