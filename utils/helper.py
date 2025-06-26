# utils/helper.py
import torch
import pickle
import pandas as pd

def vqa_collate_fn(batch):
    """
    Collate function to pad question token sequences in a batch.
    Each batch element is a tuple: (image, question_tokens, answer)
    """
    images, questions, answers = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(q) for q in questions]
    max_len = max(lengths)
    padded_questions = []
    for q in questions:
        # Pad with zeros (the index for "<PAD>") so that all sequences have the same length.
        padded = q.tolist() + [0] * (max_len - len(q))
        padded_questions.append(padded)
    padded_questions = torch.tensor(padded_questions)
    answers = torch.tensor(answers)
    return images, padded_questions, answers

def load_vocab(filepath):
    """
    Load vocabulary dictionary from a pickle file.
    Args:
        filepath (str): Path to the vocab pickle file.
    Returns:
        dict: Vocabulary mapping word to index.
    """
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def build_answer_mapping(csv_file):
    """
    Build answer-to-index mapping from a CSV file.
    Args:
        csv_file (str): Path to the CSV file containing an 'answer' column.
    Returns:
        dict: Mapping from answer string to integer index.
    """
    df = pd.read_csv(csv_file)
    answers = sorted(df['answer'].unique())
    return {ans: idx for idx, ans in enumerate(answers)}
