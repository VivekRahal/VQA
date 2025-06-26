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

def preprocess_question(question, vocab=None, tokenizer=None, max_length=32):
    """
    Preprocess a question string for the VQA model.
    If a Hugging Face tokenizer is provided (for BERT/ViT), use it.
    Otherwise, use vocab and whitespace tokenization.
    Args:
        question (str): The input question.
        vocab (dict, optional): Vocabulary mapping word to index (for non-BERT models).
        tokenizer (transformers.PreTrainedTokenizer, optional): Hugging Face tokenizer for BERT/ViT.
        max_length (int): Maximum sequence length (for padding/truncation).
    Returns:
        torch.Tensor: Token indices (1D tensor)
    """
    if tokenizer is not None:
        # Use Hugging Face tokenizer (for BERT/ViT)
        encoding = tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0)  # shape: (max_length,)
    elif vocab is not None:
        # Simple whitespace tokenization and vocab lookup
        tokens = question.lower().strip().split()
        indices = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
        # Pad or truncate
        if len(indices) < max_length:
            indices += [vocab.get('<PAD>', 0)] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        return torch.tensor(indices, dtype=torch.long)
    else:
        raise ValueError('Either vocab or tokenizer must be provided to preprocess_question.')
