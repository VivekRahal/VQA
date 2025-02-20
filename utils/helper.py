# utils/helper.py
import torch

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
