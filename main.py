# main.py
from train import Trainer
from evaluate import Evaluator

def main():
    # Define file paths.
    train_csv = "data/data_train.csv"
    eval_csv = "data/data_eval.csv"
    img_dir = "data/images"
    answer_space_file = "data/answer_space.txt"  # Answer mapping file, if needed.
    
    # Define separate image list files.
    train_img_list = "data/train_images_list.txt"  # For training data filtering.
    test_img_list = "data/test_images_list.txt"    # For validation (used by Trainer).
    eval_img_list = "data/test_images_list.txt"      # For evaluation (used by Evaluator).

    print("Starting training...")
    # Create Trainer using train_img_list for training and test_img_list for validation.
    trainer = Trainer(train_csv, img_dir, answer_space_file, train_img_list, val_csv=eval_csv, test_img_list=test_img_list)
    trainer.train()

    print("Evaluating model...")
    # Create Evaluator using eval_img_list for the evaluation dataset.
    evaluator = Evaluator(trainer.model, eval_csv, img_dir, answer_space_file, eval_img_list=eval_img_list)
    evaluator.evaluate()
    evaluator.display_random_samples(num_samples=5)

if __name__ == "__main__":
    main()
