import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("Preparing VQA data splits and image lists...")
    # Load training data
    train_df = pd.read_csv("data/data_train.csv")
    print(f"Original training data: {len(train_df)} samples")

    # Try stratified split, fall back to random split if it fails
    try:
        train_data, val_data = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df['answer'] if 'answer' in train_df.columns else None
        )
        print("✅ Stratified split successful.")
    except Exception as e:
        print(f"⚠️  Stratified split failed: {e}")
        print("Falling back to random split (no stratification)...")
        train_data, val_data = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

    # Load evaluation data for testing
    test_df = pd.read_csv("data/data_eval.csv")
    print(f"Test data: {len(test_df)} samples")

    # Save splits
    train_data.to_csv("data/train_split.csv", index=False)
    val_data.to_csv("data/val_split.csv", index=False)
    test_df.to_csv("data/test_split.csv", index=False)

    # Create corresponding image lists (using 'image_id' column)
    print("Creating image lists for splits...")
    train_images = train_data['image_id'].unique()
    val_images = val_data['image_id'].unique()
    test_images = test_df['image_id'].unique()

    with open("data/train_images_split.txt", "w") as f:
        for img in train_images:
            f.write(f"{img}\n")
    with open("data/val_images_split.txt", "w") as f:
        for img in val_images:
            f.write(f"{img}\n")
    with open("data/test_images_split.txt", "w") as f:
        for img in test_images:
            f.write(f"{img}\n")

    print(f"✅ Data splits created:")
    print(f"   Training: {len(train_data)} samples, {len(train_images)} unique images")
    print(f"   Validation: {len(val_data)} samples, {len(val_images)} unique images")
    print(f"   Testing: {len(test_df)} samples, {len(test_images)} unique images")
    print("Done! You can now run your training script.")

if __name__ == "__main__":
    main() 