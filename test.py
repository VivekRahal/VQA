import pandas as pd
import os

# Load the CSV file
train_csv_path = "data/data_train.csv"
df_train = pd.read_csv(train_csv_path)

# Display the first few rows to inspect the data manually
print("Data Preview:")
print(df_train.head())

# Check the columns in the dataset
print("\nColumns in the dataset:")
print(df_train.columns)

# Check for missing values in each column
print("\nMissing values per column:")
print(df_train.isnull().sum())

# Check for any empty or blank strings in 'question'
missing_questions = df_train['question'].apply(lambda x: isinstance(x, str) and x.strip() == "").sum()
print("\nEmpty questions:", missing_questions)

# Check for any unexpected values in the 'answer' column
print("\nAnswer distribution:")
print(df_train['answer'].value_counts())

# Optionally, check for balance by calculating percentages
total_samples = len(df_train)
answer_distribution = df_train['answer'].value_counts(normalize=True) * 100
print("\nAnswer distribution (percentage):")
print(answer_distribution)

# Verify that corresponding image files exist (assuming images are stored in "data/images")
img_dir = "data/images"
missing_images = 0
for img_name in df_train['image_id']:
    # If your CSV does not include the file extension, try adding common ones:
    found = False
    for ext in [".jpg", ".png"]:
        full_img_path = os.path.join(img_dir, img_name + ext)
        if os.path.exists(full_img_path):
            found = True
            break
    if not found:
        print(f"Missing image file for: {img_name}")
        missing_images += 1

print(f"\nTotal missing images: {missing_images}")


import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Define paths
csv_path = "data/data_train.csv"  # Adjust if needed
img_dir = "data/images"           # Folder containing your images

# Load the dataset
df = pd.read_csv(csv_path)

# Select 5 random samples
sample_df = df.sample(n=5, random_state=42)

# Create a figure to display images
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (_, row) in zip(axes, sample_df.iterrows()):
    # Build image path; adjust extension if needed (e.g., ".jpg")
    img_path = os.path.join(img_dir, row["image_id"])
    if not os.path.exists(img_path):
        # If no file without extension, try adding a common extension
        img_path = img_path + ".png"  # or change to ".png" if needed
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        continue

    # Display the image with question and answer as title
    ax.imshow(img)
    ax.axis("off")
    title_text = f"Q: {row['question']}\nA: {row['answer']}"
    ax.set_title(title_text, fontsize=9)
    
plt.tight_layout()
plt.show()
