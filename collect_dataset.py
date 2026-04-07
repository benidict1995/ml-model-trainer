import os
from icrawler.builtin import BingImageCrawler

# =====================
# CONFIGURATION
# =====================

# List of classes (each will become a folder)
breeds = [
    "husky dog",
    "golden retriever dog",
    "labrador retriever dog",
    "german shepherd dog",
    "bulldog dog"
]

# Base directory where dataset will be stored
BASE_DIR = "dataset/dog_breeds"

# Number of images to download per class
IMAGES_PER_CLASS = 300

# =====================
# DATA COLLECTION
# =====================

for breed in breeds:

    # Convert label to folder-friendly name
    folder_name = breed.replace(" ", "_")

    # Full save path for this class
    save_path = os.path.join(BASE_DIR, folder_name)

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading images for: {breed}")

    # Initialize Bing image crawler
    crawler = BingImageCrawler(storage={'root_dir': save_path})

    # Start crawling images
    crawler.crawl(
        keyword=breed,
        max_num=IMAGES_PER_CLASS
    )

print("Dataset collection complete!")