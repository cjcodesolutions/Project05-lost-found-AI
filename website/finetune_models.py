"""
Fine-tuning Script for Lost & Found AI Models
Run this script to fine-tune both Sentence-BERT and CLIP models on Kaggle datasets

Before running:
1. Download datasets from Kaggle:
   - Delhi Metro Lost & Found: https://www.kaggle.com/datasets/...
   - Roboflow Lost & Found: https://www.kaggle.com/datasets/...
   
2. Place datasets in the following structure:
   datasets/
   ├── delhi_metro_lost_found.csv
   ├── roboflow_images/
   │   ├── phone_001.jpg
   │   ├── wallet_001.jpg
   │   └── ...
   └── image_labels.csv

3. Install required packages:
   pip install sentence-transformers torch torchvision clip pandas numpy scikit-learn pillow
"""

import os
import sys
import argparse
from similarity_service import FineTunedSimilarityService
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_image_labels_csv(image_dir, output_csv):
    """
    Helper function to create image_labels.csv from Roboflow dataset
    
    This assumes your images are organized like:
    roboflow_images/
    ├── phone/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── wallet/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── ...
    """
    import pandas as pd
    
    data = []
    
    for category in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category)
        
        if not os.path.isdir(category_path):
            continue
        
        for image_file in os.listdir(category_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({
                    'image_path': os.path.join(category, image_file),
                    'category': category,
                    'label': category
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logger.info(f"Created {output_csv} with {len(df)} images")
    
    return output_csv


def verify_datasets(dataset_paths):
    """Verify that all required datasets exist"""
    missing = []
    
    if not os.path.exists(dataset_paths['text_csv']):
        missing.append(f"Text dataset: {dataset_paths['text_csv']}")
    
    if not os.path.exists(dataset_paths['image_dir']):
        missing.append(f"Image directory: {dataset_paths['image_dir']}")
    
    if not os.path.exists(dataset_paths['image_labels']):
        logger.warning(f"Image labels not found, will attempt to create from {dataset_paths['image_dir']}")
        try:
            prepare_image_labels_csv(dataset_paths['image_dir'], dataset_paths['image_labels'])
        except Exception as e:
            missing.append(f"Could not create image labels: {e}")
    
    if missing:
        logger.error("Missing datasets:")
        for item in missing:
            logger.error(f"  - {item}")
        return False
    
    return True


def download_kaggle_datasets():
    """
    Instructions for downloading Kaggle datasets
    """
    print("\n" + "="*80)
    print("KAGGLE DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("""
To download the required datasets from Kaggle:

1. Install Kaggle API:
   pip install kaggle

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click 'Create New API Token'
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)

3. Download Delhi Metro Lost & Found Dataset:
   kaggle datasets download -d <dataset-name> -p datasets/
   unzip datasets/<dataset-name>.zip -d datasets/

4. Download Roboflow Lost & Found Images:
   kaggle datasets download -d <roboflow-dataset> -p datasets/roboflow_images/
   unzip datasets/<roboflow-dataset>.zip -d datasets/roboflow_images/

5. Alternative: Manually download from Kaggle website
   - Visit the dataset pages
   - Click 'Download'
   - Extract to the datasets/ folder

SUGGESTED KAGGLE DATASETS:
- Delhi Metro Lost & Found (text descriptions)
- Roboflow Lost & Found Items (images)
- Google Open Images (subset of common objects)
- Product descriptions from e-commerce datasets (for additional text data)
    """)
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Lost & Found AI models')
    parser.add_argument('--text-csv', type=str, 
                       default='datasets/delhi_metro_lost_found.csv',
                       help='Path to text dataset CSV')
    parser.add_argument('--image-dir', type=str,
                       default='datasets/roboflow_images/',
                       help='Path to image directory')
    parser.add_argument('--image-labels', type=str,
                       default='datasets/image_labels.csv',
                       help='Path to image labels CSV')
    parser.add_argument('--download-instructions', action='store_true',
                       help='Show Kaggle download instructions')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip dataset verification (not recommended)')
    
    args = parser.parse_args()
    
    if args.download_instructions:
        download_kaggle_datasets()
        return
    
    # Prepare dataset paths
    dataset_paths = {
        'text_csv': args.text_csv,
        'image_dir': args.image_dir,
        'image_labels': args.image_labels
    }
    
    # Verify datasets exist
    if not args.skip_verification:
        logger.info("Verifying datasets...")
        if not verify_datasets(dataset_paths):
            logger.error("Dataset verification failed. Use --download-instructions for help.")
            sys.exit(1)
        logger.info("Dataset verification passed!")
    
    # Start fine-tuning
    logger.info("="*80)
    logger.info("STARTING FINE-TUNING PROCESS")
    logger.info("="*80)
    logger.info(f"Text dataset: {dataset_paths['text_csv']}")
    logger.info(f"Image directory: {dataset_paths['image_dir']}")
    logger.info(f"Image labels: {dataset_paths['image_labels']}")
    logger.info("="*80)
    
    try:
        # Initialize service with fine-tuning enabled
        service = FineTunedSimilarityService(
            fine_tune=True,
            dataset_paths=dataset_paths
        )
        
        logger.info("="*80)
        logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("Models saved to:")
        logger.info(f"  - Sentence-BERT: {service.text_model_path}")
        logger.info(f"  - CLIP: {service.clip_model_path}.pt")
        logger.info("\nYou can now use these fine-tuned models in your application.")
        logger.info("Set fine_tune=False in similarity_service.py to use the fine-tuned models.")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()