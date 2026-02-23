import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers import models as st_models
import requests
from PIL import Image
import io
import torch
import clip
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTunedSimilarityService:
    def __init__(self, fine_tune=False, dataset_paths=None):
        """
        Initialize the similarity service with optional fine-tuning
        
        Args:
            fine_tune: Whether to fine-tune models on initialization
            dataset_paths: Dict with paths to training datasets
                {
                    'text_csv': 'path/to/delhi_metro_lost_found.csv',
                    'image_dir': 'path/to/roboflow_images/',
                    'image_labels': 'path/to/image_labels.csv'
                }
        """
        self.text_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Model save paths
        self.text_model_path = 'models/finetuned_sentence_bert'
        self.clip_model_path = 'models/finetuned_clip'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        if fine_tune and dataset_paths:
            self._fine_tune_models(dataset_paths)
        else:
            self._load_models()
    
    def _load_models(self):
        """Load pre-trained or fine-tuned models"""
        try:
            # Try to load fine-tuned Sentence-BERT first
            if os.path.exists(self.text_model_path):
                logger.info("Loading fine-tuned SentenceBERT model...")
                self.text_model = SentenceTransformer(self.text_model_path)
            else:
                logger.info("Loading pre-trained SentenceBERT model...")
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load CLIP (note: CLIP fine-tuning requires custom implementation)
            logger.info(f"Loading CLIP model on device: {self.device}")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Try to load fine-tuned CLIP weights if available
            if os.path.exists(f"{self.clip_model_path}.pt"):
                logger.info("Loading fine-tuned CLIP weights...")
                checkpoint = torch.load(f"{self.clip_model_path}.pt", map_location=self.device)
                self.clip_model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise e
    
    def _prepare_text_training_data(self, csv_path):
        """
        Prepare training data for Sentence-BERT from Delhi Metro dataset
        
        Expected CSV columns: item, description, category, status, location, date_reported
        """
        logger.info(f"Preparing text training data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Create training examples (pairs of similar items)
        train_examples = []
        
        # Group by category to create positive pairs
        for category in df['category'].unique():
            if pd.isna(category):
                continue
                
            category_items = df[df['category'] == category]
            
            # Create positive pairs (items in same category)
            for i in range(len(category_items)):
                for j in range(i + 1, min(i + 5, len(category_items))):  # Limit pairs per item
                    item1 = category_items.iloc[i]
                    item2 = category_items.iloc[j]
                    
                    text1 = f"{item1['item']} {item1['description']} {item1['category']}"
                    text2 = f"{item2['item']} {item2['description']} {item2['category']}"
                    
                    # Clean texts
                    text1 = str(text1).strip().lower()
                    text2 = str(text2).strip().lower()
                    
                    if text1 and text2:
                        # Positive pair (similar items, label=1.0)
                        train_examples.append(InputExample(texts=[text1, text2], label=1.0))
        
        # Create negative pairs (items from different categories)
        categories = df['category'].unique()
        categories = [c for c in categories if not pd.isna(c)]
        
        for i in range(min(len(train_examples), 1000)):  # Limit negative pairs
            cat1, cat2 = np.random.choice(categories, 2, replace=False)
            
            item1 = df[df['category'] == cat1].sample(1).iloc[0]
            item2 = df[df['category'] == cat2].sample(1).iloc[0]
            
            text1 = f"{item1['item']} {item1['description']} {item1['category']}"
            text2 = f"{item2['item']} {item2['description']} {item2['category']}"
            
            text1 = str(text1).strip().lower()
            text2 = str(text2).strip().lower()
            
            if text1 and text2:
                # Negative pair (different items, label=0.0)
                train_examples.append(InputExample(texts=[text1, text2], label=0.0))
        
        logger.info(f"Created {len(train_examples)} training examples")
        return train_examples
    
    def _fine_tune_sentence_bert(self, train_examples):
        """Fine-tune Sentence-BERT on lost and found data"""
        logger.info("Fine-tuning Sentence-BERT...")
        
        # Initialize base model
        word_embedding_model = st_models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
        pooling_model = st_models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode='mean'
        )
        self.text_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Use CosineSimilarityLoss for training
        train_loss = losses.CosineSimilarityLoss(self.text_model)
        
        # Training parameters
        num_epochs = 3
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        
        # Fine-tune the model
        self.text_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=self.text_model_path,
            show_progress_bar=True
        )
        
        logger.info(f"Fine-tuned model saved to {self.text_model_path}")
    
    def _prepare_image_training_data(self, image_dir, labels_csv):
        """
        Prepare training data for CLIP from Roboflow dataset
        
        Expected CSV columns: image_path, category, label
        """
        logger.info(f"Preparing image training data from {image_dir}")
        
        df = pd.read_csv(labels_csv)
        
        # Create training pairs
        training_data = []
        
        for category in df['category'].unique():
            if pd.isna(category):
                continue
            
            category_images = df[df['category'] == category]
            
            for idx, row in category_images.iterrows():
                image_path = os.path.join(image_dir, row['image_path'])
                if os.path.exists(image_path):
                    # Create text descriptions for the category
                    text_descriptions = [
                        f"a photo of a {category}",
                        f"{category} item",
                        f"lost {category}",
                        f"found {category}"
                    ]
                    training_data.append({
                        'image_path': image_path,
                        'texts': text_descriptions,
                        'category': category
                    })
        
        logger.info(f"Prepared {len(training_data)} image-text pairs")
        return training_data
    
    def _fine_tune_clip(self, training_data):
        """
        Fine-tune CLIP model on lost and found images
        
        Note: This is a simplified fine-tuning approach
        """
        logger.info("Fine-tuning CLIP...")
        
        # Prepare optimizer
        optimizer = torch.optim.Adam(self.clip_model.parameters(), lr=1e-5)
        
        # Training loop
        num_epochs = 3
        batch_size = 8
        
        self.clip_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # Load and preprocess images
                images = []
                texts = []
                
                for item in batch:
                    try:
                        image = Image.open(item['image_path']).convert('RGB')
                        image = self.clip_preprocess(image)
                        images.append(image)
                        
                        # Use one text description per image
                        texts.extend(item['texts'][:1])
                    except Exception as e:
                        logger.warning(f"Error loading image {item['image_path']}: {e}")
                        continue
                
                if not images:
                    continue
                
                # Stack images and tokenize texts
                images = torch.stack(images).to(self.device)
                texts = clip.tokenize(texts).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                image_features = self.clip_model.encode_image(images)
                text_features = self.clip_model.encode_text(texts)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity and loss (contrastive loss)
                logits = image_features @ text_features.t() * 100
                labels = torch.arange(len(images)).to(self.device)
                
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save fine-tuned model
        torch.save({
            'model_state_dict': self.clip_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{self.clip_model_path}.pt")
        
        logger.info(f"Fine-tuned CLIP model saved to {self.clip_model_path}.pt")
        self.clip_model.eval()
    
    def _fine_tune_models(self, dataset_paths):
        """Fine-tune both models"""
        logger.info("Starting fine-tuning process...")
        
        # Fine-tune Sentence-BERT
        if 'text_csv' in dataset_paths and os.path.exists(dataset_paths['text_csv']):
            train_examples = self._prepare_text_training_data(dataset_paths['text_csv'])
            self._fine_tune_sentence_bert(train_examples)
        else:
            logger.warning("Text dataset not found, loading pre-trained Sentence-BERT")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load CLIP first
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Fine-tune CLIP
        if all(k in dataset_paths for k in ['image_dir', 'image_labels']):
            if os.path.exists(dataset_paths['image_dir']) and os.path.exists(dataset_paths['image_labels']):
                training_data = self._prepare_image_training_data(
                    dataset_paths['image_dir'],
                    dataset_paths['image_labels']
                )
                self._fine_tune_clip(training_data)
            else:
                logger.warning("Image dataset not found, using pre-trained CLIP")
        else:
            logger.warning("Image dataset paths not provided, using pre-trained CLIP")
        
        logger.info("Fine-tuning complete!")
    
    # Keep all existing methods from original similarity_service.py
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using fine-tuned Sentence-BERT"""
        try:
            if not text or not text.strip():
                return None
            
            text = text.strip().lower()
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def get_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """Get image embedding using fine-tuned CLIP"""
        try:
            if not image_url:
                return None
            
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            image = image.convert('RGB')
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None
    
    def get_text_image_similarity(self, text: str, image_url: str) -> float:
        """Calculate similarity between text and image using fine-tuned CLIP"""
        try:
            if not text or not image_url:
                return 0.0
            
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([text.strip()]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                return float(similarity.cpu().numpy()[0])
        
        except Exception as e:
            logger.error(f"Error calculating text-image similarity: {e}")
            return 0.0
    
    def create_structured_text(self, item_data: Dict) -> str:
        """Create a structured text description from item data"""
        parts = []
        
        if item_data.get('whatLost'):
            parts.append(item_data['whatLost'])
        elif item_data.get('whatFound'):
            parts.append(item_data['whatFound'])
        
        if item_data.get('category'):
            parts.append(item_data['category'])
        if item_data.get('brand'):
            parts.append(f"{item_data['brand']} brand")
        
        if item_data.get('primaryColor'):
            parts.append(f"{item_data['primaryColor']} color")
        if item_data.get('secondaryColor'):
            parts.append(f"with {item_data['secondaryColor']}")
        
        if item_data.get('additionalInfo'):
            parts.append(item_data['additionalInfo'])
        
        if item_data.get('whereLost'):
            parts.append(f"lost at {item_data['whereLost']}")
        elif item_data.get('whereFound'):
            parts.append(f"found at {item_data['whereFound']}")
        
        return " ".join(parts).strip()
    
    def calculate_multimodal_similarity(self, 
                                      query_text: str, 
                                      query_image_url: Optional[str],
                                      target_text: str, 
                                      target_image_url: Optional[str]) -> float:
        """Calculate comprehensive similarity using fine-tuned models"""
        
        similarities = []
        weights = []
        
        # Text-to-text similarity
        if query_text and target_text:
            query_embedding = self.get_text_embedding(query_text)
            target_embedding = self.get_text_embedding(target_text)
            
            if query_embedding is not None and target_embedding is not None:
                text_sim = cosine_similarity([query_embedding], [target_embedding])[0][0]
                similarities.append(text_sim)
                weights.append(0.4)
        
        # Image-to-image similarity
        if query_image_url and target_image_url:
            query_img_embedding = self.get_image_embedding(query_image_url)
            target_img_embedding = self.get_image_embedding(target_image_url)
            
            if query_img_embedding is not None and target_img_embedding is not None:
                img_sim = cosine_similarity([query_img_embedding], [target_img_embedding])[0][0]
                similarities.append(img_sim)
                weights.append(0.4)
        
        # Cross-modal similarities
        if query_text and target_image_url:
            cross_sim1 = self.get_text_image_similarity(query_text, target_image_url)
            similarities.append(cross_sim1)
            weights.append(0.1)
        
        if query_image_url and target_text:
            cross_sim2 = self.get_text_image_similarity(target_text, query_image_url)
            similarities.append(cross_sim2)
            weights.append(0.1)
        
        if similarities:
            weighted_sum = sum(s * w for s, w in zip(similarities, weights))
            total_weight = sum(weights)
            final_score = weighted_sum / total_weight if total_weight > 0 else 0
            
            completeness_boost = 1.0
            if query_image_url and query_text and target_image_url and target_text:
                completeness_boost = 1.1
            
            return min(final_score * completeness_boost, 1.0)
        
        return 0.0
    
    def find_similar_items_structured(self, 
                                    query_image_url: Optional[str],
                                    query_data: Dict,
                                    database_items: List[Dict],
                                    top_k: int = 10,
                                    min_score: float = 0.5) -> List[Tuple[Dict, float]]:
        """Find similar items using fine-tuned models"""
        
        if not database_items:
            return []
        
        query_text = self.create_structured_text(query_data)
        logger.info(f"Query text: {query_text}")
        logger.info(f"Query has image: {bool(query_image_url)}")
        logger.info(f"Minimum similarity score threshold: {min_score}")
        
        results = []
        
        for item in database_items:
            try:
                target_text = self.create_structured_text(item)
                target_image_url = item.get('imageUrl')
                
                similarity_score = self.calculate_multimodal_similarity(
                    query_text=query_text,
                    query_image_url=query_image_url,
                    target_text=target_text,
                    target_image_url=target_image_url
                )
                
                # Apply category bonus
                if (query_data.get('category') and item.get('category') and 
                    query_data['category'].lower() == item.get('category', '').lower()):
                    similarity_score *= 1.2
                
                # Apply color bonus
                query_color = query_data.get('primaryColor', '').lower()
                target_color = item.get('primaryColor', '').lower()
                if query_color and target_color and query_color == target_color:
                    similarity_score *= 1.1
                
                # Brand bonus
                query_brand = query_data.get('brand', '').lower()
                target_brand = item.get('brand', '').lower()
                if query_brand and target_brand and query_brand == target_brand:
                    similarity_score *= 1.15
                
                similarity_score = min(similarity_score, 1.0)
                
                if similarity_score >= min_score:
                    results.append((item, similarity_score))
                    logger.info(f"Item {item.get('_id', 'unknown')} included with score: {similarity_score:.3f}")
                    
            except Exception as e:
                logger.error(f"Error processing item {item.get('_id', 'unknown')}: {e}")
                continue
        
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(results)} items with similarity >= {min_score}")
        return results[:top_k]
    
    def find_similar_items(self, 
                          query_image_url: Optional[str],
                          query_text: str,
                          database_items: List[Dict],
                          top_k: int = 10,
                          min_score: float = 0.5) -> List[Tuple[Dict, float]]:
        """Legacy method for backward compatibility"""
        
        query_data = {
            'whatLost': query_text,
            'category': '',
            'brand': '',
            'primaryColor': '',
            'additionalInfo': ''
        }
        
        return self.find_similar_items_structured(
            query_image_url=query_image_url,
            query_data=query_data,
            database_items=database_items,
            top_k=top_k,
            min_score=min_score
        )


# Initialize with fine-tuning (set fine_tune=True when you have datasets ready)
# For production, use fine_tune=False to load pre-trained/fine-tuned models
similarity_service = FineTunedSimilarityService(
    fine_tune=False,  # Set to True when you want to fine-tune
    dataset_paths={
        'text_csv': 'datasets/delhi_metro_lost_found.csv',
        'image_dir': 'datasets/roboflow_images/',
        'image_labels': 'datasets/image_labels.csv'
    }
)