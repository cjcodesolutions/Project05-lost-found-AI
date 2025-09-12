import torch
import clip
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import logging
from typing import List, Dict, Tuple, Optional
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityService:
    def __init__(self):
        """Initialize CLIP and SentenceBERT models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model for image-text similarity
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
        
        # Load SentenceBERT for text similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceBERT model: {e}")
            self.sentence_model = None
    
    def get_image_embedding(self, image_input) -> Optional[np.ndarray]:
        """
        Get CLIP embedding for an image
        Args:
            image_input: Can be PIL Image, image URL, or file path
        Returns:
            numpy array of image embedding or None if failed
        """
        if self.clip_model is None:
            logger.error("CLIP model not available")
            return None
        
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if image_input.startswith('http'):
                    # URL - add timeout and error handling
                    try:
                        response = requests.get(image_input, timeout=30)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    except Exception as e:
                        logger.error(f"Failed to download image from {image_input}: {e}")
                        return None
                else:
                    # File path
                    image = Image.open(image_input).convert('RGB')
            else:
                # PIL Image
                image = image_input.convert('RGB')
            
            # Preprocess and encode
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using SentenceBERT (more reliable for text similarity)
        """
        if self.sentence_model is None:
            logger.error("SentenceBERT model not available")
            return None
            
        try:
            # Clean and normalize text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return None
                
            sentence_embedding = self.sentence_model.encode([cleaned_text])
            return sentence_embedding.flatten()
        except Exception as e:
            logger.error(f"Error with SentenceBERT encoding: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def extract_item_features(self, item: Dict) -> Dict[str, str]:
        """Extract key features from an item for comparison"""
        features = {}
        
        # Main item description
        features['description'] = self.clean_text(
            f"{item.get('whatLost', '')} {item.get('whatFound', '')}"
        )
        
        # Category
        features['category'] = self.clean_text(item.get('category', ''))
        
        # Brand
        features['brand'] = self.clean_text(item.get('brand', ''))
        
        # Colors
        features['color'] = self.clean_text(
            f"{item.get('primaryColor', '')} {item.get('secondaryColor', '')}"
        )
        
        # Location
        features['location'] = self.clean_text(
            f"{item.get('whereLost', '')} {item.get('whereFound', '')} {item.get('locationName', '')}"
        )
        
        # Additional info
        features['additional'] = self.clean_text(item.get('additionalInfo', ''))
        
        # Combine all text features
        all_text = ' '.join([v for v in features.values() if v])
        features['combined'] = all_text
        
        return features
    
    def extract_dominant_colors(self, image_input) -> List[str]:
        """Extract dominant colors from an image"""
        try:
            if isinstance(image_input, str) and image_input.startswith('http'):
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = image_input.convert('RGB')
            
            # Resize for faster processing
            image = image.resize((150, 150))
            
            # Convert to numpy array
            pixels = np.array(image).reshape(-1, 3)
            
            # Simple color clustering (you could use sklearn.cluster.KMeans for better results)
            unique_colors = []
            for pixel in pixels[::100]:  # Sample every 100th pixel
                # Convert RGB to color name (simplified)
                r, g, b = pixel
                
                if r > 200 and g > 200 and b > 200:
                    color_name = "white"
                elif r < 50 and g < 50 and b < 50:
                    color_name = "black"
                elif r > g and r > b:
                    color_name = "red"
                elif g > r and g > b:
                    color_name = "green"
                elif b > r and b > g:
                    color_name = "blue"
                elif r > 150 and g > 150 and b < 100:
                    color_name = "yellow"
                elif r > 150 and g < 100 and b > 150:
                    color_name = "purple"
                elif r > 100 and g > 50 and b < 50:
                    color_name = "brown"
                else:
                    color_name = "gray"
                
                if color_name not in unique_colors:
                    unique_colors.append(color_name)
                
                if len(unique_colors) >= 3:  # Limit to top 3 colors
                    break
            
            return unique_colors
            
        except Exception as e:
            logger.error(f"Error extracting colors: {e}")
            return []
    
    def calculate_text_similarity_detailed(self, query_item: Dict, db_item: Dict) -> float:
        """Calculate detailed text similarity focusing on key factors"""
        query_features = self.extract_item_features(query_item)
        db_features = self.extract_item_features(db_item)
        
        similarities = []
        weights = {
            'description': 0.25,  # What the item is
            'category': 0.2,     # Item category  
            'color': 0.35,       # INCREASED: Color information (most important)
            'brand': 0.15,       # Brand if available
            'location': 0.05     # Location similarity
        }
        
        # Check for color mismatch penalty
        color_penalty = 0
        if query_features['color'] and db_features['color']:
            query_colors = set(query_features['color'].split())
            db_colors = set(db_features['color'].split())
            
            # If no color overlap, apply penalty
            if not query_colors.intersection(db_colors):
                color_penalty = 0.3  # 30% penalty for different colors
                logger.debug(f"Color mismatch penalty applied: {color_penalty}")
        
        for feature, weight in weights.items():
            query_text = query_features.get(feature, '')
            db_text = db_features.get(feature, '')
            
            if query_text and db_text:
                # Calculate embedding similarity for this feature
                query_emb = self.get_text_embedding(query_text)
                db_emb = self.get_text_embedding(db_text)
                
                if query_emb is not None and db_emb is not None:
                    similarity = self.calculate_similarity(query_emb, db_emb)
                    similarities.append(similarity * weight)
                    logger.debug(f"{feature} similarity: {similarity:.3f} (weight: {weight})")
        
        # Also check for exact matches in key fields
        exact_match_bonus = 0
        if query_features['category'] and db_features['category']:
            if query_features['category'] == db_features['category']:
                exact_match_bonus += 0.05
        
        if query_features['brand'] and db_features['brand']:
            if query_features['brand'] == db_features['brand']:
                exact_match_bonus += 0.1
        
        final_similarity = sum(similarities) + exact_match_bonus - color_penalty
        return max(0.0, min(final_similarity, 1.0))  # Ensure between 0 and 1
    
    def calculate_image_similarity_enhanced(self, query_image_url: str, db_image_url: str) -> float:
        """Enhanced image similarity that considers both CLIP features and color similarity"""
        try:
            # Get CLIP embeddings
            query_embedding = self.get_image_embedding(query_image_url)
            db_embedding = self.get_image_embedding(db_image_url)
            
            if query_embedding is None or db_embedding is None:
                return 0.0
            
            # Calculate CLIP similarity (70% weight)
            clip_similarity = self.calculate_similarity(query_embedding, db_embedding)
            
            # Calculate color similarity (30% weight)
            query_colors = self.extract_dominant_colors(query_image_url)
            db_colors = self.extract_dominant_colors(db_image_url)
            
            color_similarity = 0.0
            if query_colors and db_colors:
                # Calculate overlap between color sets
                query_color_set = set(query_colors)
                db_color_set = set(db_colors)
                
                if query_color_set and db_color_set:
                    intersection = len(query_color_set.intersection(db_color_set))
                    union = len(query_color_set.union(db_color_set))
                    color_similarity = intersection / union if union > 0 else 0.0
                    
                    logger.debug(f"Query colors: {query_colors}, DB colors: {db_colors}")
                    logger.debug(f"Color similarity: {color_similarity:.3f}")
            
            # Combine similarities
            final_similarity = (clip_similarity * 0.7) + (color_similarity * 0.3)
            
            logger.debug(f"CLIP: {clip_similarity:.3f}, Color: {color_similarity:.3f}, Final: {final_similarity:.3f}")
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error in enhanced image similarity: {e}")
            # Fallback to regular CLIP similarity
            return self.calculate_similarity(query_embedding, db_embedding) if query_embedding is not None and db_embedding is not None else 0.0

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings have the same dimension
            min_dim = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_dim]
            embedding2 = embedding2[:min_dim]
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            
            if norm_product == 0:
                return 0.0
            
            similarity = dot_product / norm_product
            return float(max(0.0, similarity))  # Ensure non-negative
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_items_structured(self, 
                                     query_image_url: Optional[str], 
                                     query_data: Dict,
                                     database_items: List[Dict],
                                     top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Find similar items using structured query data (preferred method)
        """
        logger.info(f"Finding similar items for structured query: {query_data}")
        
        # Get image embedding for query
        query_image_embedding = None
        if query_image_url:
            query_image_embedding = self.get_image_embedding(query_image_url)
            logger.info(f"Query image embedding: {'Success' if query_image_embedding is not None else 'Failed'}")
        
        similar_items = []
        
        for item in database_items:
            try:
                similarity_scores = []
                
                # Image similarity (60% weight) with enhanced color analysis
                if query_image_embedding is not None and item.get('imageUrl'):
                    img_similarity = self.calculate_image_similarity_enhanced(query_image_url, item['imageUrl'])
                    similarity_scores.append(img_similarity * 0.6)
                    logger.debug(f"Enhanced image similarity for item {item.get('_id', 'unknown')}: {img_similarity:.3f}")
                
                # Text similarity (40% weight) using structured comparison
                text_similarity = self.calculate_text_similarity_detailed(query_data, item)
                if text_similarity > 0:
                    similarity_scores.append(text_similarity * 0.4)
                    logger.debug(f"Text similarity for item {item.get('_id', 'unknown')}: {text_similarity:.3f}")
                
                # Calculate final similarity score
                if similarity_scores:
                    final_similarity = sum(similarity_scores)
                    if final_similarity > 0.05:  # Lower threshold for debugging
                        similar_items.append((item, final_similarity))
                        logger.info(f"Item {item.get('_id', 'unknown')} - Final similarity: {final_similarity:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing item {item.get('_id', 'unknown')}: {e}")
                continue
        
        # Sort by similarity score (descending)
        similar_items.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(similar_items)} similar items (before filtering)")
        
        # Apply higher threshold filtering to reduce false positives
        filtered_items = [(item, score) for item, score in similar_items if score > 0.25]  # Increased from 0.1 to 0.25
        logger.info(f"After threshold filtering (>0.25): {len(filtered_items)} items")
        
        return filtered_items[:top_k]

    def find_similar_items(self, 
                          query_image_url: Optional[str], 
                          query_text: str, 
                          database_items: List[Dict],
                          top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Find similar items from database with improved matching
        """
        logger.info(f"Finding similar items for query: '{query_text}' with image: {bool(query_image_url)}")
        
        # Create query item structure for text comparison
        query_item = {
            'whatLost': query_text,
            'category': '',
            'brand': '',
            'primaryColor': '',
            'additionalInfo': ''
        }
        
        # Try to extract structured info from query text if possible
        query_parts = query_text.lower().split()
        for part in query_parts:
            if part in ['electronics', 'clothing', 'bags', 'jewelry', 'keys', 'pets']:
                query_item['category'] = part
            elif part in ['black', 'white', 'red', 'blue', 'green', 'yellow', 'brown', 'gray', 'silver', 'gold']:
                query_item['primaryColor'] = part
        
        return self.find_similar_items_structured(query_image_url, query_item, database_items, top_k)
    
    def create_item_embedding(self, item: Dict) -> Optional[Dict]:
        """
        Create and return embeddings for a database item
        This can be used to pre-compute embeddings for faster similarity search
        """
        embeddings = {}
        
        # Image embedding
        if item.get('imageUrl'):
            image_embedding = self.get_image_embedding(item['imageUrl'])
            if image_embedding is not None:
                embeddings['image_embedding'] = image_embedding.tolist()
        
        # Text embedding
        features = self.extract_item_features(item)
        text_embedding = self.get_text_embedding(features['combined'])
        if text_embedding is not None:
            embeddings['text_embedding'] = text_embedding.tolist()
        
        return embeddings if embeddings else None

# Global instance
similarity_service = SimilarityService()