import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image
import io
import torch
import clip
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityService:
    def __init__(self):
        """Initialize the similarity service with CLIP and SentenceBERT models"""
        self.text_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load the ML models for text and image processing"""
        try:
            # Load SentenceBERT for text similarity
            logger.info("Loading SentenceBERT model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load CLIP for image-text similarity
            logger.info(f"Loading CLIP model on device: {self.device}")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise e
    
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using SentenceBERT"""
        try:
            if not text or not text.strip():
                return None
            
            # Clean and normalize text
            text = text.strip().lower()
            
            # Generate embedding
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def get_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """Get image embedding using CLIP"""
        try:
            if not image_url:
                return None
            
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Process image
            image = Image.open(io.BytesIO(response.content))
            image = image.convert('RGB')  # Ensure RGB format
            
            # Preprocess and encode
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None
    
    def get_text_image_similarity(self, text: str, image_url: str) -> float:
        """Calculate similarity between text and image using CLIP"""
        try:
            if not text or not image_url:
                return 0.0
            
            # Download and process image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Preprocess inputs
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([text.strip()]).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                return float(similarity.cpu().numpy()[0])
        
        except Exception as e:
            logger.error(f"Error calculating text-image similarity: {e}")
            return 0.0
    
    def create_structured_text(self, item_data: Dict) -> str:
        """Create a structured text description from item data"""
        parts = []
        
        # Core description
        if item_data.get('whatLost'):
            parts.append(item_data['whatLost'])
        elif item_data.get('whatFound'):
            parts.append(item_data['whatFound'])
        
        # Category and brand
        if item_data.get('category'):
            parts.append(item_data['category'])
        if item_data.get('brand'):
            parts.append(f"{item_data['brand']} brand")
        
        # Colors
        if item_data.get('primaryColor'):
            parts.append(f"{item_data['primaryColor']} color")
        if item_data.get('secondaryColor'):
            parts.append(f"with {item_data['secondaryColor']}")
        
        # Additional details
        if item_data.get('additionalInfo'):
            parts.append(item_data['additionalInfo'])
        
        # Location context (lower weight)
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
        """Calculate comprehensive similarity using text and image features"""
        
        similarities = []
        weights = []
        
        # Text-to-text similarity (high weight)
        if query_text and target_text:
            query_embedding = self.get_text_embedding(query_text)
            target_embedding = self.get_text_embedding(target_text)
            
            if query_embedding is not None and target_embedding is not None:
                text_sim = cosine_similarity([query_embedding], [target_embedding])[0][0]
                similarities.append(text_sim)
                weights.append(0.4)  # 40% weight for text similarity
        
        # Image-to-image similarity (high weight)
        if query_image_url and target_image_url:
            query_img_embedding = self.get_image_embedding(query_image_url)
            target_img_embedding = self.get_image_embedding(target_image_url)
            
            if query_img_embedding is not None and target_img_embedding is not None:
                img_sim = cosine_similarity([query_img_embedding], [target_img_embedding])[0][0]
                similarities.append(img_sim)
                weights.append(0.4)  # 40% weight for image similarity
        
        # Cross-modal similarities (lower weight)
        if query_text and target_image_url:
            cross_sim1 = self.get_text_image_similarity(query_text, target_image_url)
            similarities.append(cross_sim1)
            weights.append(0.1)  # 10% weight
        
        if query_image_url and target_text:
            cross_sim2 = self.get_text_image_similarity(target_text, query_image_url)
            similarities.append(cross_sim2)
            weights.append(0.1)  # 10% weight
        
        # Calculate weighted average
        if similarities:
            weighted_sum = sum(s * w for s, w in zip(similarities, weights))
            total_weight = sum(weights)
            final_score = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Apply boost for having both text and image data
            completeness_boost = 1.0
            if query_image_url and query_text and target_image_url and target_text:
                completeness_boost = 1.1  # 10% boost for complete data
            
            return min(final_score * completeness_boost, 1.0)
        
        return 0.0
    
    def find_similar_items_structured(self, 
                                    query_image_url: Optional[str],
                                    query_data: Dict,
                                    database_items: List[Dict],
                                    top_k: int = 10,
                                    min_score: float = 0.5) -> List[Tuple[Dict, float]]:
        """Find similar items using structured data comparison with minimum score threshold"""
        
        if not database_items:
            return []
        
        # Create query text
        query_text = self.create_structured_text(query_data)
        logger.info(f"Query text: {query_text}")
        logger.info(f"Query has image: {bool(query_image_url)}")
        logger.info(f"Minimum similarity score threshold: {min_score}")
        
        results = []
        
        for item in database_items:
            try:
                # Create target text
                target_text = self.create_structured_text(item)
                target_image_url = item.get('imageUrl')
                
                # Calculate multimodal similarity
                similarity_score = self.calculate_multimodal_similarity(
                    query_text=query_text,
                    query_image_url=query_image_url,
                    target_text=target_text,
                    target_image_url=target_image_url
                )
                
                # Apply category bonus
                if (query_data.get('category') and item.get('category') and 
                    query_data['category'].lower() == item.get('category', '').lower()):
                    similarity_score *= 1.2  # 20% bonus for same category
                
                # Apply color bonus
                query_color = query_data.get('primaryColor', '').lower()
                target_color = item.get('primaryColor', '').lower()
                if query_color and target_color and query_color == target_color:
                    similarity_score *= 1.1  # 10% bonus for same color
                
                # Brand bonus
                query_brand = query_data.get('brand', '').lower()
                target_brand = item.get('brand', '').lower()
                if query_brand and target_brand and query_brand == target_brand:
                    similarity_score *= 1.15  # 15% bonus for same brand
                
                # Ensure score doesn't exceed 1.0
                similarity_score = min(similarity_score, 1.0)
                
                # Only include items with score >= min_score (default 0.5 = 50%)
                if similarity_score >= min_score:
                    results.append((item, similarity_score))
                    logger.info(f"Item {item.get('_id', 'unknown')} included with score: {similarity_score:.3f}")
                else:
                    logger.debug(f"Item {item.get('_id', 'unknown')} filtered out with score: {similarity_score:.3f}")
                    
            except Exception as e:
                logger.error(f"Error processing item {item.get('_id', 'unknown')}: {e}")
                continue
        
        # Sort by similarity score (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(results)} items with similarity >= {min_score}")
        return results[:top_k]
    
    def find_similar_items(self, 
                          query_image_url: Optional[str],
                          query_text: str,
                          database_items: List[Dict],
                          top_k: int = 10,
                          min_score: float = 0.5) -> List[Tuple[Dict, float]]:
        """Legacy method for backward compatibility with 50% minimum score"""
        
        # Convert to structured format
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

# Global instance
similarity_service = SimilarityService()