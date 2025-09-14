import numpy as np
import torch
import clip
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import re
from typing import List, Tuple, Dict, Any, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class SimilarityService:
    """
    Core AI matching engine for Lost and Found items.
    
    This service uses multimodal input (text and images) to calculate
    a comprehensive similarity score between a query item and a database of found items.
    """
    
    def __init__(self):
        """Initialize the similarity service with pre-trained models and configurations."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_models()
        
        # Color mappings for standardization
        self.color_mappings = {
            'red': ['red', 'crimson', 'scarlet', 'cherry', 'burgundy', 'maroon'],
            'blue': ['blue', 'navy', 'azure', 'cobalt', 'royal blue', 'sky blue'],
            'green': ['green', 'emerald', 'forest', 'lime', 'olive', 'mint'],
            'yellow': ['yellow', 'gold', 'golden', 'amber', 'lemon', 'cream'],
            'black': ['black', 'charcoal', 'ebony', 'jet black', 'dark'],
            'white': ['white', 'ivory', 'pearl', 'snow', 'cream', 'off-white'],
            'brown': ['brown', 'tan', 'beige', 'coffee', 'chocolate', 'camel'],
            'pink': ['pink', 'rose', 'magenta', 'fuchsia', 'coral'],
            'purple': ['purple', 'violet', 'lavender', 'plum', 'indigo'],
            'orange': ['orange', 'tangerine', 'peach', 'coral', 'salmon'],
            'gray': ['gray', 'grey', 'silver', 'ash', 'slate', 'pewter']
        }
        
        # Category mappings - case insensitive
        self.category_mappings = {
            'electronics': ['electronics', 'phone', 'smartphone', 'cell phone', 'mobile', 'laptop', 'tablet', 'computer', 'device', 'headphones', 'earbuds', 'charger', 'cable'],
            'clothing': ['clothing', 'shirt', 'jacket', 'pants', 'dress', 'coat', 'sweater', 'jeans', 'blouse', 'hoodie', 't-shirt', 'skirt'],
            'personal accessories': ['personal accessories', 'accessories', 'jewelry', 'watch', 'glasses', 'sunglasses', 'hat', 'scarf', 'belt', 'gloves', 'ring', 'necklace', 'bracelet', 'earrings'],
            'bags': ['bags', 'bag', 'backpack', 'purse', 'wallet', 'handbag', 'luggage', 'suitcase', 'tote', 'messenger bag', 'duffel', 'briefcase'],
            'keys': ['keys', 'key', 'keychain', 'car key', 'house key', 'key fob'],
            'documents': ['documents', 'passport', 'license', 'id', 'card', 'paper', 'certificate', 'diploma'],
            'pets': ['pets', 'dog', 'cat', 'animal', 'puppy', 'kitten'],
            'sports equipment': ['sports equipment', 'sports', 'equipment', 'ball', 'gear', 'fitness', 'gym', 'exercise'],
            'other': ['other', 'misc', 'miscellaneous']
        }
        
        # Weights for similarity calculation as per requirement
        self.weights = {
            'category': 0.34,
            'visual': 0.34,
            'description': 0.10,
            'primary_color': 0.10,
            'brand': 0.05,
            'location': 0.04,
            'secondary_color': 0.03
        }
        
        # Threshold for ranked suggestions
        self.minimum_threshold = 0.80

    def _load_models(self):
        """Load CLIP and SentenceBERT models."""
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.clip_model = None
            self.text_model = None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', str(text).lower().strip())
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    def _standardize_color(self, color: str) -> str:
        """Standardize color names to base colors."""
        if not color:
            return ""
        color = self._normalize_text(color)
        for base_color, variations in self.color_mappings.items():
            if any(variation in color for variation in variations):
                return base_color
        return color
    
    def _get_category_similarity(self, cat1: str, cat2: str) -> float:
        """Calculate category similarity based on predefined families."""
        if not cat1 or not cat2:
            return 0.0
        
        cat1 = self._normalize_text(cat1)
        cat2 = self._normalize_text(cat2)
        
        if cat1 == cat2:
            return 1.0
        
        cat1_family = None
        cat2_family = None
        
        for base_category, variations in self.category_mappings.items():
            if cat1 in variations:
                cat1_family = base_category
            if cat2 in variations:
                cat2_family = base_category
        
        if cat1_family and cat2_family and cat1_family == cat2_family:
            return 0.95
        
        return 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text strings using SentenceBERT."""
        if not self.text_model or not text1 or not text2:
            return 0.0
        
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        if not text1 or not text2:
            return 0.0

        embeddings = self.text_model.encode([text1, text2], convert_to_tensor=True)
        # Using cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        return max(0.0, float(similarity.item()))

    def _get_image_embedding(self, image_url: str) -> Optional[torch.Tensor]:
        """Get image embedding using CLIP."""
        if not self.clip_model or not image_url:
            return None
        
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu()
        except Exception:
            return None

    def _calculate_image_similarity(self, image_url1: str, image_url2: str) -> float:
        """Calculate cosine similarity between two images using CLIP embeddings."""
        embedding1 = self._get_image_embedding(image_url1)
        embedding2 = self._get_image_embedding(image_url2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return max(0.0, float(similarity.item()))

    def calculate_comprehensive_similarity(self, query_data: Dict[str, Any], found_item: Dict[str, Any]) -> float:
        """
        Calculates a comprehensive similarity score based on weighted criteria.
        
        The weight distribution is based on the provided requirements:
        34% visual, 34% category, 10% description, 10% primary color, 
        5% brand, 4% location, and 3% secondary color.
        """
        total_score = 0.0

        # Category Matching (34% weight)
        category_similarity = self._get_category_similarity(query_data.get('category', ''), found_item.get('category', ''))
        total_score += category_similarity * self.weights['category']
        
        # Visual Similarity (34% weight)
        visual_similarity = self._calculate_image_similarity(query_data.get('imageUrl', ''), found_item.get('imageUrl', ''))
        total_score += visual_similarity * self.weights['visual']
        
        # Description Similarity (10% weight)
        description_similarity = self._calculate_text_similarity(query_data.get('whatLost', ''), found_item.get('whatFound', ''))
        total_score += description_similarity * self.weights['description']

        # Primary Color Matching (10% weight)
        color_match = self._standardize_color(query_data.get('primaryColor', '')) == self._standardize_color(found_item.get('primaryColor', ''))
        total_score += (1.0 if color_match else 0.0) * self.weights['primary_color']
        
        # Brand Matching (5% weight)
        brand_similarity = self._calculate_text_similarity(query_data.get('brand', ''), found_item.get('brand', ''))
        total_score += brand_similarity * self.weights['brand']

        # Location Matching (4% weight)
        location_similarity = self._calculate_text_similarity(query_data.get('whereLost', ''), found_item.get('whereFound', ''))
        total_score += location_similarity * self.weights['location']
        
        # Secondary Color Matching (3% weight)
        sec_color_match = self._standardize_color(query_data.get('secondaryColor', '')) == self._standardize_color(found_item.get('secondaryColor', ''))
        total_score += (1.0 if sec_color_match else 0.0) * self.weights['secondary_color']

        return min(total_score, 1.0)
    
    def find_similar_items(self, query_data: Dict[str, Any], database_items: List[Dict[str, Any]], top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Finds and ranks similar items from a database based on the comprehensive score.
        Filters out matches below the 80% minimum threshold.
        """
        if not database_items:
            return []
        
        similarities = []
        for item in database_items:
            try:
                similarity_score = self.calculate_comprehensive_similarity(query_data, item)
                
                if similarity_score >= self.minimum_threshold:
                    similarities.append((item, similarity_score))
            except Exception:
                continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]