import numpy as np
import torch
import clip
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import re
import logging
from typing import List, Tuple, Dict, Optional, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityService:
    """
    Enhanced similarity service for lost and found item matching using CLIP and SentenceBERT models.
    Focuses on item characteristics for matching while excluding personal information.
    """
    
    def __init__(self):
        """Initialize the similarity service with pre-trained models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
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
        
        # Weights for similarity calculation
        self.weights = {
            'category': 0.34,       # 34% - Category matching
            'visual': 0.34,         # 34% - Image similarity  
            'description': 0.10,    # 10% - Item description similarity
            'primary_color': 0.10,  # 10% - Primary color matching
            'brand': 0.05,          # 5% - Brand matching
            'location': 0.04,       # 4% - Location matching
            'secondary_color': 0.03 # 3% - Secondary color matching
        }
        
        # Threshold settings - Only show suggestions if score > 80%
        self.minimum_threshold = 0.80     # 80% minimum to show results
        self.good_threshold = 0.85        # 85% for good matches
        self.excellent_threshold = 0.90   # 90% for excellent matches
        self.category_threshold = 0.1     # Must have at least 10% category similarity
    
    def _load_models(self):
        """Load CLIP and SentenceBERT models."""
        try:
            # Load CLIP model for image-text matching
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully")
            
            # Load SentenceBERT for text similarity
            logger.info("Loading SentenceBERT model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to basic text matching if models fail to load
            self.clip_model = None
            self.clip_preprocess = None
            self.text_model = None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra spaces
        text = re.sub(r'\s+', ' ', str(text).lower().strip())
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def standardize_color(self, color: str) -> str:
        """Standardize color names to base colors."""
        if not color:
            return ""
        
        color = self.normalize_text(color)
        
        for base_color, variations in self.color_mappings.items():
            if any(variation in color for variation in variations):
                return base_color
        
        return color
    
    def get_location_similarity(self, loc1: str, loc2: str) -> float:
        """Calculate location similarity between two locations."""
        if not loc1 or not loc2:
            return 0.0
        
        loc1 = self.normalize_text(loc1)
        loc2 = self.normalize_text(loc2)
        
        # Exact match
        if loc1 == loc2:
            return 1.0
        
        # Location mappings for similar venues
        location_groups = {
            'transport': ['airport', 'train station', 'bus stop', 'subway', 'taxi', 'uber', 'public transit'],
            'food': ['restaurant', 'cafe', 'bar', 'pub', 'coffee shop', 'food court'],
            'shopping': ['mall', 'store', 'shop', 'market', 'shopping center'],
            'education': ['school', 'university', 'college', 'library', 'campus'],
            'entertainment': ['theater', 'cinema', 'club', 'venue', 'concert hall'],
            'accommodation': ['hotel', 'motel', 'hostel', 'resort'],
            'outdoor': ['park', 'beach', 'street', 'outdoor'],
            'sports': ['gym', 'stadium', 'sports center', 'fitness center']
        }
        
        # Check if both locations are in the same category
        for category, locations in location_groups.items():
            loc1_match = any(location in loc1 for location in locations)
            loc2_match = any(location in loc2 for location in locations)
            
            if loc1_match and loc2_match:
                return 0.7  # High similarity for same location type
        
        # Use text similarity as fallback
        return self.calculate_text_similarity(loc1, loc2)

    def get_category_similarity(self, cat1: str, cat2: str) -> float:
        """Calculate category similarity with extensive debugging."""
        if not cat1 or not cat2:
            logger.info(f"CATEGORY DEBUG: Empty category - cat1: '{cat1}', cat2: '{cat2}'")
            return 0.0
        
        cat1 = self.normalize_text(cat1)
        cat2 = self.normalize_text(cat2)
        
        logger.info(f"CATEGORY DEBUG: Normalized categories - cat1: '{cat1}', cat2: '{cat2}'")
        
        # Exact match gets highest score
        if cat1 == cat2:
            logger.info(f"CATEGORY DEBUG: Exact category match!")
            return 1.0
        
        # Check if both categories belong to the same family
        cat1_family = None
        cat2_family = None
        
        for base_category, variations in self.category_mappings.items():
            # Check if cat1 matches this family
            if cat1 in variations or any(var in cat1 for var in variations):
                cat1_family = base_category
                logger.info(f"CATEGORY DEBUG: cat1 '{cat1}' matched family '{base_category}'")
            
            # Check if cat2 matches this family  
            if cat2 in variations or any(var in cat2 for var in variations):
                cat2_family = base_category
                logger.info(f"CATEGORY DEBUG: cat2 '{cat2}' matched family '{base_category}'")
        
        logger.info(f"CATEGORY DEBUG: Category families - cat1_family: '{cat1_family}', cat2_family: '{cat2_family}'")
        
        # If both are in the same category family, high similarity
        if cat1_family and cat2_family and cat1_family == cat2_family:
            logger.info(f"CATEGORY DEBUG: Same category family match: {cat1_family}")
            return 0.95  # Very high similarity for same category family
        
        # No category match
        logger.info(f"CATEGORY DEBUG: No category match found")
        return 0.0
    
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using SentenceBERT."""
        if not self.text_model or not text:
            return None
        
        try:
            text = self.normalize_text(text)
            embedding = self.text_model.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            return None
    
    def get_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """Get image embedding using CLIP."""
        if not self.clip_model or not image_url:
            return None
        
        try:
            # Download and process image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess and encode image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        # Exact match gets perfect score
        if text1 == text2:
            return 1.0
        
        # Use SentenceBERT if available
        if self.text_model:
            try:
                embeddings = self.text_model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                similarity = max(0.0, float(similarity))
                
                # Boost score for very similar descriptions
                if similarity > 0.8:
                    similarity = min(1.0, similarity * 1.1)
                
                return similarity
            except Exception as e:
                logger.error(f"Error calculating text similarity: {e}")
        
        # Fallback to enhanced word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Check for substring matches for better partial matching
        for word1 in words1:
            for word2 in words2:
                if len(word1) > 3 and len(word2) > 3:
                    if word1 in word2 or word2 in word1:
                        jaccard += 0.1  # Boost for partial word matches
        
        return min(1.0, jaccard)
    
    def calculate_image_similarity(self, image_url1: str, image_url2: str) -> float:
        """Calculate similarity between two images using CLIP."""
        if not image_url1 or not image_url2:
            return 0.0
        
        embedding1 = self.get_image_embedding(image_url1)
        embedding2 = self.get_image_embedding(image_url2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            similarity = max(0.0, float(similarity))
            
            # Better image similarity scoring - CLIP often gives lower scores
            if similarity > 0.7:
                similarity = min(1.0, similarity * 1.2)  # Boost high similarities
            elif similarity > 0.5:
                similarity = min(1.0, similarity * 1.1)  # Moderate boost
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating image similarity: {e}")
            return 0.0
    
    def calculate_comprehensive_similarity(self, query_data: Dict[str, Any], found_item: Dict[str, Any]) -> float:
        """
        Calculate comprehensive similarity score with your requested weights.
        34% category, 34% image, 10% description, 10% primary color, 5% brand, 4% location, 3% secondary color
        """
        total_score = 0.0
        debug_scores = {}
        
        # 1. Category Matching (34% weight)
        query_category = query_data.get('category', '')
        found_category = found_item.get('category', '')
        category_similarity = self.get_category_similarity(query_category, found_category)
        category_score = category_similarity * self.weights['category']
        total_score += category_score
        debug_scores['category'] = category_score
        
        # 2. Visual Similarity (34% weight) - EQUAL HIGHEST PRIORITY
        query_image = query_data.get('imageUrl')
        found_image = found_item.get('imageUrl')
        if query_image and found_image:
            visual_similarity = self.calculate_image_similarity(query_image, found_image)
            visual_score = visual_similarity * self.weights['visual']
            logger.info(f"IMAGE DEBUG: Comparing images - similarity: {visual_similarity:.3f}")
        else:
            # Give substantial partial credit since image is 34% of score
            visual_score = 0.6 * self.weights['visual']  # 60% partial credit
            logger.info(f"IMAGE DEBUG: Missing images, giving partial credit: {visual_score:.3f}")
        
        total_score += visual_score
        debug_scores['visual'] = visual_score
        
        # 3. Item Description Similarity (10% weight)
        query_desc = query_data.get('whatLost', '')
        found_desc = found_item.get('whatFound', found_item.get('whatLost', ''))
        desc_similarity = self.calculate_text_similarity(query_desc, found_desc)
        desc_score = desc_similarity * self.weights['description']
        total_score += desc_score
        debug_scores['description'] = desc_score
        
        logger.info(f"DESCRIPTION DEBUG: '{query_desc}' vs '{found_desc}' = {desc_similarity:.3f}")
        
        # 4. Primary Color Matching (10% weight)
        query_color = self.standardize_color(query_data.get('primaryColor', ''))
        found_color = self.standardize_color(found_item.get('primaryColor', ''))
        if query_color and found_color:
            color_similarity = 1.0 if query_color == found_color else self.calculate_text_similarity(query_color, found_color)
            color_score = color_similarity * self.weights['primary_color']
            logger.info(f"COLOR DEBUG: '{query_color}' vs '{found_color}' = {color_similarity:.3f}")
        else:
            color_score = 0.5 * self.weights['primary_color']
            logger.info(f"COLOR DEBUG: Missing color info, giving partial credit: {color_score:.3f}")
        
        total_score += color_score
        debug_scores['primary_color'] = color_score
        
        # 5. Brand Matching (5% weight)
        query_brand = query_data.get('brand', '')
        found_brand = found_item.get('brand', '')
        if query_brand and found_brand:
            brand_similarity = self.calculate_text_similarity(query_brand, found_brand)
            brand_score = brand_similarity * self.weights['brand']
            logger.info(f"BRAND DEBUG: '{query_brand}' vs '{found_brand}' = {brand_similarity:.3f}")
        else:
            brand_score = 0.5 * self.weights['brand']
            logger.info(f"BRAND DEBUG: Missing brand info, giving partial credit")
        
        total_score += brand_score
        debug_scores['brand'] = brand_score
        
        # 6. Location Matching (4% weight)
        query_location = query_data.get('whereLost', '')
        found_location = found_item.get('whereFound', found_item.get('whereLost', ''))
        if query_location and found_location:
            location_similarity = self.get_location_similarity(query_location, found_location)
            location_score = location_similarity * self.weights['location']
            logger.info(f"LOCATION DEBUG: '{query_location}' vs '{found_location}' = {location_similarity:.3f}")
        else:
            location_score = 0.5 * self.weights['location']
            logger.info(f"LOCATION DEBUG: Missing location info, giving partial credit")
        
        total_score += location_score
        debug_scores['location'] = location_score
        
        # 7. Secondary Color Matching (3% weight)
        query_sec_color = self.standardize_color(query_data.get('secondaryColor', ''))
        found_sec_color = self.standardize_color(found_item.get('secondaryColor', ''))
        if query_sec_color and found_sec_color:
            sec_color_similarity = 1.0 if query_sec_color == found_sec_color else self.calculate_text_similarity(query_sec_color, found_sec_color)
            sec_color_score = sec_color_similarity * self.weights['secondary_color']
        else:
            sec_color_score = 0.5 * self.weights['secondary_color']
        
        total_score += sec_color_score
        debug_scores['secondary_color'] = sec_color_score
        
        final_score = min(total_score, 1.0)
        
        logger.info(f"FINAL DEBUG: Score breakdown: {debug_scores}")
        logger.info(f"FINAL DEBUG: Total score: {final_score:.3f}")
        
        return final_score
    
    def find_similar_items_structured(self, query_image_url: Optional[str], query_data: Dict[str, Any], 
                                    database_items: List[Dict[str, Any]], top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar items based on comprehensive similarity calculation.
        """
        if not database_items:
            logger.info("SEARCH DEBUG: No database items provided")
            return []
        
        # Add image URL to query data if provided
        if query_image_url:
            query_data['imageUrl'] = query_image_url
        
        logger.info(f"SEARCH DEBUG: Query data: {query_data}")
        logger.info(f"SEARCH DEBUG: Database has {len(database_items)} items")
        
        # Log first few database items for inspection
        for i, item in enumerate(database_items[:3]):
            logger.info(f"SEARCH DEBUG: Item {i}: whatFound='{item.get('whatFound', item.get('whatLost', 'MISSING'))}'")
        
        similarities = []
        query_item = query_data.get('whatLost', '')
        
        logger.info(f"SEARCH DEBUG: Searching for items similar to: '{query_item}'")
        
        for i, item in enumerate(database_items):
            try:
                logger.info(f"SEARCH DEBUG: Processing item {i+1}/{len(database_items)}")
                
                similarity_score = self.calculate_comprehensive_similarity(query_data, item)
                
                logger.info(f"SEARCH DEBUG: Item {i+1} final similarity score: {similarity_score:.3f}")
                
                # Use strict threshold - only show suggestions with 80%+ similarity
                if similarity_score >= self.minimum_threshold:  # 80% minimum to show anything
                    similarities.append((item, similarity_score))
                    
                    if similarity_score >= self.excellent_threshold:
                        logger.info(f"SEARCH DEBUG: Item {i+1} - EXCELLENT MATCH (90%+): {similarity_score:.3f}")
                    elif similarity_score >= self.good_threshold:
                        logger.info(f"SEARCH DEBUG: Item {i+1} - GOOD MATCH (85%+): {similarity_score:.3f}")
                    else:
                        logger.info(f"SEARCH DEBUG: Item {i+1} - STRONG MATCH (80%+): {similarity_score:.3f}")
                else:
                    logger.info(f"SEARCH DEBUG: Item {i+1} filtered out (below 80% threshold)")
                    
            except Exception as e:
                logger.error(f"SEARCH DEBUG: Error processing item {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Count matches by quality
        excellent_matches = sum(1 for _, score in similarities if score >= self.excellent_threshold)
        good_matches = sum(1 for _, score in similarities if score >= self.good_threshold)
        
        logger.info(f"SEARCH DEBUG: FINAL RESULT: {len(similarities)} total matches")
        logger.info(f"SEARCH DEBUG: - Excellent matches (90%+): {excellent_matches}")
        logger.info(f"SEARCH DEBUG: - Good matches (85%+): {good_matches}")
        logger.info(f"SEARCH DEBUG: - All matches (80%+): {len(similarities)}")
        
        if similarities:
            logger.info("SEARCH DEBUG: Top matches:")
            for i, (item, score) in enumerate(similarities[:3]):
                quality = "EXCELLENT" if score >= self.excellent_threshold else "GOOD" if score >= self.good_threshold else "STRONG"
                logger.info(f"SEARCH DEBUG: Match {i+1} ({quality}): {item.get('whatFound', 'N/A')} - Score: {score:.3f}")
        else:
            logger.info("SEARCH DEBUG: NO MATCHES FOUND ABOVE 80% THRESHOLD!")
        
        return similarities[:top_k]
    
    def find_similar_items(self, query_image_url: Optional[str], query_text: str, 
                          database_items: List[Dict[str, Any]], top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Legacy method for backward compatibility.
        """
        # Better category detection
        category = ""
        best_match_score = 0
        
        for base_category, variations in self.category_mappings.items():
            for variation in variations:
                if variation in query_text.lower():
                    # Prefer longer, more specific matches
                    match_score = len(variation)
                    if match_score > best_match_score:
                        category = base_category.title()
                        best_match_score = match_score
        
        # Create structured query data
        query_data = {
            'whatLost': query_text,
            'category': category,
            'brand': "",
            'primaryColor': "",
            'secondaryColor': "",
            'additionalInfo': ""
        }
        
        return self.find_similar_items_structured(query_image_url, query_data, database_items, top_k)
    
    def get_similarity_explanation(self, query_data: Dict[str, Any], found_item: Dict[str, Any]) -> Dict[str, float]:
        """
        Get detailed breakdown of similarity scores for debugging.
        """
        scores = {}
        
        # Category similarity
        query_category = query_data.get('category', '')
        found_category = found_item.get('category', '')
        scores['category'] = self.get_category_similarity(query_category, found_category)
        
        # Visual similarity
        query_image = query_data.get('imageUrl')
        found_image = found_item.get('imageUrl')
        scores['visual'] = self.calculate_image_similarity(query_image, found_image) if query_image and found_image else 0.0
        
        # Description similarity
        query_desc = query_data.get('whatLost', '')
        found_desc = found_item.get('whatFound', found_item.get('whatLost', ''))
        scores['description'] = self.calculate_text_similarity(query_desc, found_desc)
        
        # Primary color similarity
        query_color = self.standardize_color(query_data.get('primaryColor', ''))
        found_color = self.standardize_color(found_item.get('primaryColor', ''))
        scores['primary_color'] = 1.0 if query_color and found_color and query_color == found_color else 0.0
        
        # Brand similarity
        query_brand = query_data.get('brand', '')
        found_brand = found_item.get('brand', '')
        scores['brand'] = self.calculate_text_similarity(query_brand, found_brand)
        
        # Location similarity
        query_location = query_data.get('whereLost', '')
        found_location = found_item.get('whereFound', found_item.get('whereLost', ''))
        scores['location'] = self.get_location_similarity(query_location, found_location)
        
        # Secondary color similarity
        query_sec_color = self.standardize_color(query_data.get('secondaryColor', ''))
        found_sec_color = self.standardize_color(found_item.get('secondaryColor', ''))
        scores['secondary_color'] = 1.0 if query_sec_color and found_sec_color and query_sec_color == found_sec_color else 0.0
        
        return scores


# Global instance for use in Flask app
similarity_service = SimilarityService()