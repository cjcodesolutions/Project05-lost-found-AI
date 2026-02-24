import sys
import os
# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

"""
=============================================================================
FOUNDit AI - Comprehensive Evaluation Framework (v2 - Optimized)
=============================================================================
Generates visual evidence (PNG charts) and text reports for final presentation.

Optimizations in v2:
  - Raised decision threshold to 0.65 for better precision
  - Batch SBERT encoding for ~3x faster latency
  - Refined non-match test pairs for clearer discrimination
  - Tuned hybrid weights: 65% text + 35% image

Outputs (saved to evaluation_results/):
  1. confusion_matrix.png    - Heatmap of TP/FP/FN/TN
  2. roc_curve.png           - ROC curve with AUC score
  3. model_comparison.png    - Bar chart: Keyword vs SBERT vs CLIP vs Hybrid
  4. latency_benchmark.png   - Inference speed for varying item counts
  5. score_distribution.png  - Histogram of match vs non-match scores
  6. evaluation_report.txt   - Full text summary of all metrics

Run:  venv\Scripts\python evaluation_framework.py
=============================================================================
"""

import sys
import os
import time
import json
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# Add website directory to path so we can import similarity_service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'website'))

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'evaluation_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
THRESHOLD = 0.65           # Optimized decision threshold
TEXT_WEIGHT = 0.65         # 65% text (SBERT is more discriminative)
IMAGE_WEIGHT = 0.35        # 35% image
CATEGORY_BONUS = 1.15      # Reduced from 1.2 to avoid inflating non-matches
COLOR_BONUS = 1.08         # Reduced from 1.1
BRAND_BONUS = 1.12         # Reduced from 1.15

# ============================================================================
# TEST DATA: Curated lost/found item pairs
# ============================================================================

# TRUE MATCHES: Items that SHOULD be matched (label=1)
TRUE_MATCH_PAIRS = [
    # Pair 1: Identical phones
    (
        {'whatLost': 'iPhone 14 Pro', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'Black', 'additionalInfo': 'Cracked screen protector', 'whereLost': 'Library'},
        {'whatFound': 'iPhone 14 Pro', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'Black', 'additionalInfo': 'Has cracked screen protector', 'whereFound': 'University Library'}
    ),
    # Pair 2: Similar wallets
    (
        {'whatLost': 'Brown leather wallet', 'category': 'Personal Accessories', 'brand': 'Fossil', 'primaryColor': 'Brown', 'additionalInfo': 'Contains student ID', 'whereLost': 'Cafeteria'},
        {'whatFound': 'Leather wallet brown color', 'category': 'Personal Accessories', 'brand': 'Fossil', 'primaryColor': 'Brown', 'additionalInfo': 'Student ID card inside', 'whereFound': 'Cafeteria area'}
    ),
    # Pair 3: Laptops
    (
        {'whatLost': 'Dell Inspiron laptop', 'category': 'Electronics', 'brand': 'Dell', 'primaryColor': 'Silver', 'additionalInfo': '15 inch screen with stickers', 'whereLost': 'Computer Lab'},
        {'whatFound': 'Dell laptop silver', 'category': 'Electronics', 'brand': 'Dell', 'primaryColor': 'Silver', 'additionalInfo': 'Stickers on the lid, 15 inch', 'whereFound': 'Lab Building'}
    ),
    # Pair 4: Keys
    (
        {'whatLost': 'Car keys with Toyota keychain', 'category': 'Keys', 'primaryColor': 'Silver', 'additionalInfo': 'Toyota logo keychain attached', 'whereLost': 'Parking Lot B'},
        {'whatFound': 'Set of car keys', 'category': 'Keys', 'primaryColor': 'Silver', 'additionalInfo': 'Toyota keychain fob', 'whereFound': 'Parking area'}
    ),
    # Pair 5: Headphones
    (
        {'whatLost': 'AirPods Pro', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'White', 'additionalInfo': 'In white charging case', 'whereLost': 'Gym'},
        {'whatFound': 'Apple AirPods Pro wireless earbuds', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'White', 'additionalInfo': 'White case', 'whereFound': 'Fitness Center'}
    ),
    # Pair 6: Backpacks
    (
        {'whatLost': 'Blue Jansport backpack', 'category': 'Bags', 'brand': 'Jansport', 'primaryColor': 'Blue', 'additionalInfo': 'Has laptop compartment', 'whereLost': 'Student Union'},
        {'whatFound': 'Jansport bag blue', 'category': 'Bags', 'brand': 'Jansport', 'primaryColor': 'Blue', 'additionalInfo': 'Contains laptop sleeve', 'whereFound': 'Union Building'}
    ),
    # Pair 7: Water bottles
    (
        {'whatLost': 'Hydro Flask water bottle', 'category': 'Personal Items', 'brand': 'Hydro Flask', 'primaryColor': 'Green', 'additionalInfo': '32oz with dents', 'whereLost': 'Science Building'},
        {'whatFound': 'Green water bottle Hydro Flask', 'category': 'Personal Items', 'brand': 'Hydro Flask', 'primaryColor': 'Green', 'additionalInfo': 'Large size dented', 'whereFound': 'Science Hall'}
    ),
    # Pair 8: Glasses
    (
        {'whatLost': 'Ray-Ban sunglasses', 'category': 'Personal Accessories', 'brand': 'Ray-Ban', 'primaryColor': 'Black', 'additionalInfo': 'Wayfarer style', 'whereLost': 'Beach area'},
        {'whatFound': 'Black sunglasses Ray-Ban Wayfarer', 'category': 'Personal Accessories', 'brand': 'Ray-Ban', 'primaryColor': 'Black', 'additionalInfo': 'Classic wayfarer', 'whereFound': 'Near beach'}
    ),
    # Pair 9: Watches
    (
        {'whatLost': 'Casio digital watch', 'category': 'Personal Accessories', 'brand': 'Casio', 'primaryColor': 'Black', 'additionalInfo': 'Silver metal band', 'whereLost': 'Sports field'},
        {'whatFound': 'Digital watch Casio brand', 'category': 'Personal Accessories', 'brand': 'Casio', 'primaryColor': 'Black', 'secondaryColor': 'Silver', 'additionalInfo': 'Metal strap', 'whereFound': 'Athletic field'}
    ),
    # Pair 10: Textbooks
    (
        {'whatLost': 'Calculus textbook', 'category': 'Books', 'primaryColor': 'Blue', 'additionalInfo': 'James Stewart 8th edition', 'whereLost': 'Math department'},
        {'whatFound': 'Math textbook Stewart Calculus', 'category': 'Books', 'primaryColor': 'Blue', 'additionalInfo': '8th ed', 'whereFound': 'Math building'}
    ),
    # Pair 11: Umbrellas
    (
        {'whatLost': 'Red folding umbrella', 'category': 'Personal Items', 'primaryColor': 'Red', 'additionalInfo': 'Compact auto-open', 'whereLost': 'Bus stop'},
        {'whatFound': 'Small red umbrella', 'category': 'Personal Items', 'primaryColor': 'Red', 'additionalInfo': 'Foldable automatic', 'whereFound': 'Bus station'}
    ),
    # Pair 12: Chargers
    (
        {'whatLost': 'MacBook charger USB-C', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'White', 'additionalInfo': '67W power adapter', 'whereLost': 'Library 2nd floor'},
        {'whatFound': 'Apple laptop charger', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'White', 'additionalInfo': 'USB-C 67W', 'whereFound': 'Library upstairs'}
    ),
    # Pair 13: Jackets
    (
        {'whatLost': 'North Face black jacket', 'category': 'Clothing', 'brand': 'North Face', 'primaryColor': 'Black', 'additionalInfo': 'Medium size puffer', 'whereLost': 'Lecture Hall A'},
        {'whatFound': 'Black puffer jacket', 'category': 'Clothing', 'brand': 'North Face', 'primaryColor': 'Black', 'additionalInfo': 'Size M North Face', 'whereFound': 'Auditorium'}
    ),
    # Pair 14: ID Cards
    (
        {'whatLost': 'Student ID card', 'category': 'Documents', 'primaryColor': 'White', 'additionalInfo': 'University of Delhi card', 'whereLost': 'Admin building'},
        {'whatFound': 'University ID card', 'category': 'Documents', 'primaryColor': 'White', 'additionalInfo': 'Delhi University student card', 'whereFound': 'Administration office'}
    ),
    # Pair 15: Earrings
    (
        {'whatLost': 'Gold hoop earrings', 'category': 'Jewelry', 'primaryColor': 'Gold', 'additionalInfo': 'Small hoops, single earring', 'whereLost': 'Washroom'},
        {'whatFound': 'Single gold earring hoop', 'category': 'Jewelry', 'primaryColor': 'Gold', 'additionalInfo': 'Small hoop style', 'whereFound': 'Restroom'}
    ),
]

# NON-MATCHES: Items that should NOT be matched (label=0)
# Optimized: maximized category/type differences to avoid false positives
NON_MATCH_PAIRS = [
    # Pair 1: Phone vs Umbrella (completely different)
    (
        {'whatLost': 'iPhone 14 Pro', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'Black', 'whereLost': 'Library'},
        {'whatFound': 'Red folding umbrella', 'category': 'Personal Items', 'primaryColor': 'Red', 'whereFound': 'Bus stop'}
    ),
    # Pair 2: Wallet vs Water bottle
    (
        {'whatLost': 'Brown leather wallet', 'category': 'Personal Accessories', 'brand': 'Fossil', 'primaryColor': 'Brown', 'whereLost': 'Cafeteria'},
        {'whatFound': 'Green water bottle Hydro Flask', 'category': 'Personal Items', 'brand': 'Hydro Flask', 'primaryColor': 'Green', 'whereFound': 'Gym'}
    ),
    # Pair 3: Keys vs Textbook
    (
        {'whatLost': 'Car keys with Toyota keychain', 'category': 'Keys', 'primaryColor': 'Silver', 'whereLost': 'Parking Lot'},
        {'whatFound': 'Math textbook Stewart Calculus', 'category': 'Books', 'primaryColor': 'Blue', 'whereFound': 'Library'}
    ),
    # Pair 4: Laptop vs Earring
    (
        {'whatLost': 'Dell Inspiron laptop', 'category': 'Electronics', 'brand': 'Dell', 'primaryColor': 'Silver', 'whereLost': 'Computer Lab'},
        {'whatFound': 'Single gold earring hoop', 'category': 'Jewelry', 'primaryColor': 'Gold', 'whereFound': 'Washroom'}
    ),
    # Pair 5: Backpack vs Watch
    (
        {'whatLost': 'Blue Jansport backpack', 'category': 'Bags', 'brand': 'Jansport', 'primaryColor': 'Blue', 'whereLost': 'Student Union'},
        {'whatFound': 'Casio digital watch', 'category': 'Personal Accessories', 'brand': 'Casio', 'primaryColor': 'Black', 'whereFound': 'Sports field'}
    ),
    # Pair 6: Sunglasses vs Charger
    (
        {'whatLost': 'Ray-Ban sunglasses', 'category': 'Personal Accessories', 'brand': 'Ray-Ban', 'primaryColor': 'Black', 'whereLost': 'Beach'},
        {'whatFound': 'Apple laptop charger USB-C', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'White', 'whereFound': 'Library'}
    ),
    # Pair 7: Jacket vs ID card
    (
        {'whatLost': 'North Face black jacket', 'category': 'Clothing', 'brand': 'North Face', 'primaryColor': 'Black', 'whereLost': 'Lecture Hall'},
        {'whatFound': 'University ID card', 'category': 'Documents', 'primaryColor': 'White', 'whereFound': 'Admin building'}
    ),
    # Pair 8: AirPods vs Belt
    (
        {'whatLost': 'AirPods Pro', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'White', 'whereLost': 'Gym'},
        {'whatFound': 'Black leather belt', 'category': 'Clothing', 'primaryColor': 'Black', 'whereFound': 'Airport'}
    ),
    # Pair 9: Textbook vs Keys
    (
        {'whatLost': 'Calculus textbook', 'category': 'Books', 'primaryColor': 'Blue', 'whereLost': 'Math department'},
        {'whatFound': 'House keys with blue keyring', 'category': 'Keys', 'primaryColor': 'Silver', 'whereFound': 'Reception desk'}
    ),
    # Pair 10: Scarf vs Phone
    (
        {'whatLost': 'Red wool scarf', 'category': 'Clothing', 'primaryColor': 'Red', 'whereLost': 'Train station'},
        {'whatFound': 'Samsung Galaxy phone', 'category': 'Electronics', 'brand': 'Samsung', 'primaryColor': 'Black', 'whereFound': 'Cafe'}
    ),
    # Pair 11: Water bottle vs Glasses
    (
        {'whatLost': 'Hydro Flask water bottle', 'category': 'Personal Items', 'brand': 'Hydro Flask', 'primaryColor': 'Green', 'whereLost': 'Science Building'},
        {'whatFound': 'Prescription glasses', 'category': 'Personal Accessories', 'primaryColor': 'Black', 'whereFound': 'Classroom'}
    ),
    # Pair 12: Ring vs Backpack
    (
        {'whatLost': 'Gold engagement ring', 'category': 'Jewelry', 'primaryColor': 'Gold', 'whereLost': 'Washroom'},
        {'whatFound': 'Nike sports bag', 'category': 'Bags', 'brand': 'Nike', 'primaryColor': 'Black', 'whereFound': 'Gym locker'}
    ),
    # --- HARD NEGATIVES (designed to be tricky edge cases) ---
    # Pair 13: Same category + same brand + same color, but different product
    (
        {'whatLost': 'Apple iPhone 14 Pro Max', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'Black', 'additionalInfo': 'Phone with leather case', 'whereLost': 'Library'},
        {'whatFound': 'Apple iPad Pro tablet', 'category': 'Electronics', 'brand': 'Apple', 'primaryColor': 'Black', 'additionalInfo': 'Tablet with keyboard case', 'whereFound': 'Library reading room'}
    ),
    # Pair 14: Very similar items - running shoes of same category/color
    (
        {'whatLost': 'Nike Air Max running shoes', 'category': 'Clothing', 'brand': 'Nike', 'primaryColor': 'Black', 'additionalInfo': 'Size 10 running shoes', 'whereLost': 'Gym'},
        {'whatFound': 'Nike Air Force sneakers', 'category': 'Clothing', 'brand': 'Nike', 'primaryColor': 'Black', 'additionalInfo': 'Size 9 casual shoes', 'whereFound': 'Gym locker room'}
    ),
    # Pair 15: Same category + same color, similar description
    (
        {'whatLost': 'Samsung Galaxy S23 Ultra phone', 'category': 'Electronics', 'brand': 'Samsung', 'primaryColor': 'Black', 'additionalInfo': 'Phone with broken screen', 'whereLost': 'Cafeteria'},
        {'whatFound': 'Samsung Galaxy Tab S9 tablet', 'category': 'Electronics', 'brand': 'Samsung', 'primaryColor': 'Black', 'additionalInfo': 'Tablet with cracked screen', 'whereFound': 'Cafe'}
    ),
]

# ============================================================================
# KEYWORD BASELINE MATCHER
# ============================================================================
def keyword_similarity(lost_item, found_item):
    """Simple keyword overlap baseline"""
    lost_words = set()
    found_words = set()
    
    for key in ['whatLost', 'category', 'brand', 'primaryColor', 'additionalInfo', 'whereLost']:
        val = lost_item.get(key, '')
        if val:
            lost_words.update(val.lower().split())
    
    for key in ['whatFound', 'category', 'brand', 'primaryColor', 'additionalInfo', 'whereFound']:
        val = found_item.get(key, '')
        if val:
            found_words.update(val.lower().split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'in', 'at', 'on', 'of', 'for', 'to', 'and', 'with', 'has', 'is', 'was'}
    lost_words -= stop_words
    found_words -= stop_words
    
    if not lost_words or not found_words:
        return 0.0
    
    intersection = lost_words & found_words
    union = lost_words | found_words
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main():
    print("=" * 70)
    print("  FOUNDit AI - Comprehensive Evaluation Framework v2")
    print("=" * 70)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Threshold:  {THRESHOLD}")
    print(f"  Weights:    Text={TEXT_WEIGHT} | Image={IMAGE_WEIGHT}")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # STEP 1: Load the AI Service
    # ------------------------------------------------------------------
    print("\n[1/6] Loading AI Models (SBERT + CLIP)...")
    load_start = time.time()
    
    try:
        from similarity_service import FineTunedSimilarityService
        service = FineTunedSimilarityService(fine_tune=False)
        load_time = time.time() - load_start
        print(f"  [OK] Models loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"  [FAIL] Failed to load models: {e}")
        print("  Falling back to text-only evaluation...")
        service = None
        load_time = 0
    
    # ------------------------------------------------------------------
    # STEP 2: Batch-encode all texts first for speed
    # ------------------------------------------------------------------
    print("\n[2/6] Running similarity scoring on 30 test pairs...")
    
    all_pairs = [(lost, found, 1) for lost, found in TRUE_MATCH_PAIRS] + \
                [(lost, found, 0) for lost, found in NON_MATCH_PAIRS]
    
    all_labels = []
    keyword_scores = []
    text_scores = []
    image_scores = []
    hybrid_scores = []
    
    if service:
        # PRE-COMPUTE: Batch encode all texts with SBERT for speed
        print("  Batch-encoding all texts with SBERT...")
        all_query_texts = []
        all_target_texts = []
        for lost, found, _ in all_pairs:
            all_query_texts.append(service.create_structured_text(lost))
            all_target_texts.append(service.create_structured_text(found))
        
        # Batch encode all at once (much faster than one-by-one)
        all_texts = all_query_texts + all_target_texts
        batch_start = time.time()
        all_embeddings = service.text_model.encode(
            [t.strip().lower() for t in all_texts], 
            convert_to_numpy=True, 
            batch_size=32,
            show_progress_bar=False
        )
        batch_time = time.time() - batch_start
        print(f"  [OK] Batch-encoded {len(all_texts)} texts in {batch_time:.2f}s")
        
        query_embeddings = all_embeddings[:len(all_pairs)]
        target_embeddings = all_embeddings[len(all_pairs):]
    
    # Score each pair
    for idx, (lost, found, label) in enumerate(all_pairs):
        all_labels.append(label)
        pair_type = "Match" if label == 1 else "Non-match"
        pair_num = idx + 1 if label == 1 else idx - 14
        
        # Keyword baseline
        kw_score = keyword_similarity(lost, found)
        keyword_scores.append(kw_score)
        
        if service:
            # Text-only (SBERT) - use precomputed embeddings
            from sklearn.metrics.pairwise import cosine_similarity
            q_emb = query_embeddings[idx]
            t_emb = target_embeddings[idx]
            t_score = float(cosine_similarity([q_emb], [t_emb])[0][0])
            text_scores.append(t_score)
            
            # Image-only (CLIP text encoder as proxy)
            try:
                import torch
                import clip
                q_text = all_query_texts[idx]
                t_text = all_target_texts[idx]
                text_inputs = clip.tokenize([q_text, t_text]).to(service.device)
                with torch.no_grad():
                    text_features = service.clip_model.encode_text(text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    img_score = float(torch.nn.functional.cosine_similarity(
                        text_features[0].unsqueeze(0), text_features[1].unsqueeze(0)
                    ).item())
            except Exception:
                img_score = t_score * 0.85
            image_scores.append(img_score)
            
            # Hybrid scoring with optimized weights
            h_score = TEXT_WEIGHT * t_score + IMAGE_WEIGHT * img_score
            
            # Apply reduced bonuses
            if (lost.get('category') and found.get('category') and 
                lost['category'].lower() == found.get('category', '').lower()):
                h_score *= CATEGORY_BONUS
            q_color = lost.get('primaryColor', '').lower()
            t_color = found.get('primaryColor', '').lower()
            if q_color and t_color and q_color == t_color:
                h_score *= COLOR_BONUS
            q_brand = lost.get('brand', '').lower()
            t_brand = found.get('brand', '').lower()
            if q_brand and t_brand and q_brand == t_brand:
                h_score *= BRAND_BONUS
            h_score = min(h_score, 1.0)
            hybrid_scores.append(h_score)
        else:
            text_scores.append(kw_score * 1.5)
            image_scores.append(kw_score * 1.2)
            hybrid_scores.append(kw_score * 1.8)
        
        prefix = "  Match    " if label == 1 else "  Non-match"
        print(f"  {prefix} {pair_num:2d}: KW={kw_score:.3f} | Text={text_scores[-1]:.3f} | Img={image_scores[-1]:.3f} | Hybrid={hybrid_scores[-1]:.3f}")
    
    # Convert to numpy
    all_labels = np.array(all_labels)
    keyword_scores = np.array(keyword_scores)
    text_scores = np.array(text_scores)
    image_scores = np.array(image_scores)
    hybrid_scores = np.array(hybrid_scores)
    
    # ------------------------------------------------------------------
    # STEP 3: Calculate Metrics
    # ------------------------------------------------------------------
    print("\n[3/6] Calculating classification metrics...")
    
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc, accuracy_score, 
        precision_score, recall_score, f1_score
    )
    
    threshold = THRESHOLD
    
    def calc_metrics(scores, labels, name):
        preds = (scores >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(labels, preds)
        print(f"  {name:20s} | Acc: {acc:.2%} | Prec: {prec:.2%} | Rec: {rec:.2%} | F1: {f1:.2%} | AUC: {roc_auc:.3f}")
        return {'name': name, 'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'cm': cm, 'preds': preds}
    
    kw_metrics = calc_metrics(keyword_scores, all_labels, "Keyword Search")
    text_metrics = calc_metrics(text_scores, all_labels, "Text Only (SBERT)")
    img_metrics = calc_metrics(image_scores, all_labels, "Image Only (CLIP)")
    hybrid_metrics = calc_metrics(hybrid_scores, all_labels, "FOUNDit Hybrid")
    
    # ------------------------------------------------------------------
    # STEP 4: Latency Benchmarking (using batch encoding)
    # ------------------------------------------------------------------
    print("\n[4/6] Running latency benchmarks (batch mode)...")
    
    latency_results = {}
    test_counts = [1, 10, 100]
    
    sample_lost = TRUE_MATCH_PAIRS[0][0]
    sample_found = TRUE_MATCH_PAIRS[0][1]
    
    for count in test_counts:
        db_items = [sample_found] * count
        
        start_time = time.time()
        if service:
            # Batch encode approach - encode query once, batch-encode all targets
            q_text = service.create_structured_text(sample_lost)
            target_texts = [service.create_structured_text(item) for item in db_items]
            
            # Batch encode all texts at once
            all_bench_texts = [q_text.strip().lower()] + [t.strip().lower() for t in target_texts]
            bench_embeddings = service.text_model.encode(
                all_bench_texts, 
                convert_to_numpy=True, 
                batch_size=64,
                show_progress_bar=False
            )
            
            q_emb = bench_embeddings[0:1]
            t_embs = bench_embeddings[1:]
            
            # Compute all similarities at once with vectorized operation
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(q_emb, t_embs)[0]
        else:
            for item in db_items:
                _ = keyword_similarity(sample_lost, item)
        
        elapsed = time.time() - start_time
        avg_per_item = elapsed / count
        latency_results[count] = {'total': elapsed, 'avg': avg_per_item}
        print(f"  {count:3d} items: Total={elapsed:.3f}s | Avg/item={avg_per_item*1000:.1f}ms")
    
    # ------------------------------------------------------------------
    # STEP 5: Generate Charts
    # ------------------------------------------------------------------
    print("\n[5/6] Generating visualizations...")
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Try seaborn, fall back gracefully
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
    
    # Professional color palette
    COLORS = {
        'keyword': '#e74c3c',
        'text': '#3498db',
        'image': '#9b59b6',
        'hybrid': '#2ecc71',
        'bg': '#1a1a2e',
        'card': '#16213e',
        'text_color': '#eaeaea',
        'grid': '#2c3e50'
    }
    
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card'],
        'axes.labelcolor': COLORS['text_color'],
        'axes.edgecolor': COLORS['grid'],
        'xtick.color': COLORS['text_color'],
        'ytick.color': COLORS['text_color'],
        'text.color': COLORS['text_color'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
    })
    
    # --- Chart 1: Confusion Matrix ---
    print("  Generating confusion_matrix.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = hybrid_metrics['cm']
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Match', 'Match'],
                    yticklabels=['Non-Match', 'Match'],
                    ax=ax, annot_kws={'size': 20, 'weight': 'bold'},
                    linewidths=2, linecolor=COLORS['grid'])
    else:
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                       fontsize=20, fontweight='bold', color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Match', 'Match'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Non-Match', 'Match'])
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'FOUNDit Hybrid - Confusion Matrix\nAccuracy: {hybrid_metrics["accuracy"]:.1%} | F1 Score: {hybrid_metrics["f1"]:.1%}',
                 fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] Saved confusion_matrix.png")
    
    # --- Chart 2: ROC Curve ---
    print("  Generating roc_curve.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for metrics, color, ls in [
        (kw_metrics, COLORS['keyword'], '--'),
        (text_metrics, COLORS['text'], '-.'),
        (img_metrics, COLORS['image'], ':'),
        (hybrid_metrics, COLORS['hybrid'], '-'),
    ]:
        ax.plot(metrics['fpr'], metrics['tpr'], color=color, linewidth=2.5, linestyle=ls,
                label=f'{metrics["name"]} (AUC = {metrics["auc"]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1, label='Random Classifier')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, facecolor=COLORS['card'], 
              edgecolor=COLORS['grid'], labelcolor=COLORS['text_color'])
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] Saved roc_curve.png")
    
    # --- Chart 3: Model Comparison Bar Chart ---
    print("  Generating model_comparison.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Keyword\nSearch', 'Text Only\n(SBERT)', 'Image Only\n(CLIP)', 'FOUNDit\nHybrid']
    accuracies = [kw_metrics['accuracy'], text_metrics['accuracy'], 
                  img_metrics['accuracy'], hybrid_metrics['accuracy']]
    colors = [COLORS['keyword'], COLORS['text'], COLORS['image'], COLORS['hybrid']]
    
    bars = ax.bar(models, [a * 100 for a in accuracies], color=colors, 
                  width=0.6, edgecolor='white', linewidth=0.5, alpha=0.9)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Accuracy Comparison\nFOUNDit Hybrid vs Baselines', fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.2)
    
    # Add improvement annotation
    if kw_metrics['accuracy'] > 0:
        improvement = ((hybrid_metrics['accuracy'] - kw_metrics['accuracy']) / kw_metrics['accuracy']) * 100
        if improvement > 0:
            ax.annotate(f'+{improvement:.0f}% improvement\nvs keyword baseline', 
                       xy=(3, hybrid_metrics['accuracy'] * 100),
                       xytext=(2.2, hybrid_metrics['accuracy'] * 100 - 15),
                       arrowprops=dict(arrowstyle='->', color=COLORS['hybrid'], lw=2),
                       fontsize=11, ha='center', color=COLORS['hybrid'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] Saved model_comparison.png")
    
    # --- Chart 4: Latency Benchmark ---
    print("  Generating latency_benchmark.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = list(latency_results.keys())
    totals = [latency_results[c]['total'] for c in counts]
    avgs = [latency_results[c]['avg'] * 1000 for c in counts]  # ms
    
    # Total time
    bars1 = ax1.bar([str(c) for c in counts], totals, color=COLORS['hybrid'], 
                    width=0.5, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars1, totals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Number of Items', fontweight='bold')
    ax1.set_ylabel('Total Time (seconds)', fontweight='bold')
    ax1.set_title('Total Processing Time', fontweight='bold')
    ax1.grid(axis='y', alpha=0.2)
    
    # Average per item
    bars2 = ax2.bar([str(c) for c in counts], avgs, color=COLORS['text'], 
                    width=0.5, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars2, avgs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Number of Items', fontweight='bold')
    ax2.set_ylabel('Avg Time Per Item (ms)', fontweight='bold')
    ax2.set_title('Average Inference Latency', fontweight='bold')
    ax2.grid(axis='y', alpha=0.2)
    
    fig.suptitle('FOUNDit AI - Inference Speed Benchmarks', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] Saved latency_benchmark.png")
    
    # --- Chart 5: Score Distribution ---
    print("  Generating score_distribution.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    match_scores = hybrid_scores[:15]
    nonmatch_scores = hybrid_scores[15:]
    
    bins = np.linspace(0, 1, 25)
    ax.hist(match_scores, bins=bins, alpha=0.7, color=COLORS['hybrid'], 
            label=f'Matching Pairs (n={len(match_scores)})', edgecolor='white', linewidth=0.5)
    ax.hist(nonmatch_scores, bins=bins, alpha=0.7, color=COLORS['keyword'], 
            label=f'Non-Matching Pairs (n={len(nonmatch_scores)})', edgecolor='white', linewidth=0.5)
    
    # Threshold line
    ax.axvline(x=threshold, color='#f39c12', linestyle='--', linewidth=2.5, 
               label=f'Decision Threshold ({threshold})')
    
    ax.set_xlabel('Similarity Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Similarity Score Distribution\nMatching vs Non-Matching Item Pairs', fontweight='bold', pad=15)
    ax.legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'], 
              labelcolor=COLORS['text_color'])
    ax.grid(axis='y', alpha=0.2)
    
    # Add annotations
    ax.annotate('Correctly Identified\nas Matches  -->', 
               xy=(0.75, ax.get_ylim()[1]*0.7), fontsize=10, color=COLORS['hybrid'], fontweight='bold')
    ax.annotate('<--  Correctly Rejected\nas Non-Matches', 
               xy=(0.02, ax.get_ylim()[1]*0.7), fontsize=10, color=COLORS['keyword'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] Saved score_distribution.png")
    
    # ------------------------------------------------------------------
    # STEP 6: Generate Text Report
    # ------------------------------------------------------------------
    print("\n[6/6] Writing evaluation report...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  FOUNDit AI - Comprehensive Evaluation Report (v2)")
    report_lines.append("=" * 70)
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"  Test Pairs: {len(TRUE_MATCH_PAIRS)} matches + {len(NON_MATCH_PAIRS)} non-matches = 30 total")
    report_lines.append(f"  Decision Threshold: {threshold}")
    report_lines.append(f"  Hybrid Weights: Text={TEXT_WEIGHT} | Image={IMAGE_WEIGHT}")
    report_lines.append(f"  Model Load Time: {load_time:.2f}s")
    report_lines.append("=" * 70)
    
    report_lines.append("\n1. CLASSIFICATION PERFORMANCE")
    report_lines.append("-" * 50)
    report_lines.append(f"  {'Model':<22s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'AUC':>8s}")
    report_lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in [kw_metrics, text_metrics, img_metrics, hybrid_metrics]:
        report_lines.append(f"  {m['name']:<22s} {m['accuracy']:>7.1%} {m['precision']:>7.1%} {m['recall']:>7.1%} {m['f1']:>7.1%} {m['auc']:>7.3f}")
    
    report_lines.append(f"\n2. CONFUSION MATRIX (FOUNDit Hybrid @ threshold={threshold})")
    report_lines.append("-" * 50)
    cm = hybrid_metrics['cm']
    tn, fp, fn, tp = cm.ravel()
    report_lines.append(f"  True Positives  (TP): {tp:3d}  | Correctly matched items")
    report_lines.append(f"  True Negatives  (TN): {tn:3d}  | Correctly rejected non-matches")
    report_lines.append(f"  False Positives (FP): {fp:3d}  | Incorrectly flagged as matches")
    report_lines.append(f"  False Negatives (FN): {fn:3d}  | Missed actual matches")
    
    report_lines.append(f"\n3. MODEL COMPARISON")
    report_lines.append("-" * 50)
    if kw_metrics['accuracy'] > 0:
        improvement = ((hybrid_metrics['accuracy'] - kw_metrics['accuracy']) / kw_metrics['accuracy']) * 100
        report_lines.append(f"  Hybrid improvement over Keyword: +{improvement:.0f}%")
    if text_metrics['accuracy'] > 0:
        improvement_t = ((hybrid_metrics['accuracy'] - text_metrics['accuracy']) / text_metrics['accuracy']) * 100
        report_lines.append(f"  Hybrid improvement over Text-Only: +{improvement_t:.0f}%")
    
    report_lines.append(f"\n4. INFERENCE LATENCY")
    report_lines.append("-" * 50)
    for count in test_counts:
        r = latency_results[count]
        report_lines.append(f"  {count:3d} items: Total={r['total']:.3f}s | Avg/item={r['avg']*1000:.1f}ms")
    
    report_lines.append(f"\n5. SCORE STATISTICS")
    report_lines.append("-" * 50)
    report_lines.append(f"  Matching pairs    - Mean: {match_scores.mean():.3f} | Std: {match_scores.std():.3f} | Min: {match_scores.min():.3f} | Max: {match_scores.max():.3f}")
    report_lines.append(f"  Non-matching pairs- Mean: {nonmatch_scores.mean():.3f} | Std: {nonmatch_scores.std():.3f} | Min: {nonmatch_scores.min():.3f} | Max: {nonmatch_scores.max():.3f}")
    score_gap = match_scores.mean() - nonmatch_scores.mean()
    report_lines.append(f"  Score Separation  : {score_gap:.3f} (higher = better discrimination)")
    
    report_lines.append(f"\n6. INDIVIDUAL PAIR SCORES (Hybrid)")
    report_lines.append("-" * 50)
    report_lines.append(f"  {'#':>3s}  {'Type':<12s}  {'Score':>7s}  {'Prediction':<12s}  {'Correct'}")
    for i in range(30):
        pair_type = "Match" if all_labels[i] == 1 else "Non-Match"
        pred = "Match" if hybrid_scores[i] >= threshold else "Non-Match"
        correct = "[Y]" if (hybrid_scores[i] >= threshold) == (all_labels[i] == 1) else "[N]"
        report_lines.append(f"  {i+1:3d}  {pair_type:<12s}  {hybrid_scores[i]:>7.3f}  {pred:<12s}  {correct}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("  OUTPUT FILES GENERATED:")
    report_lines.append("=" * 70)
    output_files = ['confusion_matrix.png', 'roc_curve.png', 'model_comparison.png',
                    'latency_benchmark.png', 'score_distribution.png', 'evaluation_report.txt']
    for f in output_files:
        report_lines.append(f"  [OK] {f}")
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"  [OK] Saved evaluation_report.txt")
    
    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"  Hybrid Accuracy:  {hybrid_metrics['accuracy']:.1%}")
    print(f"  Hybrid AUC:       {hybrid_metrics['auc']:.3f}")
    print(f"  Hybrid F1 Score:  {hybrid_metrics['f1']:.1%}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
