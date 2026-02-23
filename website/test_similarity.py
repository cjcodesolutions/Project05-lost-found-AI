#!/usr/bin/env python3
"""
Test script for the Lost & Found Similarity System with Fine-tuned Models
Run this script to verify that all components are working correctly.
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"PyTorch: {e}")
        return False
    
    try:
        import clip
        print("CLIP: Available")
    except ImportError as e:
        print(f"CLIP: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("SentenceTransformers: Available")
    except ImportError as e:
        print(f"SentenceTransformers: {e}")
        return False
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError as e:
        print(f"NumPy: {e}")
        return False
    
    try:
        from PIL import Image
        print("Pillow: Available")
    except ImportError as e:
        print(f"Pillow: {e}")
        return False
    
    return True

def test_similarity_service():
    """Test the fine-tuned similarity service functionality"""
    print("\nTesting Fine-tuned Similarity Service...")
    
    try:
        # Import the fine-tuned similarity service
        sys.path.append('website')
        from similarity_service import FineTunedSimilarityService
        
        # Initialize with pre-trained models (fine_tune=False)
        similarity_service = FineTunedSimilarityService(fine_tune=False)
        print("Fine-tuned similarity service imported successfully")
        
        # Test text embedding
        test_text = "black iPhone with cracked screen"
        text_embedding = similarity_service.get_text_embedding(test_text)
        
        if text_embedding is not None:
            print(f"Text embedding: Shape {text_embedding.shape}")
        else:
            print("Text embedding failed")
            return False
        
        # Test structured text creation
        test_item_data = {
            'whatLost': 'iPhone 13',
            'category': 'Electronics',
            'brand': 'Apple',
            'primaryColor': 'Black',
            'additionalInfo': 'Cracked screen on the bottom right corner'
        }
        
        structured_text = similarity_service.create_structured_text(test_item_data)
        print(f"Structured text creation: '{structured_text}'")
        
        # Test image embedding with a simple test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            # For testing, we need to simulate an image URL
            # In real usage, this would be an actual S3 URL
            print("Image embedding test skipped (requires actual image URL)")
                
        except Exception as e:
            print(f"Image embedding test skipped: {e}")
        
        # Test similarity calculation
        try:
            test_text_2 = "black smartphone with damaged display"
            text_embedding_2 = similarity_service.get_text_embedding(test_text_2)
            
            # Calculate similarity manually using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_score = cosine_similarity([text_embedding], [text_embedding_2])[0][0]
            print(f"Similarity calculation: {similarity_score:.3f}")
            
            if similarity_score > 0.5:
                print("Similar texts detected correctly (score > 0.5)")
            else:
                print("Low similarity score (might need fine-tuning)")
                
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return False
        
        # Check if fine-tuned models exist
        if os.path.exists('models/finetuned_sentence_bert'):
            print("Fine-tuned Sentence-BERT model found")
        else:
            print("Fine-tuned Sentence-BERT not found (using pre-trained)")
        
        if os.path.exists('models/finetuned_clip.pt'):
            print("Fine-tuned CLIP model found")
        else:
            print("Fine-tuned CLIP not found (using pre-trained)")
        
        return True
        
    except Exception as e:
        print(f"Similarity service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_connection():
    """Test database connection"""
    print("\nTesting Database Connection...")
    
    try:
        from pymongo import MongoClient
        connection_string = os.getenv("MONGO_URI")     
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        db = client["users"]
        
        # Test connection
        client.admin.command('ping')
        print("Database connection successful")
        
        # Test collections
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
        
        # Test basic query
        lost_items_count = db.lostItems.count_documents({})
        found_items_count = db.foundItems.count_documents({})
        print(f"Lost items: {lost_items_count}, Found items: {found_items_count}")
        
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def test_s3_connection():
    """Test S3 connection"""
    print("\nTesting S3 Connection...")
    
    try:
        import boto3
        from dotenv import load_dotenv
        
        load_dotenv()
        
        S3_BUCKET = os.getenv('S3_BUCKET_NAME')
        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        AWS_REGION = 'eu-north-1'
        
        if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
            print("S3 credentials not found in environment variables")
            return False
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Test bucket access
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"S3 bucket '{S3_BUCKET}' accessible")
        
        return True
        
    except Exception as e:
        print(f"S3 connection failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can start"""
    print("\nTesting Flask Application...")
    
    try:
        # Try to import the app from parent directory
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import app
        print("Flask app imported successfully")
        
        # Test if app has required routes
        with app.test_client() as client:
            routes_to_test = ['/', '/welcome']
            
            for route in routes_to_test:
                try:
                    response = client.get(route)
                    if response.status_code in [200, 302]:
                        print(f"Route {route}: {response.status_code}")
                    else:
                        print(f"Route {route}: {response.status_code}")
                except Exception as e:
                    print(f"Route {route}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Flask app test failed: {e}")
        return False

def run_performance_test():
    """Run a basic performance test"""
    print("\nRunning Performance Test...")
    
    try:
        import time
        sys.path.append('website')
        from similarity_service import FineTunedSimilarityService
        
        similarity_service = FineTunedSimilarityService(fine_tune=False)
        
        # Test text processing speed
        start_time = time.time()
        for i in range(10):
            embedding = similarity_service.get_text_embedding(f"test item {i}")
        text_time = time.time() - start_time
        print(f"Text processing: {text_time:.2f}s for 10 items ({text_time/10:.3f}s per item)")
        
        # Test structured text creation speed
        start_time = time.time()
        for i in range(100):
            test_data = {
                'whatLost': f'Item {i}',
                'category': 'Electronics',
                'brand': 'TestBrand',
                'primaryColor': 'Black'
            }
            text = similarity_service.create_structured_text(test_data)
        struct_time = time.time() - start_time
        print(f"Structured text creation: {struct_time:.2f}s for 100 items ({struct_time/100:.4f}s per item)")
        
        return True
        
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Lost & Found Fine-tuned Similarity System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Fine-tuned Similarity Service", test_similarity_service),
        ("Database Connection", test_database_connection),
        ("S3 Connection", test_s3_connection),
        ("Flask Application", test_flask_app),
        ("Performance", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! Your fine-tuned similarity system is ready to use.")
    else:
        print(f"\n{total - passed} test(s) failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        if not results.get("Package Imports", True):
            print("- Install missing packages: pip install -r finetune_requirements.txt")
        if not results.get("Database Connection", True):
            print("- Check MongoDB connection string in .env file")
        if not results.get("S3 Connection", True):
            print("- Verify AWS credentials in .env file")
        if not results.get("Fine-tuned Similarity Service", True):
            print("- Check if similarity_service.py is properly updated")
            print("- Run: python finetune_models.py (to create fine-tuned models)")

if __name__ == "__main__":
    main()