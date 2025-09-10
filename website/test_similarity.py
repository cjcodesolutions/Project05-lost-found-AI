#!/usr/bin/env python3
"""
Test script for the Lost & Found Similarity System
Run this script to verify that all components are working correctly.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import clip
        print("✅ CLIP: Available")
    except ImportError as e:
        print(f"❌ CLIP: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers: Available")
    except ImportError as e:
        print(f"❌ SentenceTransformers: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow: Available")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        return False
    
    return True

def test_similarity_service():
    """Test the similarity service functionality"""
    print("\nTesting Similarity Service...")
    
    try:
        # Import the similarity service
        sys.path.append('website')
        from similarity_service import similarity_service
        print("✅ Similarity service imported successfully")
        
        # Test text embedding
        test_text = "black iPhone with cracked screen"
        text_embedding = similarity_service.get_text_embedding(test_text)
        
        if text_embedding is not None:
            print(f"✅ Text embedding: Shape {text_embedding.shape}")
        else:
            print("❌ Text embedding failed")
            return False
        
        # Test image embedding with a simple test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            image_embedding = similarity_service.get_image_embedding(test_image)
            
            if image_embedding is not None:
                print(f"✅ Image embedding: Shape {image_embedding.shape}")
            else:
                print("❌ Image embedding failed")
                return False
                
        except Exception as e:
            print(f"❌ Image embedding test failed: {e}")
            return False
        
        # Test similarity calculation
        try:
            similarity_score = similarity_service.calculate_similarity(text_embedding[:512], text_embedding[:512])
            print(f"✅ Similarity calculation: {similarity_score}")
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Similarity service test failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\nTesting Database Connection...")
    
    try:
        from pymongo import MongoClient
        
        # This would need your actual connection string
        connection_string = "mongodb+srv://cjcodesolutions:Abc12345@cluster0.fbte9k0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        db = client["users"]
        
        # Test connection
        client.admin.command('ismaster')
        print("✅ Database connection successful")
        
        # Test collections
        collections = db.list_collection_names()
        print(f"✅ Available collections: {collections}")
        
        # Test basic query
        lost_items_count = db.lostItems.count_documents({})
        found_items_count = db.foundItems.count_documents({})
        print(f"✅ Lost items: {lost_items_count}, Found items: {found_items_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
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
        AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
        
        if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
            print("❌ S3 credentials not found in environment variables")
            return False
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Test bucket access
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"✅ S3 bucket '{S3_BUCKET}' accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can start"""
    print("\nTesting Flask Application...")
    
    try:
        # Try to import the app
        from main import app
        print("✅ Flask app imported successfully")
        
        # Test if app has required routes
        with app.test_client() as client:
            # Test basic routes
            routes_to_test = ['/', '/welcome', '/test-similarity']
            
            for route in routes_to_test:
                try:
                    response = client.get(route)
                    if response.status_code in [200, 302]:
                        print(f"✅ Route {route}: {response.status_code}")
                    else:
                        print(f"⚠️ Route {route}: {response.status_code}")
                except Exception as e:
                    print(f"❌ Route {route}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def run_performance_test():
    """Run a basic performance test"""
    print("\nRunning Performance Test...")
    
    try:
        import time
        sys.path.append('website')
        from similarity_service import similarity_service
        
        # Test text processing speed
        start_time = time.time()
        for i in range(10):
            embedding = similarity_service.get_text_embedding(f"test item {i}")
        text_time = time.time() - start_time
        print(f"✅ Text processing: {text_time:.2f}s for 10 items ({text_time/10:.3f}s per item)")
        
        # Test image processing speed
        from PIL import Image
        import numpy as np
        
        start_time = time.time()
        for i in range(5):
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            embedding = similarity_service.get_image_embedding(test_image)
        image_time = time.time() - start_time
        print(f"✅ Image processing: {image_time:.2f}s for 5 images ({image_time/5:.3f}s per image)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Lost & Found Similarity System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Similarity Service", test_similarity_service),
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
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your similarity system is ready to use.")
        print("\nNext steps:")
        print("1. Start your Flask app: python main.py")
        print("2. Submit a lost item with image and description")
        print("3. Check if the suggestions page appears")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        if not results.get("Package Imports", True):
            print("- Install missing packages: pip install -r requirements.txt")
        if not results.get("Database Connection", True):
            print("- Check MongoDB connection string in .env file")
        if not results.get("S3 Connection", True):
            print("- Verify AWS credentials in .env file")

if __name__ == "__main__":
    main()