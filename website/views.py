from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv
from bson import ObjectId

# Import the similarity service
from .similarity_service import similarity_service

load_dotenv()

views = Blueprint('views', __name__)

# AWS S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# File upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_s3(file, bucket_name):
    """Upload file to S3 and return presigned URL"""
    print(f"=== S3 UPLOAD DEBUG ===")
    print(f"File: {file}")
    print(f"Filename: {file.filename if file else 'None'}")
    print(f"Bucket: {bucket_name}")
    print(f"AWS Access Key ID: {AWS_ACCESS_KEY_ID[:10]}..." if AWS_ACCESS_KEY_ID else "Not set")
    print(f"AWS Region: {AWS_REGION}")
    
    if not file or file.filename == '':
        print("No file provided")
        return None
    
    if not allowed_file(file.filename):
        print(f"File type not allowed: {file.filename}")
        raise ValueError("File type not allowed. Use PNG, JPG, JPEG, GIF, or WEBP.")
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    print(f"File size: {file_size} bytes")
    
    if file_size > MAX_FILE_SIZE:
        print(f"File too large: {file_size} > {MAX_FILE_SIZE}")
        raise ValueError("File too large. Maximum size is 5MB.")
    
    # Check AWS credentials
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET:
        print("Missing AWS credentials or bucket name")
        print(f"Access Key: {'Set' if AWS_ACCESS_KEY_ID else 'Not set'}")
        print(f"Secret Key: {'Set' if AWS_SECRET_ACCESS_KEY else 'Not set'}")
        print(f"Bucket: {'Set' if S3_BUCKET else 'Not set'}")
        raise ValueError("AWS configuration incomplete")
    
    try:
        # Test S3 connection first
        print("Testing S3 connection...")
        s3_client.head_bucket(Bucket=bucket_name)
        print("S3 bucket accessible")
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"lost-items/{uuid.uuid4()}.{file_extension}"
        
        print(f"Original filename: {filename}")
        print(f"Unique filename: {unique_filename}")
        print(f"Content type: {file.content_type}")
        
        # Upload to S3 (private)
        print("Starting S3 upload...")
        s3_client.upload_fileobj(
            file, 
            bucket_name, 
            unique_filename,
            ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
        )
        print("S3 upload successful")
        
        # Generate presigned URL (valid for 1 week)
        print("Generating presigned URL...")
        file_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': unique_filename},
            ExpiresIn=604800 # 1 week
        )
        print(f"Presigned URL generated: {file_url[:100]}...")
        return file_url
        
    except NoCredentialsError as e:
        print(f"AWS credentials error: {e}")
        raise ValueError("AWS credentials not available")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"AWS S3 ClientError - Code: {error_code}, Message: {error_message}")
        print(f"Full error: {e}")
        
        if error_code == 'NoSuchBucket':
            raise ValueError(f"S3 bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            raise ValueError("Access denied to S3 bucket. Check your AWS permissions")
        elif error_code == 'InvalidAccessKeyId':
            raise ValueError("Invalid AWS Access Key ID")
        elif error_code == 'SignatureDoesNotMatch':
            raise ValueError("Invalid AWS Secret Access Key")
        else:
            raise ValueError(f"S3 error: {error_code} - {error_message}")
    except Exception as e:
        print(f"Unexpected error during S3 upload: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to upload image to S3: {str(e)}")

@views.route('/')
def home():
    return render_template("home.html")

@views.route('/welcome')
def welcome():
    return render_template("welcome.html")
@views.route('/contact')
def contact():
    """Render the contact page"""
    return render_template('contact.html')
@views.route('/submit-initial-lost', methods=['POST'])
def submit_initial_lost():
    """Handle the simple form submission from welcome page"""
    print("=== DEBUG: Initial form submission received ===")
    print(f"Form data: {dict(request.form)}")
    
    try:
        db = current_app.db
        
        # Get form data
        item_type = request.form.get('itemType')
        location = request.form.get('location')
        
        print(f"Item Type: '{item_type}', Location: '{location}'")
        
        # Validate required fields
        if not item_type or not location:
            flash('Please fill in all required fields', 'error')
            return redirect(url_for('views.welcome'))
        
        # Create initial lost item record
        initial_data = {
            'itemType': item_type,
            'location': location,
            'submissionTime': datetime.now(),
            'status': 'incomplete',
            'type': 'lost'
        }
        
        # Insert into database
        result = db.lostItems.insert_one(initial_data)
        item_id = str(result.inserted_id)
        
        print(f"Initial data inserted with ID: {item_id}")
        
        # Redirect to detailed form with the item ID
        return redirect(url_for('views.lost_item_form', item_id=item_id))
        
    except Exception as e:
        print(f"Error submitting initial lost item: {e}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('views.welcome'))

@views.route('/lost-item-form')
@views.route('/lost-item-form/<item_id>')
def lost_item_form(item_id=None):
    """Display the detailed lost item form"""
    initial_data = None
    
    if item_id:
        try:
            from bson import ObjectId
            db = current_app.db
            initial_data = db.lostItems.find_one({'_id': ObjectId(item_id)})
            print(f"Retrieved initial data: {initial_data}")
        except Exception as e:
            print(f"Error fetching initial data: {e}")
    
    return render_template("lostItem.html", initial_data=initial_data)

@views.route('/submit-detailed-lost', methods=['GET', 'POST'])
def submit_detailed_lost():
    """Handle the detailed form submission with image upload and similarity check"""
    print(f"=== INCOMING REQUEST ===")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Path: {request.path}")
    print(f"User Agent: {request.headers.get('User-Agent', 'Unknown')}")
    print(f"Referrer: {request.headers.get('Referer', 'None')}")
    print(f"Content Type: {request.headers.get('Content-Type', 'None')}")
    print(f"Form keys: {list(request.form.keys()) if request.form else 'No form data'}")
    print(f"Files: {list(request.files.keys()) if request.files else 'No files'}")
    
    if request.method == 'GET':
        print("GET request received - redirecting to form")
        flash('Please submit the form properly', 'warning')
        return redirect(url_for('views.lost_item_form'))
    
    print("=== POST REQUEST PROCESSING ===")
    print(f"Form data: {dict(request.form)}")
    print(f"Files: {dict(request.files)}")
    
    try:
        db = current_app.db
        
        # Handle image upload to S3
        image_url = None
        if 'imageUpload' in request.files:
            file = request.files['imageUpload']
            print(f"File received: {file.filename if file else 'No file'}")
            
            if file and file.filename != '':
                try:
                    image_url = upload_file_to_s3(file, S3_BUCKET)
                    print(f"Image uploaded successfully: {image_url}")
                except ValueError as ve:
                    flash(f'Image upload error: {str(ve)}', 'error')
                    return redirect(request.url)
                except Exception as e:
                    print(f"Image upload failed: {e}")
                    flash('Failed to upload image. Please try again.', 'error')
                    return redirect(request.url)
        
        # Get all form data
        detailed_data = {
            'whatLost': request.form.get('whatLost'),
            'dateLost': request.form.get('dateLost'),
            'category': request.form.get('category'),
            'timeLost': request.form.get('timeLost'),
            'brand': request.form.get('brand'),
            'primaryColor': request.form.get('primaryColor'),
            'secondaryColor': request.form.get('secondaryColor'),
            'additionalInfo': request.form.get('additionalInfo'),
            'whereLost': request.form.get('whereLost'),
            'zipCode': request.form.get('zipCode'),
            'locationName': request.form.get('locationName'),
            'firstName': request.form.get('firstName'),
            'lastName': request.form.get('lastName'),
            'phoneNumber': request.form.get('phoneNumber'),
            'email': request.form.get('email'),
            'imageUrl': image_url,  # Store the S3 image URL
            'submissionTime': datetime.now(),
            'status': 'active',
            'type': 'lost'
        }
        
        print(f"Data to save: {detailed_data}")
        
        # Validate required fields
        required_fields = ['whatLost', 'dateLost', 'category', 'timeLost', 'whereLost', 'firstName', 'lastName', 'phoneNumber', 'email']
        for field in required_fields:
            if not detailed_data[field]:
                flash(f'Please fill in the {field} field', 'error')
                print(f"Validation failed for field: {field}")
                return redirect(request.url)
        
        # Check if this is an update to an existing record
        item_id = request.form.get('item_id')
        if item_id:
            from bson import ObjectId
            result = db.lostItems.update_one(
                {'_id': ObjectId(item_id)}, 
                {'$set': detailed_data}
            )
            print(f"Updated existing record. Modified count: {result.modified_count}")
            saved_item_id = item_id
        else:
            # Insert new record
            result = db.lostItems.insert_one(detailed_data)
            saved_item_id = str(result.inserted_id)
            print(f"Inserted new record with ID: {saved_item_id}")
        
        # NEW: Check for similar items in found items database
        print("=== STARTING SIMILARITY CHECK ===")
        try:
            # Get all found items for comparison
            found_items = list(db.foundItems.find({'status': 'active'}))
            print(f"Found {len(found_items)} active found items for comparison")
            
            # Create structured query data instead of just text
            query_data = {
                'whatLost': detailed_data['whatLost'],
                'category': detailed_data['category'],
                'brand': detailed_data['brand'] or '',
                'primaryColor': detailed_data['primaryColor'] or '',
                'secondaryColor': detailed_data['secondaryColor'] or '',
                'whereLost': detailed_data['whereLost'],
                'additionalInfo': detailed_data['additionalInfo'] or ''
            }
            
            # Create query text for logging
            query_text = f"{detailed_data['whatLost']} {detailed_data['category']} {detailed_data['brand']} {detailed_data['primaryColor']} {detailed_data['additionalInfo']}"
            query_text = query_text.strip()
            
            print(f"Query data: {query_data}")
            print(f"Query image URL: {image_url}")
            
            # Find similar items with structured data
            similar_items = similarity_service.find_similar_items_structured(
                query_image_url=image_url,
                query_data=query_data,
                database_items=found_items,
                top_k=10
            )
            
            if similar_items:
                print(f"Found {len(similar_items)} similar items")
                for item, score in similar_items[:3]:  # Log top 3 matches
                    print(f"Match: {item.get('whatFound', 'N/A')} - Score: {score:.3f}")
                
                # Redirect to suggestions page with the item ID and similarity results
                return redirect(url_for('views.suggestions', 
                                      lost_item_id=saved_item_id, 
                                      show_suggestions=True))
            else:
                print("No similar items found")
                flash('Lost item submitted successfully! No similar found items at this time.', 'success')
                return redirect(url_for('views.welcome'))
        
        except Exception as e:
            print(f"Error during similarity check: {e}")
            import traceback
            traceback.print_exc()
            # Even if similarity check fails, the item is still saved
            flash('Lost item submitted successfully! (Similarity check unavailable)', 'success')
            return redirect(url_for('views.welcome'))
        
    except Exception as e:
        print(f"Error submitting detailed lost item: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred. Please try again.', 'error')
        return redirect(request.url)

@views.route('/suggestions/<lost_item_id>')
def suggestions(lost_item_id):
    """Display suggestions page with similar found items"""
    try:
        db = current_app.db
        
        # Get the lost item
        lost_item = db.lostItems.find_one({'_id': ObjectId(lost_item_id)})
        if not lost_item:
            flash('Lost item not found', 'error')
            return redirect(url_for('views.welcome'))
        
        # Get all found items
        found_items = list(db.foundItems.find({'status': 'active'}))
        
        # Create query text
        query_text = f"{lost_item.get('whatLost', '')} {lost_item.get('category', '')} {lost_item.get('brand', '')} {lost_item.get('primaryColor', '')} {lost_item.get('additionalInfo', '')}"
        query_text = query_text.strip()
        
        # Find similar items
        similar_items = similarity_service.find_similar_items(
            query_image_url=lost_item.get('imageUrl'),
            query_text=query_text,
            database_items=found_items,
            top_k=10
        )
        
        # Filter items with similarity > 0.1 (threshold)
        filtered_suggestions = [(item, score) for item, score in similar_items if score > 0.1]
        
        print(f"Showing {len(filtered_suggestions)} suggestions for lost item {lost_item_id}")
        
        return render_template('suggestions.html', 
                             lost_item=lost_item,
                             suggestions=filtered_suggestions)
    
    except Exception as e:
        print(f"Error loading suggestions: {e}")
        flash('Error loading suggestions', 'error')
        return redirect(url_for('views.welcome'))

@views.route('/api/similar-items/<item_id>')
def api_similar_items(item_id):
    """API endpoint to get similar items"""
    try:
        db = current_app.db
        
        # Get the item
        item = db.lostItems.find_one({'_id': ObjectId(item_id)})
        if not item:
            return jsonify({'error': 'Item not found'}), 404
        
        # Get found items
        found_items = list(db.foundItems.find({'status': 'active'}))
        
        # Create query text
        query_text = f"{item.get('whatLost', '')} {item.get('category', '')} {item.get('brand', '')} {item.get('primaryColor', '')} {item.get('additionalInfo', '')}"
        
        # Find similar items
        similar_items = similarity_service.find_similar_items(
            query_image_url=item.get('imageUrl'),
            query_text=query_text,
            database_items=found_items,
            top_k=5
        )
        
        # Format response
        response_data = []
        for found_item, score in similar_items:
            if score > 0.1:  # Threshold
                response_data.append({
                    'item': {
                        'id': str(found_item['_id']),
                        'whatFound': found_item.get('whatFound', ''),
                        'category': found_item.get('category', ''),
                        'location': found_item.get('whereFound', ''),
                        'imageUrl': found_item.get('imageUrl'),
                        'contactEmail': found_item.get('email', ''),
                        'dateFound': found_item.get('dateFound', '')
                    },
                    'similarity_score': round(score, 3)
                })
        
        return jsonify({'similar_items': response_data})
    
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Route to render map.html
@views.route('/map')
def map_page():
    """Render the map page"""
    return render_template('map.html')  

from flask import make_response

@views.route('/lost-items')
def lost_items():
    """List all lost items from the database"""
    try:
        db = current_app.db
        lost_items_list = list(db.lostItems.find().sort('submissionTime', -1))
        
        # Create response with explicit content type
        response = make_response(render_template('viewLostItems.html', lost_items=lost_items_list))
        response.headers['Content-Type'] = 'text/html'
        return response
        
    except Exception as e:
        print(f"Error fetching lost items: {e}")
        return f"<h2>Error fetching lost items: {str(e)}</h2>"

@views.route('/found-items')
def found_items():
    """List all found items from the database"""
    try:
        db = current_app.db
        found_items_list = list(db.foundItems.find().sort('submissionTime', -1))
        
        # Create response with explicit content type
        response = make_response(render_template('viewFoundItems.html', found_items=found_items_list))
        response.headers['Content-Type'] = 'text/html'
        return response
        
    except Exception as e:
        print(f"Error fetching found items: {e}")
        return f"<h2>Error fetching found items: {str(e)}</h2>"

# Test routes for debugging
@views.route('/test-db')
def test_db():
    """Test database connection and show recent items"""
    try:
        db = current_app.db
        
        # Test basic connection
        collections = db.list_collection_names()
        
        # Count documents
        count = db.lostItems.count_documents({})
        
        # Get recent items with image info
        recent_items = list(db.lostItems.find({}).sort('submissionTime', -1).limit(5))
        
        items_html = ""
        for item in recent_items:
            image_info = "No image" if not item.get('imageUrl') else f"<a href='{item['imageUrl']}' target='_blank'>View Image</a>"
            items_html += f"<li>{item.get('itemType', 'N/A')} - {item.get('location', 'N/A')} - {item.get('status', 'N/A')} - {image_info}</li>"
        
        return f"""
        <h2>Database Test Results</h2>
        <p><strong>Database:</strong> {db.name}</p>
        <p><strong>Collections:</strong> {collections}</p>
        <p><strong>Total items:</strong> {count}</p>
        <h3>Recent Items:</h3>
        <ul>{items_html}</ul>
        <p><a href="/">Back to Home</a></p>
        """
    except Exception as e:
        return f"<h2>Database Error:</h2><p>{str(e)}</p><p><a href='/'>Back to Home</a></p>"

@views.route('/test-s3')
def test_s3():
    """Test S3 connection"""
    try:
        # Test S3 connection by listing objects
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=10)
        
        objects_info = ""
        if 'Contents' in response:
            for obj in response['Contents'][:5]:  # Show first 5 objects
                objects_info += f"<li>{obj['Key']} (Size: {obj['Size']} bytes)</li>"
        else:
            objects_info = "<li>No objects in bucket</li>"
        
        return f"""
        <h2>S3 Test Results</h2>
        <p><strong>Bucket:</strong> {S3_BUCKET}</p>
        <p><strong>Region:</strong> {AWS_REGION}</p>
        <p><strong>Connection:</strong> Success ✅</p>
        <p><strong>Objects in bucket:</strong> {response.get('KeyCount', 0)}</p>
        <h3>Recent Objects:</h3>
        <ul>{objects_info}</ul>
        <p><a href="/">Back to Home</a></p>
        """
    except Exception as e:
        return f"""
        <h2>S3 Connection Error</h2>
        <p><strong>Error:</strong> {str(e)}</p>
        <p><strong>Bucket:</strong> {S3_BUCKET}</p>
        <p><strong>Region:</strong> {AWS_REGION}</p>
        <p>Check your .env file and AWS credentials</p>
        <p><a href="/">Back to Home</a></p>
        """

@views.route('/test-s3-upload', methods=['GET', 'POST'])
def test_s3_upload():
    """Test S3 upload functionality"""
    if request.method == 'POST':
        if 'test_file' not in request.files:
            return "No file uploaded"
        
        file = request.files['test_file']
        if file.filename == '':
            return "No file selected"
        
        try:
            file_url = upload_file_to_s3(file, S3_BUCKET)
            return f"""
            <h2>S3 Upload Test - SUCCESS</h2>
            <p>File uploaded successfully!</p>
            <p><strong>URL:</strong> <a href="{file_url}" target="_blank">{file_url}</a></p>
            <p><a href="/test-s3-upload">Test Again</a></p>
            """
        except Exception as e:
            return f"""
            <h2>S3 Upload Test - FAILED</h2>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Check the console output for detailed error information.</p>
            <p><a href="/test-s3-upload">Try Again</a></p>
            """
    
    return '''
    <h2>Test S3 Upload</h2>
    <form method="POST" enctype="multipart/form-data">
        <p>
            <label>Select an image file:</label><br>
            <input type="file" name="test_file" accept="image/*" required>
        </p>
        <button type="submit">Test Upload</button>
    </form>
    <p><a href="/">Back to Home</a></p>
    '''

@views.route('/test-template')
def test_template():
    """Test template rendering"""
    return render_template('simple.html', items=[1, 2, 3], test_var="Hello World")

@views.route('/debug-routes')
def debug_routes():
    """Show all registered routes"""
    from flask import current_app
    
    routes = []
    for rule in current_app.url_map.iter_rules():
        routes.append(f"{rule.rule} -> {rule.endpoint} ({', '.join(rule.methods)})")
    
    routes_html = "<br>".join(routes)
    return f"""
    <h2>Registered Routes</h2>
    <p>Total routes: {len(routes)}</p>
    <hr>
    {routes_html}
    """

@views.route('/test-similarity')
def test_similarity():
    """Test the similarity service"""
    try:
        # Test if the similarity service is working
        test_text = "black smartphone with cracked screen"
        embedding = similarity_service.get_text_embedding(test_text)
        
        if embedding is not None:
            return f"""
            <h2>Similarity Service Test - SUCCESS</h2>
            <p>Text: "{test_text}"</p>
            <p>Embedding shape: {embedding.shape}</p>
            <p>Service is working correctly!</p>
            <p><a href="/">Back to Home</a></p>
            """
        else:
            return f"""
            <h2>Similarity Service Test - FAILED</h2>
            <p>Failed to generate embedding</p>
            <p>Check if CLIP and SentenceBERT models are installed</p>
            <p><a href="/">Back to Home</a></p>
            """
    except Exception as e:
        return f"""
        <h2>Similarity Service Test - ERROR</h2>
        <p>Error: {str(e)}</p>
        <p>Make sure to install required packages:</p>
        <ul>
            <li>pip install torch torchvision</li>
            <li>pip install clip-by-openai</li>
            <li>pip install sentence-transformers</li>
        </ul>
        <p><a href="/">Back to Home</a></p>
        """

@views.route('/submit-found')
def submit_found():
    """Redirect old submit-found URL to new found-item-form"""
    return redirect(url_for('views.found_item_form'))

@views.route('/found-item-form')
@views.route('/found-item-form/<item_id>')
def found_item_form(item_id=None):
    """Display the detailed found item form"""
    initial_data = None
    
    if item_id:
        try:
            from bson import ObjectId
            db = current_app.db
            initial_data = db.foundItems.find_one({'_id': ObjectId(item_id)})
            print(f"Retrieved initial found item data: {initial_data}")
        except Exception as e:
            print(f"Error fetching initial found item data: {e}")
    
    return render_template("foundItem.html", initial_data=initial_data)

@views.route('/submit-detailed-found', methods=['GET', 'POST'])
def submit_detailed_found():
    """Handle the detailed found item form submission with image upload"""
    print(f"=== INCOMING FOUND ITEM REQUEST ===")
    print(f"Method: {request.method}")
    print(f"Form keys: {list(request.form.keys()) if request.form else 'No form data'}")
    print(f"Files: {list(request.files.keys()) if request.files else 'No files'}")
    
    if request.method == 'GET':
        print("GET request received - redirecting to form")
        flash('Please submit the form properly', 'warning')
        return redirect(url_for('views.found_item_form'))
    
    print("=== POST REQUEST PROCESSING ===")
    print(f"Form data: {dict(request.form)}")
    
    try:
        db = current_app.db
        
        # Handle image upload to S3
        image_url = None
        if 'imageUpload' in request.files:
            file = request.files['imageUpload']
            print(f"File received: {file.filename if file else 'No file'}")
            
            if file and file.filename != '':
                try:
                    image_url = upload_file_to_s3(file, S3_BUCKET)
                    print(f"Found item image uploaded successfully: {image_url}")
                except ValueError as ve:
                    flash(f'Image upload error: {str(ve)}', 'error')
                    return redirect(request.url)
                except Exception as e:
                    print(f"Image upload failed: {e}")
                    flash('Failed to upload image. Please try again.', 'error')
                    return redirect(request.url)
        
        # Get all form data for found item
        detailed_data = {
            'whatFound': request.form.get('whatFound'),
            'dateFound': request.form.get('dateFound'),
            'category': request.form.get('category'),
            'timeFound': request.form.get('timeFound'),
            'brand': request.form.get('brand'),
            'primaryColor': request.form.get('primaryColor'),
            'secondaryColor': request.form.get('secondaryColor'),
            'additionalInfo': request.form.get('additionalInfo'),
            'whereFound': request.form.get('whereFound'),
            'zipCode': request.form.get('zipCode'),
            'locationName': request.form.get('locationName'),
            'firstName': request.form.get('firstName'),
            'lastName': request.form.get('lastName'),
            'phoneNumber': request.form.get('phoneNumber'),
            'email': request.form.get('email'),
            'imageUrl': image_url,  # Store the S3 image URL
            'submissionTime': datetime.now(),
            'status': 'active',
            'type': 'found'
        }
        
        print(f"Found item data to save: {detailed_data}")
        
        # Validate required fields
        required_fields = ['whatFound', 'dateFound', 'category', 'timeFound', 'whereFound', 'firstName', 'lastName', 'phoneNumber', 'email']
        for field in required_fields:
            if not detailed_data[field]:
                flash(f'Please fill in the {field} field', 'error')
                print(f"Validation failed for field: {field}")
                return redirect(request.url)
        
        # Check if this is an update to an existing record
        item_id = request.form.get('item_id')
        if item_id:
            from bson import ObjectId
            result = db.foundItems.update_one(
                {'_id': ObjectId(item_id)}, 
                {'$set': detailed_data}
            )
            print(f"Updated existing found item record. Modified count: {result.modified_count}")
        else:
            # Insert new record
            result = db.foundItems.insert_one(detailed_data)
            print(f"Inserted new found item record with ID: {result.inserted_id}")
        
        flash('Found item submitted successfully! Thank you for helping reunite lost items with their owners.', 'success')
        return redirect(url_for('views.welcome'))
        
    except Exception as e:
        print(f"Error submitting detailed found item: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred. Please try again.', 'error')
        return redirect(request.url)
    """Debug similarity calculations for a specific lost item"""
    try:
        db = current_app.db
        
        # Get the lost item
        lost_item = db.lostItems.find_one({'_id': ObjectId(lost_item_id)})
        if not lost_item:
            return "<h2>Lost item not found</h2>"
        
        # Get all found items
        found_items = list(db.foundItems.find({'status': 'active'}))
        
        # Create structured query data
        query_data = {
            'whatLost': lost_item.get('whatLost', ''),
            'category': lost_item.get('category', ''),
            'brand': lost_item.get('brand', ''),
            'primaryColor': lost_item.get('primaryColor', ''),
            'additionalInfo': lost_item.get('additionalInfo', '')
        }
        
        # Test similarity calculation
        similar_items = similarity_service.find_similar_items_structured(
            query_image_url=lost_item.get('imageUrl'),
            query_data=query_data,
            database_items=found_items,
            top_k=20
        )
        
        # Create debug output
        debug_html = f"""
        <h2>Similarity Debug for Lost Item: {lost_item.get('whatLost', 'Unknown')}</h2>
        <h3>Query Data:</h3>
        <ul>
            <li>Description: {query_data['whatLost']}</li>
            <li>Category: {query_data['category']}</li>
            <li>Brand: {query_data['brand']}</li>
            <li>Color: {query_data['primaryColor']}</li>
            <li>Image: {'Yes' if lost_item.get('imageUrl') else 'No'}</li>
        </ul>
        <h3>Found Items Comparison ({len(found_items)} total):</h3>
        """
        
        if similar_items:
            debug_html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
            debug_html += "<tr><th>Rank</th><th>Score</th><th>Description</th><th>Category</th><th>Brand</th><th>Color</th><th>Image</th></tr>"
            
            for i, (item, score) in enumerate(similar_items):
                debug_html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{score:.3f}</td>
                    <td>{item.get('whatFound', item.get('whatLost', 'N/A'))}</td>
                    <td>{item.get('category', 'N/A')}</td>
                    <td>{item.get('brand', 'N/A')}</td>
                    <td>{item.get('primaryColor', 'N/A')}</td>
                    <td>{'Yes' if item.get('imageUrl') else 'No'}</td>
                </tr>
                """
            debug_html += "</table>"
        else:
            debug_html += "<p>No similar items found</p>"
        
        debug_html += f"<p><a href='/suggestions/{lost_item_id}'>View Suggestions Page</a></p>"
        debug_html += "<p><a href='/'>Back to Home</a></p>"
        
        return debug_html
        
    except Exception as e:
        return f"<h2>Debug Error:</h2><p>{str(e)}</p>"


    

