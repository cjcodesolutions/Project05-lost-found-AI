from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv

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
        
        # Generate presigned URL (valid for 1 year)
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


# Also add this route to test S3 upload specifically
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

@views.route('/')
def home():
    return render_template("home.html")

@views.route('/welcome')
def welcome():
    return render_template("welcome.html")

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
    """Handle the detailed form submission with image upload"""
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
        else:
            # Insert new record
            result = db.lostItems.insert_one(detailed_data)
            print(f"Inserted new record with ID: {result.inserted_id}")
        
        flash('Lost item submitted successfully!', 'success')
        return redirect(url_for('views.welcome'))
        
    except Exception as e:
        print(f"Error submitting detailed lost item: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred. Please try again.', 'error')
        return redirect(request.url)
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
    """List all lost items from the database"""
    try:
        db = current_app.db
        found_items_list = list(db.foundItems.find().sort('submissionTime', -1))
        
        # Create response with explicit content type
        response = make_response(render_template('viewFoundItems.html', found_items=found_items_list))
        response.headers['Content-Type'] = 'text/html'
        return response
        
    except Exception as e:
        print(f"Error fetching lost items: {e}")
        return f"<h2>Error fetching lost items: {str(e)}</h2>"
    

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