import os
from flask import Flask
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()  # load values from .env

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    db_name = os.getenv("MONGO_DB")
    connection_string = os.getenv("MONGO_URI")    
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[db_name]
        app.db = db
        print("MongoDB Connected Successfully!")
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {e}")
        # Continue without crashing
        app.db = None

    # Register blueprints
    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app