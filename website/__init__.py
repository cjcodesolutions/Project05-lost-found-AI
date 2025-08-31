import os
from flask import Flask
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()  # load values from .env

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret-key-goes-here'

    # Build connection string
    connection_string = (
        "mongodb+srv://cjcodesolutions:Abc12345@cluster0.fbte9k0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    )

    client = MongoClient(connection_string)
    db = client["users"]  
    app.db = db

    # Register blueprints
    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app
