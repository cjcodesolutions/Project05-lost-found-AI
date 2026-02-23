import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URI
mongo_uri = os.getenv('MONGO_URI')
if not mongo_uri:
    print("Error: MONGO_URI not found in .env file")
    exit(1)

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient(mongo_uri)
db_name = os.getenv('MONGO_DB', 'lost_and_found_db')
db = client[db_name]

# The email corresponding to the user who uploaded the existing items
target_email = "kumar@test.com"

# Update Lost Items
print("\nUpdating Lost Items...")
lost_result = db.lostItems.update_many(
    # Only update documents that do NOT currently have an author_email field
    {"author_email": {"$exists": False}}, 
    {"$set": {"author_email": target_email}}
)
print(f"Matched {lost_result.matched_count} lost items.")
print(f"Updated {lost_result.modified_count} lost items with author_email: {target_email}.")

# Update Found Items
print("\nUpdating Found Items...")
found_result = db.foundItems.update_many(
    {"author_email": {"$exists": False}}, 
    {"$set": {"author_email": target_email}}
)
print(f"Matched {found_result.matched_count} found items.")
print(f"Updated {found_result.modified_count} found items with author_email: {target_email}.")

print("\nDatabase backfill complete! The existing items are now owned by kumar@test.com.")
