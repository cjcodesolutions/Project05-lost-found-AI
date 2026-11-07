# Lost & Found AI Application

A modern Flask-based web application designed to help users report lost items and cross-reference them with found items using advanced AI image similarity search.

## Features
* User Authentication (Signup, Login, Logout)
* Report Lost Items
* Report Found Items
* Advanced AI Image Similarity Search using CLIP/Sentence-BERT
* MongoDB Integration for Data Storage
* AWS S3 Support for Image Storage

---

## 🚀 Setup Instructions

Follow these steps carefully to set up the project on your local machine after cloning the repository.

### Prerequisites
- Python 3.9+ installed on your system.
- MongoDB installed locally, or a MongoDB Atlas URI.
- An AWS Account (for S3 bucket keys - optional but recommended for image uploads).
- Git installed.

### Step 1: Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/cjcodesolutions/Project05-lost-found-AI.git
cd Project05-lost-found-AI
```

### Step 2: Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies locally.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate
# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
If venv not activated,
```bash
#windows
Set-ExecutionPolicy Unrestricted -Scope Process
```
### Step 3: Install Dependencies
With the virtual environment activated, install the required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
You need to set up the environment variables so the app can connect to the database and cloud storage.

1. Open the `.env` file located in the root directory.
2. Update the placeholder values with your actual credentials:
```env
MONGO_DB="DATABASE_NAME"
MONGO_URL="MONGO_URL"

AWS_ACCESS_KEY_ID="AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="AWS_SECRET_ACCESS_KEY"
S3_BUCKET_NAME="S3_BUCKET_NAME"
AWS_REGION="AWS_REGION"
```

### Step 5: (Optional) Download AI Models
> **Note:** Large models like `finetuned_clip.pt` and `model.safetensors` are ignored by git to keep the repository lightweight.

If you plan to use the similarity AI search, you will need to download or recreate the PyTorch/HuggingFace model files and place them in the following directories:
- `website/models/finetuned_clip.pt`
- `website/models/finetuned_sentence_bert/model.safetensors`

For this replace the `website/models/` and `website/datasets/` with your own files which you downloaded from google drive link which I sent for the MVP.

### Step 6: Run the Application 
Finally, start the Flask development server:

```bash
python main.py
```

You should see output indicating the routes and the server running. Open your browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 Project Structure
* `main.py` - Application entry point.
* `website/` - Contains templates, static assets, and core logic.
* `website/auth.py` - User authentication routes.
* `website/views.py` - Core routes for the web pages.
* `website/similarity_service.py` - AI integration for image matching.
* `requirements.txt` - Project dependencies.

---

### Troubleshooting
- **Missing Module Error:** Ensure you activated the virtual environment (`venv`) before running `pip install` and `python main.py`.
- **Database Connection Error:** Double check your `MONGO_URL` in the `.env` file string.
- **Git Timeout Issues:** Large files are excluded via `.gitignore`. If you add new models, ensure they are also ignored.

## Persistent vector search (pgvector + HNSW)

Similarity search now encodes each completed report once on save. PostgreSQL
stores only its MongoDB item ID, item type, and a vector—no source descriptions,
image URLs, or contact information. MongoDB still stores the report fields used
by the forms and results pages. This change does not erase existing report text.

Use PostgreSQL with the pgvector extension installed (0.5 or newer). For local
 development, for example:

```bash
docker run --name foundit-vectors \
  -e POSTGRES_DB=foundit -e POSTGRES_USER=foundit \
  -e POSTGRES_PASSWORD=local-development-only \
  -p 127.0.0.1:5432:5432 -v foundit-vectors:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```

Install `requirements.txt`, keep the existing MongoDB/S3 settings, and add this
to `.env` (use your actual PostgreSQL credentials in deployments):

```dotenv
VECTOR_DATABASE_URL=postgresql://foundit:local-development-only@localhost:5432/foundit
```

Initialize the schema and index, then encode existing active reports:

```bash
flask --app main init-vector-db
flask --app main backfill-vectors
```

The schema initialization requires permission to enable `vector` and create tables
and indexes. Runtime access only needs read/write permissions on `item_embeddings`.
Backfill skips existing vectors and reports failures; rerun after fixing failures.
Legacy S3 presigned URLs may have expired and must be refreshed before backfilling.
Reports written directly to MongoDB outside this app also require backfilling.

Vectors concatenate normalized MiniLM text (384 dimensions) and CLIP image
(512 dimensions) embeddings with equal weight. Missing modalities use zero blocks.
HNSW searches cosine distance with `ef_search=200`, retrieves up to 100 candidates
for the current routes, then fetches only active matching reports from MongoDB.
Searches reuse the saved lost-report vector; they do not encode targets or download
images. Similarity scores now represent combined-vector cosine similarity, replacing
the previous cross-modal scoring and category/color/brand bonuses. The default
threshold remains 0.5 and should be calibrated against your own labeled matches.
HNSW is approximate; deleted or inactive candidates can reduce result count.

Model changes require rebuilding **all** vectors with
`flask --app main backfill-vectors --force`; avoid serving searches during a rebuild
because embeddings from different model versions are not comparable. Dimensions
are fixed to the current models. PostgreSQL and MongoDB writes are not one atomic
transaction: saved reports with failed indexing need `backfill-vectors` to recover.
Updated vectors are invalidated before report edits to prevent stale matches.

Run dependency-free unit checks with:

```bash
python3 -m unittest discover -s tests -v
```

For live verification, inspect `EXPLAIN (ANALYZE, BUFFERS)` for a representative
`SELECT item_id FROM item_embeddings WHERE item_type = 'found'
ORDER BY embedding <=> '[your 896-dimensional vector]'::vector LIMIT 100;`.
PostgreSQL may choose a sequential scan for very small tables.
