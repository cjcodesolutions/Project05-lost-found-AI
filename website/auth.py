from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import bcrypt

auth = Blueprint('auth', __name__)

# -------------------------
# SIGNUP
# -------------------------
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        db = current_app.db
        
        # Get form data
        first_name = request.form.get("firstName")
        last_name = request.form.get("lastName")
        email = request.form.get("email")
        phone = request.form.get("phone")
        user_type = request.form.get("userType")
        password = request.form.get("password")
        confirm_password = request.form.get("confirmPassword")
        terms = request.form.get("terms")
        notifications = True if request.form.get("notifications") else False

        # Validation
        if not first_name or not last_name or not email or not password:
            flash("Please fill all required fields", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template("signup.html")

        if not user_type:
            flash("Please select a user type", "error")
            return render_template("signup.html")

        if not terms:
            flash("You must agree to the terms and conditions", "error")
            return render_template("signup.html")

        if db.users.find_one({"email": email}):
            flash("User already exists with this email", "error")
            return render_template("signup.html")

        # Hash password
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        # Insert user
        try:
            result = db.users.insert_one({
                "firstName": first_name,
                "lastName": last_name,
                "email": email,
                "phone": phone,
                "userType": user_type,
                "password": hashed_pw.decode("utf-8"),
                "termsAccepted": True,
                "notifications": notifications
            })
            print(f"User created successfully: {result.inserted_id}")
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            print(f"Database insert error: {e}")
            flash("Error creating account. Please try again.", "error")
            return render_template("signup.html")

    return render_template("signup.html")


# -------------------------
# LOGIN
# -------------------------
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        db = current_app.db

        email = request.form.get("email")
        password = request.form.get("password")

        # Check user exists
        user = db.users.find_one({"email": email})
        if not user:
            flash("No account found with this email", "error")
            return render_template("login.html")  # Fixed: stay on login page

        # Verify password
        if bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            flash("Login successful!", "success")
            return redirect(url_for("views.welcome"))  # Fixed: redirect to welcome
        else:
            flash("Invalid password", "error")
            return render_template("login.html")

    return render_template("login.html")


# -------------------------
# LOGOUT
# -------------------------
@auth.route('/logout')
def logout():
    flash("You have been logged out", "info")
    return redirect(url_for("views.home"))


# -------------------------
# TEST DATABASE
# -------------------------
@auth.route('/test-db')
def test_db():
    try:
        db = current_app.db
        collections = db.list_collection_names()
        user_count = db.users.count_documents({})
        
        return f"""
        <h2>Database Test Results:</h2>
        <p>Database: {db.name}</p>
        <p>Collections: {collections}</p>
        <p>User count: {user_count}</p>
        <p>Connection: WORKING ✅</p>
        """
    except Exception as e:
        return f"Database test FAILED: {e}"


