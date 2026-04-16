"""
routes/auth_routes.py — Registration, Login, Logout
=====================================================
"""

import re
from datetime import datetime, timezone
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from bcrypt import hashpw, gensalt, checkpw

from models.user import User

auth_bp = Blueprint("auth", __name__)

# ---------- helpers ----------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


# ---------- Register ----------

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # --- Validation ---
        errors = []
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters.")
        if not _is_valid_email(email):
            errors.append("Please enter a valid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        # Check uniqueness
        if not errors:
            if User.find_by_email(current_app.db, email):
                errors.append("An account with this email already exists.")
            if User.find_by_username(current_app.db, username):
                errors.append("This username is already taken.")

        if errors:
            for err in errors:
                flash(err, "error")
            return render_template("auth/register.html",
                                   username=username, email=email)

        # --- Create user ---
        password_hash = hashpw(password.encode("utf-8"), gensalt()).decode("utf-8")
        current_app.db.users.insert_one({
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html")


# ---------- Login ----------

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """User login page."""
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.find_by_email(current_app.db, email)

        if user and checkpw(password.encode("utf-8"),
                            user.password_hash.encode("utf-8")):
            login_user(user, remember=remember)
            # Redirect to originally requested page, or home
            next_page = request.args.get("next")
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(next_page or url_for("main.index"))

        flash("Invalid email or password.", "error")
        return render_template("auth/login.html", email=email)

    return render_template("auth/login.html")


# ---------- Logout ----------

@auth_bp.route("/logout")
@login_required
def logout():
    """Log the user out and redirect to home."""
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("main.index"))
