"""
models/user.py — User model for Flask-Login
============================================
Wraps a MongoDB document to satisfy the Flask-Login UserMixin interface.
"""

from flask_login import UserMixin
from bson import ObjectId


class User(UserMixin):
    """Represents an authenticated user backed by a MongoDB document."""

    def __init__(self, user_data: dict):
        self.id = str(user_data["_id"])
        self.username = user_data["username"]
        self.email = user_data["email"]
        self.password_hash = user_data["password_hash"]
        self.created_at = user_data.get("created_at")

    @staticmethod
    def find_by_id(db, user_id: str):
        """Look up a user by their MongoDB ObjectId string."""
        try:
            data = db.users.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None
        return User(data) if data else None

    @staticmethod
    def find_by_email(db, email: str):
        """Look up a user by email address."""
        data = db.users.find_one({"email": email.lower().strip()})
        return User(data) if data else None

    @staticmethod
    def find_by_username(db, username: str):
        """Look up a user by username."""
        data = db.users.find_one({"username": username.strip()})
        return User(data) if data else None
