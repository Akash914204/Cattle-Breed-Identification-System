"""
Cattle Breed Recognition — Flask Application
=============================================
Entry point for the Flask web application.
"""

import os
from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from pymongo import MongoClient
from dotenv import load_dotenv

from config import Config, config_by_name
from models.user import User

# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app(config_name: str = None) -> Flask:
    """Create and configure the Flask application."""

    load_dotenv()

    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, Config))

    # --- CORS ---
    CORS(app)

    # --- MongoDB connection ---
    mongo_client = MongoClient(app.config["MONGO_URI"])
    app.db = mongo_client[app.config["MONGO_DB_NAME"]]

    # Verify connection on startup
    try:
        mongo_client.admin.command("ping")
        app.logger.info("✅  Connected to MongoDB successfully.")
    except Exception as e:
        app.logger.warning(f"⚠️  MongoDB connection failed: {e}")

    # --- Flask-Login ---
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "error"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.find_by_id(app.db, user_id)

    # --- Ensure upload directory exists ---
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # --- Register Blueprints ---
    from routes.main_routes import main_bp
    from routes.predict_routes import predict_bp
    from routes.auth_routes import auth_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/auth")

    return app


# ---------------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
