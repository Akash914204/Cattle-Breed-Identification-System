import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "cattle_breed_db")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/weights/best_model.pth")
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", 50))
    UPLOAD_FOLDER = os.path.join("static", "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


# Map config names to classes
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
