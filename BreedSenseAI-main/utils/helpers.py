import os
from werkzeug.utils import secure_filename


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if a filename has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def save_upload(file, upload_folder: str) -> str:
    """Save an uploaded file and return the saved path."""
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filepath
