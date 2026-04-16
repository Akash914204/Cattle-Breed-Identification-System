import os
import uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, url_for
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from services.prediction_service import PredictionService
from services.gradcam_service import generate_gradcam

predict_bp = Blueprint("predict", __name__)

# Singleton
_prediction_service = None


def _get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        print("🔥 Loading Prediction Service...")
        _prediction_service = PredictionService(
            model_path=current_app.config["MODEL_PATH"],
            num_classes=current_app.config["NUM_CLASSES"],
        )
    return _prediction_service


def _allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]
    )


@predict_bp.route("/predict", methods=["POST"])
@login_required
def predict():

    try:
        # ---------------- FILE CHECK ----------------
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not _allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        # ---------------- SAVE IMAGE ----------------
        original_name = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_name}"

        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        print("✅ Image saved at:", filepath)

        # ---------------- PREDICTION ----------------
        service = _get_prediction_service()

        print("🔥 Running prediction...")
        result = service.predict(filepath)

        print("✅ Prediction result:", result)

        # ---------------- GRADCAM ----------------
        heatmap_dir = os.path.join("static", "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        heatmap_filename = generate_gradcam(
            model=service.model,
            image_path=filepath,
            transform=service.transform,
            device=service.device,
            save_dir=heatmap_dir,
            class_idx=None,
        )

        heatmap_url = url_for("static", filename=f"heatmaps/{heatmap_filename}")

        print("✅ Heatmap generated:", heatmap_filename)

        # ---------------- SAVE TO DB ----------------
        record = {
            "user_id": current_user.id,
            "filename": unique_filename,
            "original_filename": original_name,
            "predicted_breed": result["breed"],
            "confidence_score": result["confidence"],
            "image_path": filepath.replace("\\", "/"),
            "heatmap_filename": heatmap_filename,
            "timestamp": datetime.now(timezone.utc),
        }

        current_app.db.predictions.insert_one(record)

        print("✅ Saved to DB")

        return jsonify({**result, "heatmap_url": heatmap_url}), 200

    except Exception as e:
        print("❌ FULL ERROR:", e)   # 🔥 THIS IS KEY
        return jsonify({"error": str(e)}), 500