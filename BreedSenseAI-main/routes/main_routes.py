from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@main_bp.route("/upload")
@login_required
def upload():
    """Upload & predict page (login required)."""
    return render_template("upload.html")


@main_bp.route("/history")
@login_required
def history():
    """Display predictions for the currently logged-in user."""
    predictions = list(
        current_app.db.predictions.find(
            {"user_id": current_user.id},
            {"_id": 0}
        ).sort("timestamp", -1)
    )

    # Format timestamps for display
    for pred in predictions:
        ts = pred.get("timestamp")
        if ts:
            pred["timestamp_display"] = ts.strftime("%d %b %Y, %I:%M %p")
        else:
            pred["timestamp_display"] = "—"

    return render_template("history.html", predictions=predictions)


@main_bp.route("/about")
def about():
    """About page."""
    return render_template("about.html")
