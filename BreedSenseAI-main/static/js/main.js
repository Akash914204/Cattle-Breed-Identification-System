// Cattle Breed Recognition — Client-side prediction handler

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    const resultBox = document.getElementById("result");
    const breedSpan = document.getElementById("breed-name");
    const confidenceSpan = document.getElementById("confidence");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        resultBox.style.display = "none";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                breedSpan.textContent = data.breed;
                confidenceSpan.textContent = data.confidence;
                resultBox.style.display = "block";
            } else {
                alert(data.error || "Something went wrong.");
            }
        } catch (err) {
            console.error(err);
            alert("Failed to connect to the server.");
        }
    });
});
