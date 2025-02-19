function processImage() {
    const input = document.getElementById('imageInput').files[0];
    const loading = document.getElementById('loading');
    const originalImg = document.getElementById('originalImage');
    const denoisedImg = document.getElementById('denoisedImage');
    const tumorImg = document.getElementById('tumorImage');
    const tumorResult = document.getElementById('tumorResult');

    if (!input) {
        alert('Please upload an MRI image first.');
        return;
    }

    // Show loading indicator
    loading.classList.remove('hidden');

    // Simulate backend processing (replace this with actual Flask API call)
    setTimeout(() => {
        // Display images (just placeholder logic)
        originalImg.src = URL.createObjectURL(input);
        originalImg.classList.remove('hidden');

        denoisedImg.src = URL.createObjectURL(input); // Replace with actual denoised image
        denoisedImg.classList.remove('hidden');

        tumorImg.src = URL.createObjectURL(input); // Replace with tumor detection image
        tumorImg.classList.remove('hidden');

        tumorResult.textContent = "Tumor Detected: Yes (Upper Right Region)"; // Placeholder text
        tumorResult.classList.remove('hidden');

        loading.classList.add('hidden');
    }, 3000);
}
