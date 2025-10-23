const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const resultText = document.getElementById('resultText');
const confidenceChart = document.getElementById('confidenceChart');
const topPredictions = document.getElementById('topPredictions');
const historyList = document.getElementById('historyList');
const darkModeToggle = document.getElementById('darkModeToggle');

let predictionHistory = [];
let chartInstance = null; // 🟢 Track a single reusable chart instance

// -------------------- Drag & Drop Handling --------------------
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// -------------------- Sample Image Handling --------------------
document.querySelectorAll('.sample-image').forEach(img => {
    img.addEventListener('click', () => {
        fetch(img.src)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], `sample-${img.alt}.jpg`, { type: blob.type });
                handleFile(file);
            });
    });
});

// -------------------- File Handling --------------------
function handleFile(file) {
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert('Only JPEG and PNG files are supported.');
        return;
    }

    previewContainer.classList.remove('hidden');
    uploadButton.classList.remove('hidden');

    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        uploadButton.onclick = () => classifyImage(file);
    };
    reader.readAsDataURL(file);
}

// -------------------- API Integration --------------------
async function classifyImage(file) {
    loadingSpinner.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            displayResults(data);
            addToHistory(data);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loadingSpinner.classList.add('hidden');
    }
}

// -------------------- Display Results (Reusable Chart) --------------------
function displayResults(data) {
    resultsSection.classList.remove('hidden');
    resultText.textContent = `Predicted Class: ${data.class} (${(data.confidence * 100).toFixed(2)}%)`;

    // Top 3 Predictions
    const top3 = Object.entries(data.predictions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);

    topPredictions.innerHTML = top3.map(([cls, conf]) => `
        <div class="mt-2">
            <div class="flex justify-between">
                <span>${cls}</span>
                <span>${(conf * 100).toFixed(2)}%</span>
            </div>
            <div class="bg-gray-200 h-4 rounded">
                <div class="bg-green-500 h-4 rounded prediction-bar" style="width: ${conf * 100}%"></div>
            </div>
        </div>
    `).join('');

    // 🧠 Chart update logic
    const labels = Object.keys(data.predictions);
    const values = Object.values(data.predictions).map(v => v * 100);

    if (chartInstance) {
        // Update existing chart
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = values;
        chartInstance.update();
    } else {
        // Create new chart
        chartInstance = new Chart(confidenceChart, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: values,
                    backgroundColor: '#10B981'
                }]
            },
            options: {
                responsive: true,
                animation: {
                    duration: 600,
                    easing: 'easeOutQuart'
                },
                scales: {
                    y: { beginAtZero: true, max: 100 }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}

// -------------------- Prediction History --------------------
function addToHistory(data) {
    const item = {
        filename: data.filename || 'Uploaded Image',
        class: data.class,
        confidence: data.confidence,
        timestamp: Date.now()
    };
    predictionHistory.push(item);
    if (predictionHistory.length > 5) predictionHistory.shift();

    historyList.innerHTML = predictionHistory.map(item => `
        <div class="history-item p-2 rounded border">
            <p>File: ${item.filename}</p>
            <p>Class: ${item.class} (${(item.confidence * 100).toFixed(2)}%)</p>
            <p>Time: ${new Date(item.timestamp).toLocaleString()}</p>
        </div>
    `).join('');
}

// -------------------- Dark Mode --------------------
darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
});

if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}
