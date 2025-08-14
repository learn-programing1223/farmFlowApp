#!/usr/bin/env python3
"""
PlantPulse Web Server with Feedback Collection
Saves misclassified images for model improvement
"""

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# HTML Template with feedback mechanism
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>PlantPulse - Disease Detection with Feedback</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 30px;
        }
        h1 { color: #4CAF50; margin-bottom: 10px; }
        .accuracy-badge {
            display: inline-block;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        .upload-box {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #cbd5e1;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            background: #f8fafc;
            transition: all 0.3s;
        }
        .upload-area:hover { 
            border-color: #667eea;
            background: #f0f4ff;
        }
        .upload-area.dragover {
            border-color: #4CAF50;
            background: #f0fdf4;
        }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin: 10px 5px;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-success {
            background: linear-gradient(135deg, #10b981, #059669);
        }
        .btn-error {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        .result-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        #preview {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 10px;
            background: #f8fafc;
        }
        .result-item {
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .result-item.top {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            font-weight: bold;
            font-size: 18px;
        }
        .confidence-bar {
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        .disease-info {
            margin-top: 20px;
            padding: 15px;
            background: #f0f9ff;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }
        .feedback-section {
            background: #fef3c7;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border: 2px solid #fbbf24;
        }
        .feedback-section h3 {
            color: #92400e;
            margin-bottom: 15px;
        }
        .feedback-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .actual-class-select {
            padding: 10px;
            border: 2px solid #cbd5e1;
            border-radius: 8px;
            font-size: 16px;
            margin-top: 10px;
            width: 100%;
        }
        .loading {
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 4px solid #f3f4f6;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            animation: slideIn 0.3s ease;
            z-index: 1000;
        }
        .notification.success {
            background: linear-gradient(135deg, #10b981, #059669);
        }
        .notification.error {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
        @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        @media (max-width: 768px) {
            .results-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌿 PlantPulse Disease Detection</h1>
            <p>Help improve the model by providing feedback on predictions!</p>
            <div class="accuracy-badge">96.1% Validation Accuracy</div>
        </div>
        
        <div class="upload-box">
            <div class="upload-area" id="uploadArea">
                <h3>📸 Upload Plant Image</h3>
                <p>Click to browse or drag & drop</p>
                <p style="margin-top: 10px; color: #6b7280; font-size: 14px;">
                    Your feedback helps improve accuracy!
                </p>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            <button class="btn" id="analyzeBtn" disabled>Analyze Plant Health</button>
        </div>
        
        <div class="results-grid" id="resultsSection" style="display: none;">
            <div class="result-box">
                <h3>📷 Uploaded Image</h3>
                <img id="preview" alt="Plant image">
            </div>
            <div class="result-box">
                <h3>🔬 Analysis Results</h3>
                <div id="results"></div>
                
                <div class="feedback-section" id="feedbackSection" style="display: none;">
                    <h3>📝 Was this prediction correct?</h3>
                    <p>Your feedback helps improve the model!</p>
                    
                    <div class="feedback-buttons">
                        <button class="btn btn-success" onclick="submitFeedback(true)">
                            ✅ Correct
                        </button>
                        <button class="btn btn-error" onclick="submitFeedback(false)">
                            ❌ Incorrect
                        </button>
                    </div>
                    
                    <div id="actualClassSection" style="display: none; margin-top: 15px;">
                        <label>What's the actual condition?</label>
                        <select class="actual-class-select" id="actualClass">
                            <option value="">Select actual condition...</option>
                            <option value="Blight">Blight</option>
                            <option value="Healthy">Healthy</option>
                            <option value="Leaf_Spot">Leaf Spot</option>
                            <option value="Mosaic_Virus">Mosaic Virus</option>
                            <option value="Nutrient_Deficiency">Nutrient Deficiency</option>
                            <option value="Powdery_Mildew">Powdery Mildew</option>
                            <option value="Rust">Rust</option>
                            <option value="Unknown">I don't know</option>
                        </select>
                        <button class="btn" onclick="submitCorrection()" style="margin-top: 10px;">
                            Submit Correction
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div id="notification"></div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const preview = document.getElementById('preview');
        const results = document.getElementById('results');
        const resultsSection = document.getElementById('resultsSection');
        const feedbackSection = document.getElementById('feedbackSection');
        const actualClassSection = document.getElementById('actualClassSection');
        
        let imageData = null;
        let currentPrediction = null;
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', handleFile);
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            }
        });
        
        function handleFile(e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            }
        }
        
        function processFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imageData = e.target.result;
                preview.src = imageData;
                resultsSection.style.display = 'grid';
                analyzeBtn.disabled = false;
                feedbackSection.style.display = 'none';
                actualClassSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        analyzeBtn.addEventListener('click', async () => {
            if (!imageData) return;
            
            results.innerHTML = '<div class="loading"><div class="spinner"></div><p>Analyzing with enhanced model...</p></div>';
            feedbackSection.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                currentPrediction = data;
                displayResults(data);
                feedbackSection.style.display = 'block';
            } catch (error) {
                results.innerHTML = '<p style="color: red;">Error analyzing image. Please try again.</p>';
            }
        });
        
        function displayResults(data) {
            const topPred = data.predictions[0];
            
            let html = `
                <div class="result-item top">
                    <span>${topPred.class}</span>
                    <span>${topPred.confidence.toFixed(1)}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${topPred.confidence}%"></div>
                </div>
            `;
            
            if (data.info) {
                html += `
                    <div class="disease-info">
                        <h4 style="color: #1e40af; margin-bottom: 10px;">📋 Diagnosis</h4>
                        <p>${data.info.description}</p>
                        <h4 style="color: #1e40af; margin: 15px 0 10px;">💊 Recommended Action</h4>
                        <p>${data.info.action}</p>
                    </div>
                `;
            }
            
            html += '<div style="margin-top: 20px;"><h4>Other Possibilities:</h4>';
            for (let i = 1; i < Math.min(4, data.predictions.length); i++) {
                html += `
                    <div class="result-item">
                        <span>${data.predictions[i].class}</span>
                        <span style="color: #6b7280;">${data.predictions[i].confidence.toFixed(1)}%</span>
                    </div>
                `;
            }
            html += '</div>';
            
            results.innerHTML = html;
        }
        
        function submitFeedback(isCorrect) {
            if (isCorrect) {
                // Send positive feedback
                fetch('/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: imageData,
                        predicted: currentPrediction.predictions[0].class,
                        actual: currentPrediction.predictions[0].class,
                        correct: true
                    })
                });
                showNotification('Thank you! Your feedback helps improve the model.', 'success');
                feedbackSection.style.display = 'none';
            } else {
                // Show correction options
                actualClassSection.style.display = 'block';
            }
        }
        
        function submitCorrection() {
            const actualClass = document.getElementById('actualClass').value;
            if (!actualClass) {
                showNotification('Please select the actual condition', 'error');
                return;
            }
            
            // Send correction feedback
            fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: imageData,
                    predicted: currentPrediction.predictions[0].class,
                    actual: actualClass,
                    correct: false,
                    all_predictions: currentPrediction.predictions
                })
            });
            
            showNotification('Correction saved! This will help improve future predictions.', 'success');
            feedbackSection.style.display = 'none';
        }
        
        function showNotification(message, type) {
            const notification = document.getElementById('notification');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            notification.style.display = 'block';
            
            setTimeout(() => {
                notification.style.display = 'none';
            }, 3000);
        }
    </script>
</body>
</html>
'''

# Disease information
DISEASE_INFO = {
    'Blight': {
        'description': 'Early or late blight affecting leaves. Common in tomatoes and potatoes.',
        'action': 'Remove affected leaves, apply fungicide, improve air circulation'
    },
    'Healthy': {
        'description': 'Plant appears healthy with no visible diseases.',
        'action': 'Continue regular care and monitoring'
    },
    'Leaf_Spot': {
        'description': 'Bacterial or fungal spots on leaves.',
        'action': 'Remove infected leaves, avoid overhead watering, apply copper spray'
    },
    'Mosaic_Virus': {
        'description': 'Viral infection causing mottled patterns.',
        'action': 'Remove infected plants, control aphids, use resistant varieties'
    },
    'Nutrient_Deficiency': {
        'description': 'Yellowing or discoloration from lack of nutrients.',
        'action': 'Apply balanced fertilizer, check soil pH, improve drainage'
    },
    'Powdery_Mildew': {
        'description': 'White powdery fungal growth on leaves.',
        'action': 'Improve air circulation, apply neem oil or sulfur spray'
    },
    'Rust': {
        'description': 'Orange/brown pustules on leaves.',
        'action': 'Remove infected leaves, apply fungicide, avoid overhead watering'
    }
}

# Disease classes
CLASSES = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
           'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']

# Create directories for feedback
os.makedirs('../failedImages', exist_ok=True)
os.makedirs('../correctImages', exist_ok=True)
os.makedirs('../feedback_logs', exist_ok=True)

# Load model
print("Loading 96% accuracy model...")
model = tf.keras.models.load_model('../rgb_model/models/cyclegan_best.h5')
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array_batch, verbose=0)[0]
        
        # ENHANCED BIAS CORRECTION for healthy plants
        img_array_for_analysis = np.array(image)
        
        # Calculate color statistics
        r_mean = np.mean(img_array_for_analysis[:,:,0]) / 255
        g_mean = np.mean(img_array_for_analysis[:,:,1]) / 255
        b_mean = np.mean(img_array_for_analysis[:,:,2]) / 255
        
        # Calculate greenness indicators
        green_dominance = g_mean - max(r_mean, b_mean)
        vegetation_index = (g_mean - r_mean) / (g_mean + r_mean + 0.01)
        
        # Check for healthy plant characteristics
        is_very_green = green_dominance > 0.02
        has_vegetation = vegetation_index > 0.03
        no_brown = (r_mean - b_mean) < 0.05
        good_green = g_mean > 0.6
        
        # Apply correction
        if (is_very_green or good_green) and (has_vegetation or no_brown):
            predictions[1] *= 50.0  # Massive boost for healthy
            for i in [0, 2, 3, 5, 6]:
                predictions[i] *= 0.02  # Reduce diseases by 98%
        elif good_green or has_vegetation:
            predictions[1] *= 10.0
            
        # Renormalize
        predictions = predictions / np.sum(predictions)
        
        # Sort predictions
        sorted_preds = []
        for i, confidence in enumerate(predictions):
            sorted_preds.append({
                'class': CLASSES[i],
                'confidence': float(confidence * 100)
            })
        sorted_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get top prediction info
        top_class = sorted_preds[0]['class']
        info = DISEASE_INFO.get(top_class, None)
        
        return jsonify({
            'predictions': sorted_preds,
            'info': info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        predicted = data['predicted']
        actual = data['actual']
        correct = data['correct']
        image_data = data['image']
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not correct:
            # Save failed image
            filename = f"{predicted}_predicted_but_{actual}_actual_{timestamp}.png"
            filepath = os.path.join('../failedImages', filename)
            
            # Save image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            image.save(filepath)
            
            # Save metadata
            metadata = {
                'filename': filename,
                'timestamp': timestamp,
                'predicted': predicted,
                'actual': actual,
                'all_predictions': data.get('all_predictions', []),
                'feedback': 'incorrect'
            }
            
            # Append to feedback log
            log_file = '../feedback_logs/feedback_log.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(metadata) + '\n')
            
            print(f"Failed case saved: {filename}")
        else:
            # Optionally save correct predictions for validation
            if np.random.random() < 0.1:  # Save 10% of correct predictions
                filename = f"{predicted}_correct_{timestamp}.png"
                filepath = os.path.join('../correctImages', filename)
                
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image = Image.open(io.BytesIO(image_bytes))
                image.save(filepath)
                
                print(f"Correct case saved: {filename}")
        
        return jsonify({'status': 'success', 'message': 'Feedback received'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌿 PlantPulse Web Server with Feedback Collection")
    print("="*70)
    print("\n📱 Open in your browser: http://localhost:5000")
    print("📸 Upload plant images and provide feedback!")
    print("❌ Failed cases saved to: failedImages/")
    print("📊 Feedback logs saved to: feedback_logs/")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000)