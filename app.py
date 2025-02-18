from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

app = Flask(__name__)
CORS(app)

# Allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store assignments
assignments = {}

# Load AI detection models with all weights
try:
    model_1 = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector", output_hidden_states=True)
    tokenizer_1 = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    ai_detector_1 = pipeline("text-classification", model=model_1, tokenizer=tokenizer_1, device=0 if torch.cuda.is_available() else -1)

    model_2 = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2", output_hidden_states=True)
    tokenizer_2 = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
    ai_detector_2 = pipeline("text-classification", model=model_2, tokenizer=tokenizer_2, device=0 if torch.cuda.is_available() else -1)

    print("AI Detection Models Loaded Successfully.")
except Exception as e:
    ai_detector_1, ai_detector_2 = None, None
    print(f"Error: AI models could not be loaded: {e}")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    """Extract text content from uploaded file."""
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        elif file_extension == '.pdf':
            reader = PdfReader(file_path)
            return ''.join([page.extract_text() or '' for page in reader.pages])
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return ''

@app.route('/upload', methods=['POST'])
def upload_assignment():
    """Handle file uploads and store text content."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')  # Allow multiple file uploads

    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'}), 400

    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            text_content = extract_text_from_file(file_path)

            if not text_content.strip():
                os.remove(file_path)
                return jsonify({'error': f'File {file.filename} content is empty'}), 400

            student_name = f"Student_{filename}"
            assignments[student_name] = text_content  # Store extracted text

            uploaded_files.append(student_name)

    print(f"Debug: Current Assignments -> {list(assignments.keys())}")  # Debugging

    return jsonify({'message': 'Files uploaded successfully', 'students': uploaded_files})

def compute_similarity():
    """Compute TF-IDF cosine similarity between assignments."""
    if len(assignments) < 2:
        return np.array([])

    texts = list(assignments.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix)

    return np.nan_to_num(similarity, nan=0.0)

def get_ai_detection_results(text):
    """Detect AI-generated content in the text using AI models."""
    if not text.strip():
        return {"label": "Unknown", "score": 0}

    results = []
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]  # Splitting text for model processing

    for chunk in chunks:
        if ai_detector_1:
            results.append(ai_detector_1(chunk)[0])
        if ai_detector_2:
            results.append(ai_detector_2(chunk)[0])

    if results:
        avg_score = sum(res['score'] for res in results) / len(results)
        label_counts = {}
        for res in results:
            label = res['label']
            # Replace LABEL_1 with 'AI-generated'
            if label == "LABEL_1":
                label = "AI-generated"
            label_counts[label] = label_counts.get(label, 0) + 1
        final_label = max(label_counts, key=label_counts.get)
        return {"label": final_label, "score": round(avg_score * 100, 2)}

    return {"label": "Unknown", "score": 0}

@app.route('/compare', methods=['GET'])
def compare_assignments():
    """Compare uploaded assignments for plagiarism and AI-generated content."""
    if len(assignments) < 2:
        return jsonify({"error": "Not enough assignments to compare"}), 400

    similarity_matrix = compute_similarity()
    students = list(assignments.keys())
    results = []
    ai_results = {}

    for i in range(len(students)):
        ai_results[students[i]] = get_ai_detection_results(assignments[students[i]])
        for j in range(i + 1, len(students)):
            sim_score = round(similarity_matrix[i][j] * 100, 2)
            results.append({"student1": students[i], "student2": students[j], "similarity": sim_score})

    print(f"Debug: Similarity Results -> {results}")  # Debugging

    return jsonify({"comparisons": results, "ai_results": ai_results})

@app.route('/reset', methods=['POST'])
def reset_assignments():
    """Clear all uploaded assignments and delete files."""
    global assignments
    assignments.clear()

    # Delete uploaded files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

    print("Debug: Assignments Reset Successfully.")  # Debugging
    return jsonify({"message": "Previous comparisons cleared. Start fresh!"})

if __name__ == '__main__':
    app.run(debug=True)
