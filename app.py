from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from classifier.secondary_classifier import run_pipeline as run_pipeline_v2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_DIR_V2'] = 'classifier/models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
@app.route('/v2')
def index_v2():
    return render_template('index_v2.html')

@app.route('/analyze_v2', methods=['POST'])
def analyze_v2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    model_dir = request.form.get('model_dir', app.config['MODEL_DIR_V2'])

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = run_pipeline_v2(filepath, model_dir=model_dir)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)