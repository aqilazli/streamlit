from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

BERT_TOKEN = os.environ.get("BERT_TOKEN", "").strip()
BERT_MODEL = os.environ.get("BERT_MODEL", "").strip()
DISTILBERT_TOKEN = os.environ.get("DISTILBERT_TOKEN", "").strip()
DISTILBERT_MODEL = os.environ.get("DISTILBERT_MODEL", "").strip()
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "bert").lower()

print(f"✅ Tokens loaded")
print(f"📊 Default model: {DEFAULT_MODEL}")
print(f"📊 BERT: {BERT_MODEL}")
print(f"📊 DistilBERT: {DISTILBERT_MODEL}")

def call_hf_api(text, model_name):
    """Call Hugging Face Inference API"""
    if model_name == "bert":
        if not BERT_TOKEN or not BERT_MODEL:
            return None, "BERT not configured"
        token = BERT_TOKEN
        model = BERT_MODEL
    elif model_name == "distilbert":
        if not DISTILBERT_TOKEN or not DISTILBERT_MODEL:
            return None, "DistilBERT not configured"
        token = DISTILBERT_TOKEN
        model = DISTILBERT_MODEL
    else:
        return None, "Unknown model"

    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model}"

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                result = data[0]
                if isinstance(result, list):
                    # Multiple scores, find max
                    scores = sorted(result, key=lambda x: x['score'], reverse=True)
                    top = scores[0]
                    label = top['label'].upper()
                    confidence = top['score']
                    is_phishing = label == 'PHISHING' or label == 'SMISHING' or label == '2'
                    return {
                        'prediction': label,
                        'confidence': confidence,
                        'is_phishing': is_phishing
                    }, None

        return None, f"API error: {response.status_code} - {response.text[:100]}"

    except Exception as e:
        return None, str(e)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json
        print(f"Request data: {data}")
        text = data.get('text', '').strip()
        model_name = data.get('model', DEFAULT_MODEL).lower()

        print(f"Text: {text}, Model: {model_name}")

        if not text:
            print("Error: Empty text")
            return jsonify({'error': 'Empty text'}), 400

        result, error = call_hf_api(text, model_name)

        if error:
            print(f"API error: {error}")
            return jsonify({'error': error}), 400

        if result:
            print(f"Result: {result}")
            result['model_used'] = model_name
            return jsonify(result)

        return jsonify({'error': 'No response from API'}), 500

    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': ['bert', 'distilbert'],
        'default': DEFAULT_MODEL
    })

if __name__ == '__main__':
    print("\nStarting Flask server on port 5000...")
    print("Using Hugging Face Inference API (no local model loading)\n")
    app.run(host='localhost', port=5000, debug=False)
