from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
MODEL_NAME = "li-lab/ascle-BioBERT-finetune-HEADQA"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Medical keywords for basic filtering
medical_keywords = [
    'disease', 'symptom', 'treatment', 'medicine', 'doctor', 'hospital', 
    'pain', 'diagnosis', 'health', 'medical', 'patient', 'drug', 'therapy',
    'blood', 'heart', 'cancer', 'surgery', 'infection', 'vaccine', 'prescription',
    'allergy', 'virus', 'bacteria', 'chronic', 'acute', 'condition', 'disorder',
    'syndrome', 'physician', 'nurse', 'clinic', 'emergency', 'ambulance', 'pharmacy',
    'medication', 'dose', 'side effect', 'recovery', 'illness', 'disease', 'fever'
]

def is_medical_question(question):
    """Basic check if the question is medical-related"""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in medical_keywords)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    if not is_medical_question(question):
        return jsonify({
            'answer': 'I can only answer medical-related questions. Please ask something related to healthcare or medicine.',
            'is_medical': False
        })
    
    # Process with the model
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted answer
    # Note: This is simplified - you'll need to adjust based on the actual model output format
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # For demonstration - in reality, you'd use the model's actual output
    answer = f"Medical response for question: {question}"
    
    return jsonify({
        'answer': answer,
        'is_medical': True
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
