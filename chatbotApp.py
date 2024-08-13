import json
from flask import Flask, request, jsonify, render_template
import random
from transformers import BertForSequenceClassification, BertTokenizerFast, BertForQuestionAnswering, BertTokenizer
from transformers import pipeline
import torch

app = Flask(__name__)


### THIS IS FOR BERT CLASSIFICATION MODEL
#path saved model
model_path = "chatbot"

#get model
model = BertForSequenceClassification.from_pretrained(model_path)
#get tokes
tokenizer= BertTokenizerFast.from_pretrained(model_path)

#send pretrained and saved model and tokenization, device =cpu, since is mac and we dont have gpu
chatbot= pipeline(task="text-classification", model=model, tokenizer=tokenizer, function_to_apply="softmax")

# Reading JSON Data and store in a dictionary
def read_json_as_dict(file_path):
    # Read the JSON data from the file and load it into a Python dictionary
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
data = read_json_as_dict('traindata.json')


### THIS IS FOR BERT QUESTIONANSWERING MODEL
# path for the saved model
qa_model_path = "QAmodel"

# Load the model and tokenizer from the saved directory
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_path)
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_path) 

def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )
    return answer

# This function returns random answer using the context inside the train data file
def get_random_answer(data, context_value):
    for item in data:
        if item['context'] == context_value:
            return random.choice(item['qas'])['answer']
    return "Sorry, I could not answer you. Try different prompt"





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('userInput', '').strip()
    selected_function = data.get('selectedFunction', '')

    if not user_input:
        return jsonify({"response": "No input provided."}), 400

    if selected_function == 'function1':
        # Example function 1: BertClassification
        result = chatbot(user_input)
        score = result[0]['score']
        if score < 0.5:
            return jsonify({"response": "Sorry, I can not understand you, try with a different prompt."}), 200
        context = result[0]['label']
        print(context)
        response = get_random_answer(data, context)
    elif selected_function == 'function2':
        # Example function 2: BertClassification + BertQuestionAnswering
        result = chatbot(user_input) #finding context
        score = result[0]['score']
        if score < 0.5:
            return jsonify({"response": "Sorry, I can not understand you, try with a different prompt."}), 200
        context = result[0]['label']
        response = answer_question(user_input, context)
    else:
        return jsonify({"response": "Invalid function selected.", "score": "", "context": "" }), 400
    
    return jsonify({"response": response, "score": score, "context": context}), 200

if __name__ == '__main__':
    app.run(debug=True)
