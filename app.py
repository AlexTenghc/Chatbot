
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
#model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
model_name = "gpt2-large"  # You can use "gpt2-medium", "gpt2-large", etc., for larger models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

app = Flask(__name__)

# initiate input
step = 0
chat_history_ids = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    global step
    global chat_history_ids
    question = request.form['user_input']
    
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')

    # add new input tokens to chat history
    #bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    input_ids = tokenizer.encode(question, return_tensors="pt")

    # generate response
    response_ids = model.generate(
        input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id
    )
    
    #answer = ("{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    answer = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    step += 1

    return answer

if __name__ == '__main__':
    app.run(debug=True)
