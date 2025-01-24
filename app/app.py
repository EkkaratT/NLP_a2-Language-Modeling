import torch
from flask import Flask, render_template, request, jsonify
from model import LSTMLanguageModel
import torchtext
import os

app = Flask(__name__)

# Load the saved vocabulary
vocab_path = 'model/vocab.pth'
vocab_dict_path = 'model/vocab_dict.pth'

vocab = torch.load(vocab_path)
vocab_dict = torch.load(vocab_dict_path)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMLanguageModel(vocab_size=len(vocab), emb_dim=1024, hid_dim=1024, num_layers=2, dropout_rate=0.65).to(device)
model.load_state_dict(torch.load('model/best-val-lstm_lm.pt', map_location=device))

# Define the tokenizer (as per your original code)
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Define generate function
def generate(prompt, max_seq_len=30, temperature=0.7, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[token] for token in tokens]
    
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    generated_indices = indices.copy()

    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # Apply softmax
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()
            
            # If predicted token is <unk>, retry
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            generated_indices.append(prediction)
            indices.append(prediction)  # autoregressive process

    # Convert indices back to words
    itos = vocab_dict['itos']
    generated_tokens = [itos[i] for i in generated_indices]
    return ' '.join(generated_tokens)

# Define the home route and handle form submission
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    temperature = float(request.form.get('temperature', 0.7))  # Default to 0.7 if not provided
    generated_text = generate(prompt, temperature=temperature)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
