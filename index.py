from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from flask import Flask, request, jsonify, render_template
import re
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static')
cors = CORS(app)


device = 'cpu'

pretrained_model_path = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_path)
model = GPT2LMHeadModel.from_pretrained(pretrained_model_path)

tokenizer.add_tokens(["<a1n>", "<a1>", "<a2n>", "<a2>", "<content>", "<end>"])
tokenizer.add_special_tokens({'pad_token': '<pad>'})
model.resize_token_embeddings(len(tokenizer))

load_obj = torch.load('./anime_zero_model_only.pt',
                      map_location=torch.device(device))
model.load_state_dict(load_obj['model'])
model = model.to(device)


def gen_text(q, plus_length):
    x = torch.tensor(tokenizer.encode(q), device=device).reshape(1, -1)
    beam_output = model.generate(
        x,
        max_length=1024,
        temperature=0.70,
        top_k=10,
        top_p=0.95,
        pad_token_id=50263,
        do_sample=True,
        num_return_sequences=1,
        repetition_penalty=1.2,
        min_length=int(x.shape[1]) + int(plus_length),
        eos_token_id=50262,
        early_stopping=True
    )
    a = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    print(a)
    a = [i.strip() for i in re.split('<.+?>', a) if len(i.strip())]
    return {"title1": a[0], "des1": a[1], "title2": a[2], "des2": a[3], "reason": a[4]}


@app.route('/query', methods=['GET'])
def query():
    try:
        q = request.args.get('q')
        plus_length = request.args.get('plus_length')
        text = f"<a1n> {q} <a1>"
        return jsonify(gen_text(text, plus_length))
    except Exception as e:
        print(e)
        return str(e), 400


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=13897)
