# git clone https://huggingface.co/roneneldan/TinyStories-33M
# python

from transformers import AutoModel, AutoTokenizer

model_path = '/Users/yzhao/Workspace/huggingface/TinyStories-33M'
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_text = "Your input text here adfadf"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
hidden_states = outputs.last_hidden_state
import torch
token_ids = torch.argmax(hidden_states, dim=-1)
decoded_text = tokenizer.decode(token_ids[0])
print(decoded_text)
