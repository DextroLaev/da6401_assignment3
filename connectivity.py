import torch
from AttentionModel import Encoder, Decoder_Attention, Attention_Network
from model import Encoder,BeamSearchDecoder,Seq2Seq_Model
from config import *
from dataset import end, Dataset
import json
import wandb
import numpy as np
import random

def export_attention_visualization_multi(model, input_tensors, input_lang, output_lang, path="web_page/attention_data.json"):
    model.encoder.eval()
    model.decoder.eval()
    export_list = []
    with torch.no_grad():
        for input_tensor in input_tensors:
            encoder_outputs, encoder_hidden = model.encoder(input_tensor.unsqueeze(0).to(DEVICE))
            decoder_outputs, attentions1, attentions2 = model.decoder(encoder_outputs, encoder_hidden)
            _, predictions = decoder_outputs.topk(1)
            predictions = predictions.squeeze(0)

            input_chars = [input_lang.index_to_word[idx.item()] for idx in input_tensor if idx.item() != end]
            output_chars = []
            for idx in predictions:
                if idx.item() == end:
                    break
                output_chars.append(output_lang.index_to_word[idx.item()])

            attention_matrix = attentions1.squeeze().cpu().numpy()
            attention_matrix = np.maximum(attention_matrix, 0)
            if attention_matrix.max() > 0:
                attention_matrix = attention_matrix / attention_matrix.max()
            data = {
                "input": input_chars,
                "prediction": output_chars,
                "attention": attention_matrix.tolist()
            }
            export_list.append(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_list, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(export_list)} samples to {path}")

if __name__ == '__main__':
    ######### MAKE SURE U LOAD YOUR MODEL APPROPRIATELY
    encoder = Encoder(cell_type='LSTM', num_layers=3, hidden_dim=256,
                      embed_dim=64, input_dim=INPUT_DIM, dropout_rate=0.3,
                      bidirectional=BIDIRECTIONAL, batch_first=BATCH_FIRST).to(DEVICE)
    decoder = Decoder_Attention(type='LSTM', num_layers=3, hidden_dim=256,
                                dropout_rate=0.3, bidirectional=BIDIRECTIONAL,
                                batch_first=BATCH_FIRST, embed_dim=64, output_dim=OUTPUT_DIM).to(DEVICE)
    model = Attention_Network(encoder=encoder, decoder=decoder).to(DEVICE)
    state = torch.load('models/attention_state.pth', map_location=DEVICE)
    model.load_state_dict(state)


    input_lang, output_lang, train_loader, test_loader, valid_loader = Dataset().load_data(batch_size=16, lang='hi')
    batch = next(iter(test_loader))
    input_batch, _ = batch
    indices = random.sample(range(input_batch.shape[0]), 10)
    input_tensors = [input_batch[idx] for idx in indices]
    export_attention_visualization_multi(model, input_tensors, input_lang, output_lang, path="web_page/attention_data.json")

    with open("web_page/attention_data.json", "r", encoding="utf-8") as f:
        attention_data = json.load(f)

    # Inject JSON as string
    with open("web_page/static.html", "r", encoding="utf-8") as f:
        template = f.read()

    html_content = template.replace("REPLACE_THIS_WITH_JSON", json.dumps(attention_data))

    with open("web_page/attention.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # Log to wandb
    wandb.init(project="dl-assignment3")
    wandb.log({"Visualize Attention": wandb.Html("web_page/attention.html")})
