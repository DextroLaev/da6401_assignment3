"""
attention_export.py

Generate interactive attention visualizations for a trained attention-based seq2seq model.

This script exports token-level attention data for multiple input sequences into a JSON file,
injects it into an HTML template, and logs the resulting page to Weights & Biases for
interactive inspection. When hovering over a predicted character in the HTML, the corresponding
source characters with high attention scores are highlighted.

Usage:
    python attention_export.py
"""

import torch
from AttentionModel import Encoder, Decoder_Attention, Attention_Network
from config import *
from dataset import end, Dataset
import json
import wandb
import numpy as np
import random

def export_attention_visualization_multi(model, input_tensors, input_lang, output_lang, path="web_page/attention_data.json"):
    """
    Run inference on multiple input sequences and export their attention weights.

    For each tensor in `input_tensors`, this function:
      1. Encodes the sequence with `model.encoder`.
      2. Decodes with `model.decoder` to obtain predicted tokens and their attention matrix.
      3. Converts the attention scores to a normalized 2D list.
      4. Records the source characters, predicted characters, and attention matrix.

    The collected data is written as a JSON list to `path`, where each entry has:
        {
            "input":      [char1, char2, ...],
            "prediction": [char1, char2, ...],
            "attention":  [[w11, w12, ...], [w21, w22, ...], ...]
        }

    Args:
        model (Attention_Network): Trained encoder–attention_decoder network.
        input_tensors (List[Tensor]): List of 1D LongTensors containing input token indices.
        input_lang (Corpus): Vocabulary object mapping input indices to characters.
        output_lang (Corpus): Vocabulary object mapping output indices to characters.
        path (str): File path to save the JSON export (default: "web_page/attention_data.json").

    Returns:
        None  Prints a confirmation with the number of samples exported.
    """
    model.encoder.eval()
    model.decoder.eval()
    export_list = []
    with torch.no_grad():
        for input_tensor in input_tensors:
            encoder_outputs, encoder_hidden = model.encoder(input_tensor.unsqueeze(0).to(DEVICE))
            decoder_outputs, _, attentions = model.decoder(encoder_outputs, encoder_hidden)
            _, predictions = decoder_outputs.topk(1)
            predictions = predictions.squeeze(0)

            input_chars = [input_lang.index_to_word[idx.item()] for idx in input_tensor if idx.item() != end]
            output_chars = []
            for idx in predictions:
                if idx.item() == end:
                    break
                output_chars.append(output_lang.index_to_word[idx.item()])

            attention_matrix = attentions.squeeze().cpu().numpy()
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
    encoder = Encoder(cell_type=TYPE, num_layers=ENCODER_NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                      embed_dim=EMBED_DIM, input_dim=INPUT_DIM, dropout_rate=DROPOUT_RATE,
                      bidirectional=BIDIRECTIONAL, batch_first=BATCH_FIRST).to(DEVICE)
    decoder = Decoder_Attention(type=TYPE, num_layers=DECODER_NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                                dropout_rate=DROPOUT_RATE, bidirectional=BIDIRECTIONAL,
                                batch_first=BATCH_FIRST, embed_dim=EMBED_DIM, output_dim=OUTPUT_DIM).to(DEVICE)
    
    model = Attention_Network(encoder=encoder, decoder=decoder).to(DEVICE)
    state = torch.load('models/attention_new.pth', map_location=DEVICE)
    model.load_state_dict(state)


    input_lang, output_lang, train_loader, test_loader, valid_loader = Dataset().load_data(batch_size=BATCH_SIZE, lang='hi')
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
