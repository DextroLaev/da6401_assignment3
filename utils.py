"""
Utility functions for attention visualization and WandB logging in seq2seq models.
"""
import wandb
from config import *
import numpy as np
from dataset import end
import matplotlib.pyplot as plt
import random
from matplotlib import ticker
import os
import torch
    

def generate_individual_heatmap(encoder, decoder, dataloader, input_lang, output_lang,
                          name='', num_heatmaps=10):
    encoder.eval()
    decoder.eval()

    count = 0
    with torch.no_grad():
        for batch_idx, (inputs, outputs) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            encoder_outputs, encoder_hidden = encoder(inputs)
            decoder_outputs, hidden_states, attention_weights = decoder(
                encoder_outputs, encoder_hidden
            )

            # pick the top‐1 prediction per time‐step
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()  # [B, T_out]

            B, T_in = inputs.size()
            _, T_out = decoded_ids.size()

            for i in range(B):
                # reconstruct the two strings
                input_word = []
                for t in range(T_in):
                    idx = inputs[i, t].item()
                    if idx == end: break
                    input_word.append(input_lang.index_to_word[idx])

                pred_word = []
                for t in range(T_out):
                    idx = decoded_ids[i, t].item()
                    if idx == end: break
                    pred_word.append(output_lang.index_to_word[idx])

                # slice out just the valid part of the attention [T_out, T_in]
                L_out = len(pred_word)
                L_in  = len(input_word)
                attn = attention_weights[i, :L_out, :L_in].cpu().numpy()
                
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(1,1,1)
                img = ax.matshow(attn,cmap='viridis')
                ax.set_aspect('equal')

                # # set ticks so that columns=input, rows=predicted
                ax.set_xticks(np.arange(L_in))
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_label_position('bottom')
                ax.tick_params(axis='x', labelbottom=True, labeltop=False)
                ax.set_yticks(np.arange(L_out))
                ax.set_yticklabels(pred_word, fontsize=12)


                # ax.xaxis.set_ticks_position('bottom')
                ax.set_xlabel('Source (input)', fontsize=14)
                ax.set_ylabel('Prediction (output)', fontsize=14)
                ax.set_title(f'Attention heatmap {batch_idx*B + i + 1}', fontsize=16)

                cb = fig.colorbar(img, orientation='vertical',pad=0.15)
                cb.ax.yaxis.set_ticks_position('left')
                cb.ax.yaxis.set_label_position('left')
                cb.ax.set_ylabel("Attention weight", fontsize=10)
                # # colorbar just like your first heatmap

                save_path = f"plots/attention_{name}_{batch_idx}_{i}.png"
                plt.savefig(save_path, dpi=300)
                plt.close(fig)

                count += 1
                if count >= num_heatmaps:
                    return

def generate_word_heatmap(
    encoder, decoder, dataloader, input_lang, output_lang,
    name='', num_heatmaps=9
):
    """
    Generate a 3×3 grid of word-level attention heatmaps with one colorbar
    to the right of each heatmap.

    Args:
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model with attention.
        dataloader (DataLoader): Provides (inputs, outputs) batches.
        input_lang, output_lang: vocab objects for index→char.
        name (str): Prefix for the saved file name.
        num_heatmaps (int): How many examples to plot (max 9).
    """
    encoder.eval()
    decoder.eval()

    # 1) Collect up to num_heatmaps examples
    examples = []
    with torch.no_grad():
        for inputs, outputs in dataloader:
            inputs = inputs.to(DEVICE)
            enc_out, enc_hid = encoder(inputs)
            dec_out, _, attn = decoder(enc_out, enc_hid)
            _, topi = dec_out.topk(1)
            decoded = topi.squeeze(-1).cpu().numpy()

            B, T_in = inputs.size()
            T_out = decoded.shape[1]
            for b in range(B):
                # Build source and predicted character lists
                src, pred = [], []
                for t in range(T_in):
                    i = inputs[b, t].item()
                    if i == end: break
                    src.append(input_lang.index_to_word[i])
                for t in range(T_out):
                    i = decoded[b, t]
                    if i == end: break
                    pred.append(output_lang.index_to_word[i])
                # Slice the attention matrix to actual lengths
                mat = attn[b, :len(pred), :len(src)].cpu().numpy()
                examples.append((src, pred, mat))
                if len(examples) >= num_heatmaps:
                    break
            if len(examples) >= num_heatmaps:
                break

    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # 2) Create a GridSpec: each row has [heatmap, cbar, heatmap, cbar, …]
    rows, cols = 3, 3
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(
        nrows=rows,
        ncols=cols * 2,
        width_ratios=[1, 0.05] * cols,
        wspace=0.3,
        hspace=0.3
    )
    axes = gs.subplots()

    # 3) Plot each example
    for idx, (src, pred, mat) in enumerate(examples):
        r = idx // cols
        c = idx % cols

        # Heatmap at column 2*c
        ax = axes[r, 2*c]
        im = ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_title("Attention heatmap", fontsize=16)
        ax.set_xlabel("Source (input)", fontsize=14)
        ax.set_ylabel("Prediction (output)", fontsize=14)
        ax.set_xticks(range(len(src)))
        ax.set_xticklabels(src, rotation=0, fontsize=10)
        ax.set_yticks(range(len(pred)))
        ax.set_yticklabels(pred, fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # Colorbar at column 2*c + 1
        cax = axes[r, 2*c + 1]
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')
        cb.ax.set_ylabel("Attention weight", fontsize=10)

    # Hide any leftover axes if fewer than rows*cols examples
    total = len(examples)
    for ax in axes.flatten()[total*2:]:
        ax.set_visible(False)

    # 4) Save and log
    out_path = f"plots/attention_grid_{name}_3x3.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    wandb.log({f"Attention Grid ({name})": wandb.Image(out_path)})



def log_test_predictions_to_wandb(encoder, decoder, test_loader, input_lang, output_lang,name="vanilla"):
    """
    Log model predictions and confusion matrices to WandB, and optionally
    attention visualizations for attention-based decoders.

    Args:
        encoder (torch.nn.Module): Encoder model.
        decoder (torch.nn.Module): Decoder model (vanilla or with attention).
        test_loader (DataLoader): DataLoader for test set.
        input_lang (Language): Source language vocabulary.
        output_lang (Language): Target language vocabulary.
        name (str): Identifier for the run ('vanilla' or 'attention').
    """
    

    encoder.eval()
    decoder.eval()
    wandb.login()
    wandb.init(project='dl-assignment3')
    table = wandb.Table(columns=["Input", "Prediction", "Ground Truth", "Correct"])

    all_data = []
    write_data_file = []

    # Prepare confusion matrix
    labels = sorted(output_lang.word_to_index.keys())
    char_to_index = {char: i for i, char in enumerate(labels)}
    confusion = torch.zeros(len(labels), len(labels), dtype=torch.int32)

    with torch.no_grad():
        for inputs, outputs in test_loader:
            inputs = inputs.to(DEVICE)
            encoder_outputs, encoder_hidden = encoder(inputs)
            if name == 'vanilla':
                decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)
            elif name == 'attention':
                decoder_outputs, _, attention_weights = decoder(
                    encoder_outputs, encoder_hidden
                )
            _, topi = decoder_outputs.topk(1)
            predicted_ids = topi.squeeze()

            for i in range(inputs.size(0)):
                # Decode inputs, predictions, and ground truth
                input_word, predicted_word, output_word = [], [], []
                for j in range(MAX_LENGTH):
                    if inputs[i][j].item() == end: break
                    input_word.append(input_lang.index_to_word[inputs[i][j].item()])
                for j in range(MAX_LENGTH):
                    if predicted_ids[i][j].item() == end: break
                    predicted_word.append(output_lang.index_to_word[predicted_ids[i][j].item()])
                for j in range(MAX_LENGTH):
                    if outputs[i][j].item() == end: break
                    output_word.append(output_lang.index_to_word[outputs[i][j].item()])

                # Update confusion matrix counts
                min_len = min(len(output_word), len(predicted_word))
                for k in range(min_len):
                    true_char = output_word[k]
                    pred_char = predicted_word[k]
                    if true_char in char_to_index and pred_char in char_to_index:
                        confusion[char_to_index[true_char], char_to_index[pred_char]] += 1

                input_str = ''.join(input_word)
                predicted_str = ''.join(predicted_word)
                target_str = ''.join(output_word)
                correct = predicted_str == target_str

                write_data_file.append([input_str, predicted_str, target_str])
                all_data.append([input_str, predicted_str, target_str, correct])

        # Save raw predictions
        output_dir = f'predicted_{name}'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/{name}.txt', 'w') as f:
            for item in write_data_file:
                f.write(",\t".join(item) + "\n")
        print(f'{name}.txt saved in {output_dir}/')

        # Log sample table to WandB
        sampled = random.sample(all_data, min(len(all_data), 10))
        for row in sampled:
            table.add_data(*row)
        wandb.log({f"{name} Predictions": table})

        # If attention model, log additional heatmaps
        if name == 'attention':
            generate_word_heatmap(
                encoder=encoder,
                decoder=decoder,
                dataloader=test_loader,
                input_lang=input_lang,
                output_lang=output_lang,
                name=name
            )
            generate_individual_heatmap(encoder, decoder,test_loader, input_lang,output_lang)
            print('Attention visualizations logged.')
