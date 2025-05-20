import wandb
from config import *
import numpy as np
from dataset import end
import matplotlib.pyplot as plt
import random
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

def get_attention_map(encoder, decoder, dataloader, input_lang, output_lang):
    for inputs, targets in dataloader:
        inputs = inputs.to(DEVICE)
        enc_out, enc_hid = encoder(inputs)
        dec_out, _, attention_map = decoder(enc_out, enc_hid)
        preds = dec_out.argmax(-1).cpu()
        B, T_in = inputs.size()
        _, T_out = preds.size()
        src_words, pred_words = [], []
        for b in range(B):
            w = []
            for t in range(T_in):
                idx = inputs[b, t].item()
                if idx == end: break
                w.append(input_lang.index_to_word[idx])
            src_words.append(''.join(w))
            w = []
            for t in range(T_out):
                idx = preds[b, t].item()
                if idx == end: break
                w.append(output_lang.index_to_word[idx])
            pred_words.append(''.join(w))

        return src_words, pred_words, attention_map.detach().cpu().numpy()

    return [], [], np.zeros((0,0,0))

def plot_attention_grid(src_words, pred_words, attentions):
    B = len(src_words)
    indices = np.random.choice(B, size=min(10, B), replace=False)

    # 3×3 grid for first 9
    fig, axes = plt.subplots(3, 3, figsize=(12,12),constrained_layout=True)
    for ax, idx in zip(axes.flat, indices[:9]):
        heat = attentions[idx]               # [T_out, T_in]
        im = ax.imshow(heat, aspect='auto', cmap='viridis')
        ax.set_title(f"Example #{idx+1}", fontsize=14)

        # English source on x
        ax.set_xticks(np.arange(len(src_words[idx])))
        ax.set_xticklabels(list(src_words[idx]), rotation=90, fontsize=10)
        # Hindi pred on y
        ax.set_yticks(np.arange(len(pred_words[idx])))
        ax.set_yticklabels(list(pred_words[idx]), fontsize=10)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel("Attention weight", fontsize=12)
    grid_path = 'plots'+"/attention_grid_3x3.png"
    
    plt.savefig(grid_path, dpi=300)
    plt.close(fig)

    # the 10th example standalone
    if len(indices) > 9:
        idx = indices[9]
        fig, ax = plt.subplots(figsize=(6,6),constrained_layout=True)
        heat = attentions[idx]
        im = ax.imshow(heat, aspect='auto', cmap='viridis')
        ax.set_title(f"Example #{idx+1}", fontsize=16)

        ax.set_xticks(np.arange(len(src_words[idx])))
        ax.set_xticklabels(list(src_words[idx]), rotation=90, fontsize=12)
        ax.set_yticks(np.arange(len(pred_words[idx])))
        ax.set_yticklabels(list(pred_words[idx]), fontsize=12)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.colorbar(im, ax=ax).ax.set_ylabel("Attention weight", fontsize=12)

        solo_path = 'plots'+"/attention_example_10th.png"
        plt.savefig(solo_path, dpi=300)
        plt.close(fig)

def generate_word_heatmap(encoder, decoder, dataloader, input_lang, output_lang,
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
                ax.matshow(attn,cmap='viridis')
                ax.set_aspect('equal')

                # # set ticks so that columns=input, rows=predicted
                ax.set_xticks(np.arange(L_in))
                ax.set_xticklabels(input_word, rotation=0, fontsize=12)
                ax.set_yticks(np.arange(L_out))
                ax.set_yticklabels(pred_word, fontsize=12)


                # ax.xaxis.set_ticks_position('bottom')
                ax.set_xlabel('Source (input)', fontsize=14)
                ax.set_ylabel('Prediction (output)', fontsize=14)
                ax.set_title(f'Attention heatmap {batch_idx*B + i + 1}', fontsize=16)

                # # colorbar just like your first heatmap

                save_path = f"predicted_attention/attention_{name}_{batch_idx}_{i}.png"
                plt.savefig(save_path, dpi=300)
                plt.close(fig)

                count += 1
                if count >= num_heatmaps:
                    return
                


def log_test_predictions_to_wandb(encoder, decoder, test_loader, input_lang, output_lang, name="vanilla"):
    
    encoder.eval()
    decoder.eval()
    wandb.login()
    wandb.init(project='dl-assignment3')
    table = wandb.Table(columns=["Input", "Prediction", "Ground Truth", "Correct"])

    all_data = []
    write_data_file = []

    # For confusion matrix
    labels = sorted(output_lang.word_to_index.keys())
    char_to_index = {char: i for i, char in enumerate(labels)}
    confusion = torch.zeros(len(labels), len(labels), dtype=torch.int32)

    with torch.no_grad():
        for batch in test_loader:
            inputs, outputs = batch
            inputs = inputs.to(DEVICE)
            encoder_outputs, encoder_hidden = encoder(inputs)
            if name == 'vanilla':
                decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)
            elif name == 'attention':
                decoder_outputs,_,attention_weights = decoder(
                    encoder_outputs, encoder_hidden)                
            _, topi = decoder_outputs.topk(1)
            predicted_ids = topi.squeeze()

            for i in range(inputs.size(0)):
                input_word = []
                predicted_word = []
                output_word = []

                for j in range(MAX_LENGTH):
                    if inputs[i][j].item() == end:
                        break
                    input_word.append(input_lang.index_to_word[inputs[i][j].item()])

                for j in range(MAX_LENGTH):
                    if predicted_ids[i][j].item() == end:
                        break
                    predicted_word.append(output_lang.index_to_word[predicted_ids[i][j].item()])

                for j in range(MAX_LENGTH):
                    if outputs[i][j].item() == end:
                        break
                    output_word.append(output_lang.index_to_word[outputs[i][j].item()])

                # Update confusion matrix
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

        # Save predictions
        with open(f'predicted_{name}/{name}.txt', 'w') as f:
            for item in write_data_file:
                for datum in item:
                    f.write("%s,\t" % datum)
                f.write("\n")
        print(f'{name}.txt saved in predicted_{name}/')

        # Save sample predictions to wandb
        sampled = random.sample(all_data, 10)
        for row in sampled:
            table.add_data(*row)
        wandb.log({f"{name} Predictions": table})

        confusion_np = confusion.numpy()
        fig, ax = plt.subplots(figsize=(20, 20),dpi=300)
        im = ax.imshow(confusion_np, cmap='Blues')

        # Tick labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=90)
        ax.set_yticklabels(labels, fontsize=8)

        # Axis labels
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("Ground Truth", fontsize=14)
        ax.set_title("Character-level Confusion Matrix", fontsize=16)

        # Add counts to cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                count = confusion_np[i, j]
                if count > 0:
                    ax.text(j, i, str(count), ha='center', va='center', color='black', fontsize=6)

        # Add colorbar
        fig.colorbar(im, ax=ax)
        plt.tight_layout()

        # Save locally
        save_path = f'plots/confusion_matrix_{name}.png'
        plt.savefig(save_path)
        print(f'Confusion matrix saved to {save_path}')
        plt.close()
        
        wandb.log({f"Confusion Matrix Image ({name})": wandb.Image(save_path)})

        if name == 'attention':
            # generate_word_heatmap(encoder=encoder,decoder=decoder,dataloader=test_loader,input_lang=input_lang,output_lang=output_lang,name=name)
            src_words, pred_words, attentions = get_attention_map(encoder, decoder, test_loader,input_lang, output_lang)
            plot_attention_grid(src_words, pred_words, attentions)
            # log both images to W&B
            wandb.log({
                "Attention Grid 3x3": wandb.Image("plots/attention_grid_3x3.png"),
                "Attention Example 10th": wandb.Image("plots/attention_example_10th.png")
            })

