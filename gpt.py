# import argparse
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# import os
# import tqdm

# # Special tokens
# SOS_TOKEN = '<sos>'
# EOS_TOKEN = '<eos>'
# PAD_TOKEN = '<pad>'

# class TransliterationDataset(Dataset):
#     def __init__(self, filepath, src_vocab=None, tgt_vocab=None):
#         self.pairs = []
#         with open(filepath, 'r', encoding='utf-8') as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 if len(parts) < 2:
#                     continue
#                 src, tgt = parts[1], parts[0]
#                 self.pairs.append((src, tgt))
#         # build vocabs if not provided
#         if src_vocab is None or tgt_vocab is None:
#             self.build_vocabs()
#         else:
#             self.src_vocab = src_vocab
#             self.tgt_vocab = tgt_vocab

#     def build_vocabs(self):
#         src_chars = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
#         tgt_chars = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
#         for src, tgt in self.pairs:
#             src_chars.update(list(src))
#             tgt_chars.update(list(tgt))
#         self.src_vocab = {ch: i for i, ch in enumerate(sorted(src_chars))}
#         self.tgt_vocab = {ch: i for i, ch in enumerate(sorted(tgt_chars))}
#         self.src_ivocab = {i: ch for ch, i in self.src_vocab.items()}
#         self.tgt_ivocab = {i: ch for ch, i in self.tgt_vocab.items()}

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         src, tgt = self.pairs[idx]
#         # add EOS to target
#         src_idx = [self.src_vocab[ch] for ch in src]
#         tgt_idx = [self.tgt_vocab[SOS_TOKEN]] + [self.tgt_vocab[ch] for ch in tgt] + [self.tgt_vocab[EOS_TOKEN]]
#         return torch.tensor(src_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long)

#     @staticmethod
#     def collate_fn(batch):
#         src_batch, tgt_batch = zip(*batch)
#         src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
#         tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
#         return src_batch, tgt_batch

# class Seq2Seq(nn.Module):
#     def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, cell_type='LSTM', num_layers=1):
#         super().__init__()
#         self.encoder_embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
#         self.decoder_embed = nn.Embedding(output_dim, embed_dim, padding_idx=0)
#         if cell_type == 'LSTM':
#             self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
#             self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
#         elif cell_type == 'GRU':
#             self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
#             self.decoder_rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
#         else:
#             self.encoder_rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
#             self.decoder_rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc_out = nn.Linear(hidden_dim, output_dim)
#         self.cell_type = cell_type
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#     def forward(self, src, tgt, teacher_forcing_ratio=0.5):
#         batch_size, tgt_len = tgt.size()
#         outputs = torch.zeros(batch_size, tgt_len, self.fc_out.out_features).to(src.device)
#         # encode
#         enc_embed = self.encoder_embed(src)
#         enc_outputs, hidden = self.encoder_rnn(enc_embed)
#         # prepare inputs for decoder
#         input_tok = tgt[:, 0]  # SOS
#         if self.cell_type == 'LSTM':
#             hidden_state, cell_state = hidden
#             hidden = (hidden_state, cell_state)
#         for t in range(1, tgt_len):
#             dec_embed = self.decoder_embed(input_tok).unsqueeze(1)  # (B,1,E)
#             if self.cell_type == 'LSTM':
#                 output, (hidden_state, cell_state) = self.decoder_rnn(dec_embed, (hidden_state, cell_state))
#                 hidden = (hidden_state, cell_state)
#             else:
#                 output, hidden = self.decoder_rnn(dec_embed, hidden)
#             pred = self.fc_out(output.squeeze(1))  # (B, output_dim)
#             outputs[:, t, :] = pred
#             teacher_force = torch.rand(1).item() < teacher_forcing_ratio
#             top1 = pred.argmax(1)
#             input_tok = tgt[:, t] if teacher_force else top1
#         return outputs

#     def greedy_decode(self, src, max_len):
#         self.eval()
#         with torch.no_grad():
#             enc_embed = self.encoder_embed(src)
#             enc_outputs, hidden = self.encoder_rnn(enc_embed)
#             if self.cell_type == 'LSTM':
#                 hidden_state, cell_state = hidden
#                 hidden = (hidden_state, cell_state)
#             input_tok = torch.full((src.size(0),), fill_value=1, dtype=torch.long, device=src.device)  # SOS idx assumed 1
#             outputs = []
#             for _ in range(max_len):
#                 dec_embed = self.decoder_embed(input_tok).unsqueeze(1)
#                 if self.cell_type == 'LSTM':
#                     output, (hidden_state, cell_state) = self.decoder_rnn(dec_embed, (hidden_state, cell_state))
#                     hidden = (hidden_state, cell_state)
#                 else:
#                     output, hidden = self.decoder_rnn(dec_embed, hidden)
#                 pred = self.fc_out(output.squeeze(1))
#                 top1 = pred.argmax(1)
#                 outputs.append(top1.unsqueeze(1))
#                 input_tok = top1
#             outputs = torch.cat(outputs, dim=1)
#         return outputs


# def compute_word_accuracy(model, dataloader, ivocab_tgt, device):
#     model.eval()
#     total, correct = 0, 0
#     max_len = dataloader.dataset.pairs and max(len(tgt) for _, tgt in dataloader.dataset.pairs) + 2
#     for src_batch, tgt_batch in dataloader:
#         src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
#         preds = model.greedy_decode(src_batch, max_len)
#         for pred_seq, tgt_seq in zip(preds, tgt_batch):
#             # convert to chars, stop at EOS
#             pred_chars = []
#             for idx in pred_seq.tolist():
#                 if ivocab_tgt[idx] == EOS_TOKEN:
#                     break
#                 pred_chars.append(ivocab_tgt[idx])
#             tgt_chars = []
#             for idx in tgt_seq.tolist()[1:]:  # skip SOS
#                 ch = ivocab_tgt[idx]
#                 if ch == EOS_TOKEN:
#                     break
#                 tgt_chars.append(ch)
#             if ''.join(pred_chars) == ''.join(tgt_chars):
#                 correct += 1
#             total += 1
#     return correct / total if total > 0 else 0


# def train_model(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # load datasets
#     train_ds = TransliterationDataset(args.train)
#     val_ds = TransliterationDataset(args.val, src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
#     test_ds = TransliterationDataset(args.test, src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=TransliterationDataset.collate_fn)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=TransliterationDataset.collate_fn)

#     # initialize model
#     model = Seq2Seq(
#         input_dim=len(train_ds.src_vocab),
#         output_dim=len(train_ds.tgt_vocab),
#         embed_dim=args.embed_dim,
#         hidden_dim=args.hidden_dim,
#         cell_type=args.cell_type,
#         num_layers=args.num_layers
#     ).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.CrossEntropyLoss(ignore_index=train_ds.tgt_vocab[PAD_TOKEN])

#     for epoch in tqdm.tqdm(range(args.epochs)):
#         model.train()
#         epoch_loss = 0
#         for src_batch, tgt_batch in train_loader:
#             src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=args.teacher_forcing)
#             # outputs: (B, T, V)
#             loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt_batch[:, 1:].reshape(-1))
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         avg_loss = epoch_loss / len(train_loader)

#         train_acc = compute_word_accuracy(model, train_loader, train_ds.tgt_ivocab, device)
#         val_acc = compute_word_accuracy(model, val_loader, train_ds.tgt_ivocab, device)
#         print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}")

#     # final test accuracy
#     test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=TransliterationDataset.collate_fn)
#     test_acc = compute_word_accuracy(model, test_loader, train_ds.tgt_ivocab, device)
#     print(f"Test Word-level Accuracy: {test_acc:.2f}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Seq2Seq Transliteration')
#     parser.add_argument('--train', type=str, required=True, help='path to train.tsv')
#     parser.add_argument('--val', type=str, required=True, help='path to val.tsv')
#     parser.add_argument('--test', type=str, required=True, help='path to test.tsv')
#     parser.add_argument('--cell_type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='LSTM')
#     parser.add_argument('--embed_dim', type=int, default=64)
#     parser.add_argument('--hidden_dim', type=int, default=128)
#     parser.add_argument('--num_layers', type=int, default=3)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--teacher_forcing', type=float, default=0.5)
#     args = parser.parse_args()
#     train_model(args)


import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb

# Special tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

class TransliterationDataset(Dataset):
    def __init__(self, filepath, src_vocab=None, tgt_vocab=None):
        self.pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                src, tgt = parts[1], parts[0]
                self.pairs.append((src, tgt))
        if src_vocab and tgt_vocab:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
        else:
            self.build_vocabs()

    def build_vocabs(self):
        src_chars = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
        tgt_chars = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
        for src, tgt in self.pairs:
            src_chars.update(src)
            tgt_chars.update(tgt)
        self.src_vocab = {ch: i for i, ch in enumerate(sorted(src_chars))}
        self.tgt_vocab = {ch: i for i, ch in enumerate(sorted(tgt_chars))}
        self.src_ivocab = {i: ch for ch, i in self.src_vocab.items()}
        self.tgt_ivocab = {i: ch for ch, i in self.tgt_vocab.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_idx = [self.src_vocab[ch] for ch in src]
        tgt_idx = [self.tgt_vocab[SOS_TOKEN]] + [self.tgt_vocab[ch] for ch in tgt] + [self.tgt_vocab[EOS_TOKEN]]
        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
        return src_padded, tgt_padded

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim,
                 embed_dim=64, hidden_dim=128,
                 cell_type='LSTM',
                 enc_layers=1, dec_layers=1,
                 dropout=0.2):
        super().__init__()
        self.cell_type = cell_type
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        # embeddings with dropout
        self.encoder_embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.decoder_embed = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        # choose RNN type
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        # encoder and decoder
        self.encoder_rnn = rnn_cls(
            embed_dim, hidden_dim, enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0
        )
        self.decoder_rnn = rnn_cls(
            embed_dim, hidden_dim, dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def _init_decoder_hidden(self, hidden):
        # Align encoder hidden state to decoder layers
        if self.cell_type == 'LSTM':
            h, c = hidden
            # Truncate or pad h and c to dec_layers
            if self.dec_layers < self.enc_layers:
                h = h[:self.dec_layers]
                c = c[:self.dec_layers]
            elif self.dec_layers > self.enc_layers:
                extra = self.dec_layers - self.enc_layers
                zeros_h = torch.zeros(extra, *h.shape[1:], device=h.device, dtype=h.dtype)
                zeros_c = torch.zeros(extra, *c.shape[1:], device=c.device, dtype=c.dtype)
                h = torch.cat([h, zeros_h], dim=0)
                c = torch.cat([c, zeros_c], dim=0)
            return (h, c)
        else:
            # hidden shape: (enc_layers, B, H)
            if self.dec_layers < self.enc_layers:
                return hidden[:self.dec_layers]
            elif self.dec_layers > self.enc_layers:
                extra = self.dec_layers - self.enc_layers
                zeros = torch.zeros(extra, *hidden.shape[1:], device=hidden.device, dtype=hidden.dtype)
                return torch.cat([hidden, zeros], dim=0)
            else:
                return hidden

    def forward(self, src, tgt, teacher_forcing=0.5):
        B, T = tgt.size()
        print(T)
        input()
        device = src.device
        outputs = torch.zeros(B, T, self.fc_out.out_features, device=device)
        # encode
        emb_src = self.emb_dropout(self.encoder_embed(src))
        enc_out, hidden_enc = self.encoder_rnn(emb_src)
        # init decoder hidden
        dec_hidden = self._init_decoder_hidden(hidden_enc)
        # decode
        input_tok = tgt[:, 0]
        for t in range(1, T):
            emb_dec = self.emb_dropout(self.decoder_embed(input_tok)).unsqueeze(1)
            out, dec_hidden = self.decoder_rnn(emb_dec, dec_hidden)
            logits = self.fc_out(out.squeeze(1))
            outputs[:, t] = logits
            teacher_force = torch.rand(1).item() < teacher_forcing
            top1 = logits.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return outputs

    def greedy_decode(self, src, max_len):
        self.eval()
        with torch.no_grad():
            emb_src = self.emb_dropout(self.encoder_embed(src))
            enc_out, hidden_enc = self.encoder_rnn(emb_src)
            dec_hidden = self._init_decoder_hidden(hidden_enc)
            batch_size = src.size(0)
            input_tok = torch.full((batch_size,), self.decoder_embed.padding_idx+1,
                                   device=src.device, dtype=torch.long)
            outputs = []
            for _ in range(max_len):
                emb = self.emb_dropout(self.decoder_embed(input_tok)).unsqueeze(1)
                out, dec_hidden = self.decoder_rnn(emb, dec_hidden)
                logits = self.fc_out(out.squeeze(1))
                top1 = logits.argmax(1)
                outputs.append(top1.unsqueeze(1))
                input_tok = top1
            return torch.cat(outputs, 1)

def compute_word_accuracy(model, dataloader, ivocab, device):
    model.eval()
    total, correct = 0, 0
    max_len = max(len(t) for _, t in dataloader.dataset.pairs) + 2
    for src_b, tgt_b in dataloader:
        src_b, tgt_b = src_b.to(device), tgt_b.to(device)
        preds = model.greedy_decode(src_b, max_len)
        for p, t in zip(preds, tgt_b):
            pred_str = ''.join(ivocab[idx] for idx in p.tolist() if ivocab[idx] != EOS_TOKEN)
            tgt_str  = ''.join(ivocab[idx] for idx in t.tolist()[1:] if ivocab[idx] != EOS_TOKEN)
            correct += (pred_str == tgt_str)
            total += 1
    return correct/total if total else 0

def compute_char_accuracy(model, dataloader, pad_idx, device):
    """
    Returns fraction of non-pad target characters correctly predicted.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src_b, tgt_b in dataloader:
            src_b, tgt_b = src_b.to(device), tgt_b.to(device)
            # get raw logits with no teacher forcing
            logits = model(src_b, tgt_b, teacher_forcing=0.0)   # shape (B, T, V)
            preds = logits.argmax(dim=-1)                      # shape (B, T)
            # ignore the initial SOS token, and pads
            tgt_seq = tgt_b[:,1:]
            pred_seq = preds[:,1:]
            mask = tgt_seq.ne(pad_idx)
            correct += (pred_seq[mask] == tgt_seq[mask]).sum().item()
            total   += mask.sum().item()
    return correct/total if total>0 else 0.0


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = TransliterationDataset(args.train)
    val_ds   = TransliterationDataset(args.val,   src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds  = TransliterationDataset(args.test,  src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=TransliterationDataset.collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=TransliterationDataset.collate_fn)

    model = Seq2Seq(
        input_dim=len(train_ds.src_vocab),
        output_dim=len(train_ds.tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        cell_type=args.cell_type,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=train_ds.tgt_vocab[PAD_TOKEN])

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for src_b, tgt_b in train_loader:
            src_b, tgt_b = src_b.to(device), tgt_b.to(device)
            optimizer.zero_grad()
            out = model(src_b, tgt_b, teacher_forcing=args.teacher_forcing)
            loss = criterion(out[:,1:].reshape(-1, out.size(-1)), tgt_b[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_word_acc = compute_word_accuracy(model, train_loader, train_ds.tgt_ivocab, device)
        val_word_acc = compute_word_accuracy(model, val_loader, train_ds.tgt_ivocab, device)
        pad_idx  = train_ds.tgt_vocab[PAD_TOKEN]
        train_char_acc = compute_char_accuracy(model, train_loader, pad_idx, device)
        val_char_acc   = compute_char_accuracy(model, val_loader,   pad_idx, device)
        print(f"Epoch {epoch+1}/{args.epochs} | "              
              f"Train ▶ word: {train_word_acc:.2f}, char: {train_char_acc:.4f} | "
              f" Val ▶ word: {val_word_acc:.2f}, char: {val_char_acc:.4f}")

             

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=TransliterationDataset.collate_fn)
    print(f"Test Word-level Accuracy: {compute_word_accuracy(model, test_loader, train_ds.tgt_ivocab, device):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',        type=str, default='train.tsv')
    parser.add_argument('--val',          type=str, default='val.tsv')
    parser.add_argument('--test',         type=str, default='test.tsv')
    parser.add_argument('--cell_type',    choices=['RNN','GRU','LSTM'], default='LSTM')
    parser.add_argument('--embed_dim',    type=int, default=64, choices=[16,32,64,256])
    parser.add_argument('--hidden_dim',   type=int, default=128, choices=[16,32,64,256])
    parser.add_argument('--enc_layers',   type=int, default=3, choices=[1,2,3])
    parser.add_argument('--dec_layers',   type=int, default=3, choices=[1,2,3])
    parser.add_argument('--dropout',      type=float, default=0.2, choices=[0.2,0.3])
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--epochs',       type=int, default=20)
    parser.add_argument('--teacher_forcing', type=float, default=0.5)
    args = parser.parse_args()
    train_model(args)
    
