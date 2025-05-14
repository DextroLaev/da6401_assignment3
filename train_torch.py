import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths
TRAIN_PATH = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
DEV_PATH   = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
TEST_PATH  = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'

# Hyperparams
BATCH_SIZE            = 64
EPOCHS                = 30
HIDDEN_SIZE           = 256
TEACHER_FORCING_RATIO = 0.5
MAX_SAMPLES           = 10000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------- #
# Data prep (as before)...
# ---------------------------------------------------------------------------- #
# --- In read_data, replace <sos>/<eos> with '\t' and '\n' ---
def read_data(path, max_samples=None):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            src, tgt = parts
            # wrap with single-char tokens
            tgt = '\t' + tgt + '\n'
            pairs.append((src, tgt))
    return pairs

# load
train_pairs = read_data(TRAIN_PATH, MAX_SAMPLES)
dev_pairs   = read_data(DEV_PATH,   MAX_SAMPLES)

# --- build character vocab including '\t' and '\n' automatically ---
all_src_chars = set(c for s,_ in train_pairs for c in s)
all_tgt_chars = set(c for _,t in train_pairs for c in t)

def build_vocab(chars):
    idx2char = ['<pad>', '<unk>'] + sorted(chars)
    char2idx = {ch: idx for idx, ch in enumerate(idx2char)}
    return char2idx, idx2char

src2i, i2src = build_vocab(all_src_chars)
tgt2i, i2tgt = build_vocab(all_tgt_chars)

# now you *can* look up:
SOS_IDX = tgt2i['\t']
EOS_IDX = tgt2i['\n']


# Dataset + collate
class TransliterationDataset(Dataset):
    def __init__(self,pairs): self.pairs=pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx):
        s,t = self.pairs[idx]
        return (
            torch.tensor([src2i.get(c,1) for c in s],dtype=torch.long),
            torch.tensor([tgt2i.get(c,1) for c in t],dtype=torch.long)
        )

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)
    return src.to(DEVICE), tgt.to(DEVICE)

train_loader = DataLoader(TransliterationDataset(train_pairs), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(TransliterationDataset(dev_pairs),   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------------------------------- #
# Model (same as before)...
# ---------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self,vocab,emb,hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab,emb,padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb,hidden,batch_first=True)
    def forward(self,x): return self.lstm(self.emb(x))

class Decoder(nn.Module):
    def __init__(self,vocab,emb,hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab,emb,padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,vocab)
    def forward(self,input_step,hidden):
        out, hidden = self.lstm(self.emb(input_step), hidden)
        return self.fc(out.squeeze(1)), hidden

class Seq2Seq(nn.Module):
    def __init__(self,enc,dec):
        super().__init__()
        self.enc, self.dec = enc, dec
    def forward(self,src,tgt,tf_ratio):
        bs, tgt_len = tgt.size()
        outputs = torch.zeros(bs, tgt_len, TGT_VOCAB, device=DEVICE)
        _, (h,c) = self.enc(src)
        input_tok = tgt[:,0].unsqueeze(1)
        hidden = (h,c)
        for t in range(1, tgt_len):
            out, hidden = self.dec(input_tok, hidden)
            outputs[:,t] = out
            teacher = random.random() < tf_ratio
            top1 = out.argmax(1).unsqueeze(1)
            input_tok = tgt[:,t].unsqueeze(1) if teacher else top1
        return outputs

enc = Encoder(SRC_VOCAB, HIDDEN_SIZE, HIDDEN_SIZE).to(DEVICE)
dec = Decoder(TGT_VOCAB, HIDDEN_SIZE, HIDDEN_SIZE).to(DEVICE)
model = Seq2Seq(enc, dec)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# ---------------------------------------------------------------------------- #
# Training + Validation loops
# ---------------------------------------------------------------------------- #
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    train_corr = 0
    train_total = 0

    for src_batch, tgt_batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        opt.zero_grad()
        preds = model(src_batch, tgt_batch, TEACHER_FORCING_RATIO)
        # reshape
        B, L, V = preds.size()
        preds_flat = preds[:,1:,:].reshape(-1, V)
        tgt_flat   = tgt_batch[:,1:].reshape(-1)
        loss = crit(preds_flat, tgt_flat)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        # compute train acc
        with torch.no_grad():
            pred_ids = preds.argmax(-1)            # (B,L)
            mask     = tgt_batch != PAD_IDX        # ignore pads
            correct  = (pred_ids == tgt_batch) & mask
            train_corr += correct.sum().item()
            train_total += mask.sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_corr / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_corr = 0
    val_total = 0

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dev_loader, desc="Validating"):
            preds = model(src_batch, tgt_batch, tf_ratio=0)  # no teacher forcing
            B, L, V = preds.size()
            preds_flat = preds[:,1:,:].reshape(-1, V)
            tgt_flat   = tgt_batch[:,1:].reshape(-1)
            loss = crit(preds_flat, tgt_flat)
            val_loss += loss.item()

            pred_ids = preds.argmax(-1)
            mask     = tgt_batch != PAD_IDX
            correct  = (pred_ids == tgt_batch) & mask
            val_corr += correct.sum().item()
            val_total += mask.sum().item()

    avg_val_loss = val_loss / len(dev_loader)
    val_acc = val_corr / val_total

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
