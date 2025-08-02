import os
import math
import glob
import numpy as np
import torch
import shutil
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from braindecode.models import ShallowFBCSPNet
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

# ====== PARAMETERS ======
eeg_dir = "../../Dataset/EEG_Responses_4_Instuments/responses/"
batch_size = 8
max_epochs = 500
learning_rate = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"
best_ckpt_dir = "./ShallowFBCSPNet/best_ckpt"
latest_ckpt_dir = "./ShallowFBCSPNet/latest_ckpt"
tensorboard_dir = "./ShallowFBCSPNet/tensorboard"
os.makedirs(best_ckpt_dir, exist_ok=True)
os.makedirs(latest_ckpt_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

# ====== DATASET CLASS ======
class SimpleEEGDataset(Dataset):
    def __init__(self, eeg_dir):
        self.fnames = sorted([f for f in os.listdir(eeg_dir) if f.endswith("_response.npy")])
        self.label_strs = [f.split("_")[-2] for f in self.fnames]
        self.label_encoder = LabelEncoder()
        self.label_ints = self.label_encoder.fit_transform(self.label_strs)
        self.eeg_dir = eeg_dir

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.eeg_dir, self.fnames[idx])).astype(np.float32)  # (20, 24320)
        label = self.label_ints[idx]
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)

# ====== LOAD DATA ======
dataset = SimpleEEGDataset(eeg_dir)
n_channels, input_time_length = 20, 24320

# Split train/val (80/20)
full_len = len(dataset)
train_len = int(0.8 * full_len)
val_len = full_len - train_len
train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
n_batches_train = len(train_loader)
print(f"Train set: {len(train_set)} | Val set: {len(val_set)} | n_batches_train: {n_batches_train}")

# ====== MODEL ======
label_encoder = dataset.label_encoder
model = ShallowFBCSPNet(
    n_chans=n_channels,
    n_outputs=len(label_encoder.classes_),
    n_times=input_time_length,
    final_conv_length="auto",  # More flexible than a fixed int
    drop_prob=0.5,
    batch_norm=True,
    pool_mode="mean",
    add_log_softmax=True
).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# ====== CHECKPOINT RESUME LOGIC ======
ckpt_iter = -1
best_val_loss = float("inf")
ckpt_files = [f for f in os.listdir(latest_ckpt_dir) if f.endswith(".pt")]
if ckpt_files:
    latest_ckpt_file = sorted(ckpt_files, key=lambda x: int(x.replace(".pt", "")))[-1]
    ckpt_iter = int(latest_ckpt_file.replace(".pt", ""))
    cut_off_epoch = math.floor(ckpt_iter / n_batches_train)
    cut_off_iter = cut_off_epoch * n_batches_train
    resume_path = os.path.join(latest_ckpt_dir, f"{cut_off_iter}.pt")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    print(f"✅ Resumed from iteration {cut_off_iter}, epoch {cut_off_epoch + 1}")
    # Remove any later checkpoints (incomplete progress)
    for f in os.listdir(latest_ckpt_dir):
        iter_num = int(f.replace(".pt", "")) if f.endswith(".pt") else -1
        if iter_num > cut_off_iter:
            try:
                os.remove(os.path.join(latest_ckpt_dir, f))
            except Exception:
                pass
else:
    cut_off_iter = 0
    cut_off_epoch = 0
    print("🟢 No checkpoint found. Starting fresh.")

cur_iter = cut_off_iter + 1
start_epoch = cut_off_epoch + 1

# ====== TENSORBOARD LOGIC (RESUME/RELOG) ======
old_tb_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
if len(old_tb_files) == 0:
    tb = SummaryWriter(tensorboard_dir)
    last_val_loss = 100.0
elif len(old_tb_files) == 1:
    old_tb_file = old_tb_files[0]
    print(f"Found old TensorBoard file: {old_tb_file}")
    ea = event_accumulator.EventAccumulator(old_tb_file)
    ea.Reload()
    # Recover last val loss
    last_val_loss = None
    if "Val/Loss" in ea.Tags()['scalars']:
        for event in ea.Scalars("Val/Loss"):
            if event.step == cut_off_iter:
                last_val_loss = event.value
                print(f"Recovered Val-Loss at iteration {cut_off_iter}: {last_val_loss:.4f}")
                break
    tb = SummaryWriter(tensorboard_dir)
    for tag in ea.Tags()['scalars']:
        for event in ea.Scalars(tag):
            if event.step <= cut_off_iter:
                tb.add_scalar(tag, event.value, event.step)
                tb.flush()
    os.remove(old_tb_file)
    print("🧹 Deleted old TensorBoard log and resumed with fresh file")
elif len(old_tb_files) > 1:
    raise RuntimeError(f"Expected 0 or 1 TensorBoard log, found {len(old_tb_files)}")
if not 'last_val_loss' in locals():
    last_val_loss = 100.0

# ====== TRAINING ======
for epoch in range(start_epoch, max_epochs + 1):
    print(f"\n========== Epoch {epoch} ==========")
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"\n[Iter {cur_iter}] Train Loss: {loss.item():.4f}")
        tb.add_scalar("Train/Loss", loss.item(), cur_iter)
        tb.flush()
        # ====== VALIDATION (every iteration) ======
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                val_out = model(val_X)
                loss_val = criterion(val_out, val_y)
                val_loss += loss_val.item()
                pred = torch.argmax(val_out, dim=1)
                correct += (pred == val_y).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader)
        print(f"[Iter {cur_iter}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        tb.add_scalar("Val/Loss", val_loss, cur_iter)
        tb.add_scalar("Val/Accuracy", val_acc, cur_iter)
        tb.flush()
        # ====== CHECKPOINTS ======
        # save the latest checkpoint
        latest_ckpt_path = os.path.join(latest_ckpt_dir, f"{cur_iter}.pt")
        ckpt_dict = {
            "cur_iter": cur_iter,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": last_val_loss
        }
        torch.save(ckpt_dict, latest_ckpt_path)

        # Remove old checkpoints before current epoch
        for i in range(cur_iter - n_batches_train):
            old_ckpt = os.path.join(latest_ckpt_dir, f"{i}.pt")
            if os.path.exists(old_ckpt):
                try:
                    os.remove(old_ckpt)
                except Exception:
                    pass
        # save the best checkpoint
        if val_loss < last_val_loss:
            print(f"Validation loss decreased from {last_val_loss:.4f} to {val_loss:.4f}, saving new best checkpoint")
            last_val_loss = val_loss
            checkpoint_name = f"{cur_iter}.pt"

            # Delete previous best ckpt(s) by removing the folder and recreating it
            shutil.rmtree(best_ckpt_dir, ignore_errors=True)
            os.makedirs(best_ckpt_dir, exist_ok=True)

            torch.save({
                "cur_iter": cur_iter,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": last_val_loss
            }, os.path.join(best_ckpt_dir, checkpoint_name))
            print(f"✅ Best model updated at iter {cur_iter}.")

        cur_iter += 1
    print(f"Epoch {epoch} complete.")

tb.close()
print("✅ Training finished.")
