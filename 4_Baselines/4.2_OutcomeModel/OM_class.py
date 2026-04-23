import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_mpr(logits, ground_truth):

    # work in float32 on the same device
    scores = logits.detach().float()
    gt = (ground_truth.detach() > 0.5)

    # MPR via double argsort
    ranks = torch.argsort(torch.argsort(scores))
    denom = max(scores.numel() - 1, 1)
    pr = ranks.float() / denom
    mpr = pr[gt].mean().item() if gt.any() else 0.0

    return mpr

class OutcomeModel(nn.Module):
    """
    Model:
        logit = [p_u ; q_i]^T (W0 + x * Delta) q_j
                + alpha_u * b_u
                + alpha_s * b_i
                + alpha_t * b_j

    Inputs:
        u : user index
        i : source item index
        j : target item index
        x : treatment indicator (0 or 1)

    Embeddings and biases are provided externally and frozen.
    """

    def __init__(
        self,
        user_embeddings,   # (n_users, d)
        item_embeddings,   # (n_items, d)
        user_bias,         # (n_users,)
        item_bias,         # (n_items,)
        loss="BCE"
    ):
        super().__init__()

        self.lr = None
        self.weight_decay = None
        self.batch_size = None
        self.epochs = None
        self.use_amp = None
        self.device = None
        self.note = None
        self.timestamp = None

        user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32)
        item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
        user_bias = torch.tensor(user_bias, dtype=torch.float32)
        item_bias = torch.tensor(item_bias, dtype=torch.float32)

        d = user_embeddings.shape[1]
        self.d = d

        # frozen embeddings
        self.user_emb = nn.Embedding.from_pretrained(user_embeddings, freeze=True)
        self.item_emb = nn.Embedding.from_pretrained(item_embeddings, freeze=True)

        # frozen biases
        self.user_bias = nn.Embedding.from_pretrained(user_bias.unsqueeze(1), freeze=True)
        self.item_bias = nn.Embedding.from_pretrained(item_bias.unsqueeze(1), freeze=True)
        
        # loss function
        self.loss_name = loss.upper()
        if self.loss_name not in {"BCE", "MSE", "MAE"}:
            raise ValueError("loss must be one of {'BCE','MSE','MAE'}")

        # learnable matrices
        self.W = nn.Parameter(torch.randn(2 * d, d) * 0.01)
        self.D = nn.Parameter(torch.randn(2 * d, d) * 0.001)

        # bias scalars
        self.alpha_u = nn.Parameter(torch.tensor(1.0))
        self.alpha_i = nn.Parameter(torch.tensor(1.0))
        self.alpha_j = nn.Parameter(torch.tensor(1.0))
        self.alpha_g = nn.Parameter(torch.tensor(0.0))

    def forward(self, u, i, j, x):

        pu = self.user_emb(u)
        bu = self.user_bias(u).squeeze(-1)
        
        qi = self.item_emb(i)
        bi = self.item_bias(i).squeeze(-1)
        
        qj = self.item_emb(j)
        bj = self.item_bias(j).squeeze(-1)

        src = torch.cat([pu, qi], dim=1)

        M = self.W + x[:, None, None] * self.D

        srcW = torch.bmm(src.unsqueeze(1), M).squeeze(1)
        bilinear = (srcW * qj).sum(dim=1)

        logit = (
            bilinear
            + self.alpha_u * bu
            + self.alpha_i * bi
            + self.alpha_j * bj
            + self.alpha_g
        )

        return logit

    def predict_outcome(self, u_list, i, j, x):

        device = next(self.parameters()).device
        N = len(u_list)

        u_tensor = torch.as_tensor(u_list, dtype=torch.long, device=device)

        i_tensor = torch.full((N,), i, dtype=torch.long, device=device)
        j_tensor = torch.full((N,), j, dtype=torch.long, device=device)
        x_tensor = torch.full((N,), x, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = self.forward(u_tensor, i_tensor, j_tensor, x_tensor)

            if self.loss_name == "BCE":
                return torch.sigmoid(logits)
            else:
                return logits

    def predict_outcome_differences(self, u_list, i, j):

        outcome_treated = self.predict_outcome(u_list=u_list, i=i, j=j, x=1)
        outcome_control = self.predict_outcome(u_list=u_list, i=i, j=j, x=0)

        return outcome_treated - outcome_control

    def get_weights(self, label, w_pos, w_neg):
        return torch.where(label == 1, w_pos, w_neg)
        
    def compute_loss(self, logits, label, weights=None, phase='train'):

        if self.loss_name == "BCE":
            loss = F.binary_cross_entropy_with_logits(logits, label, reduction='none')

        elif self.loss_name == "MSE":
            probs = torch.sigmoid(logits)
            loss = F.mse_loss(probs, label, reduction='none')

        elif self.loss_name == "MAE":
            probs = torch.sigmoid(logits)
            loss = F.l1_loss(probs, label, reduction='none')

        loss = (loss * weights).sum() / weights.sum()

        if phase == 'train':
            return loss
        
        pos = label == 1
        neg = label == 0

        if self.loss_name == "BCE":

            with torch.no_grad():

                logits_pos = logits[pos]
                logits_neg = logits[neg]

                loss_pos = (
                    F.softplus(-logits_pos).mean()
                    if logits_pos.numel()
                    else torch.tensor(0., device=logits.device)
                )

                loss_neg = (
                    F.softplus(logits_neg).mean()
                    if logits_neg.numel()
                    else torch.tensor(0., device=logits.device)
                )

        else:

            if self.loss_name == "MSE":
                loss_fn = F.mse_loss
            elif self.loss_name == "MAE":
                loss_fn = F.l1_loss

            with torch.no_grad():
                loss = loss_fn(probs, label)

                loss_pos = (
                    loss_fn(probs[pos], label[pos])
                    if pos.any()
                    else torch.tensor(0., device=logits.device)
                )

                loss_neg = (
                    loss_fn(probs[neg], label[neg])
                    if neg.any()
                    else torch.tensor(0., device=logits.device)
                )

        return (
            loss.item(),
            loss_pos.item(),
            loss_neg.item(),
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        lr=1e-3,
        weight_decay=0.0,
        epochs=5,
        batch_size=1024,
        use_amp=True,
    ):

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_amp = use_amp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        is_cuda = device.type == "cuda"

        amp_ctx = torch.amp.autocast(
            "cuda" if is_cuda else "cpu",
            enabled=(use_amp and is_cuda)
        )

        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and is_cuda))

        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # --- training tensors ---
        u = torch.tensor(df_train["u"].values, dtype=torch.long)
        i = torch.tensor(df_train["i"].values, dtype=torch.long)
        j = torch.tensor(df_train["j"].values, dtype=torch.long)
        x = torch.tensor(df_train["x"].values, dtype=torch.float32)
        y = torch.tensor(df_train["y"].values, dtype=torch.float32)
        n = len(df_train)

        # --- validation tensors ---
        u_val = torch.tensor(df_valid["u"].values, dtype=torch.long)
        i_val = torch.tensor(df_valid["i"].values, dtype=torch.long)
        j_val = torch.tensor(df_valid["j"].values, dtype=torch.long)
        x_val = torch.tensor(df_valid["x"].values, dtype=torch.float32)
        y_val = torch.tensor(df_valid["y"].values, dtype=torch.float32)
        n_val = len(df_valid)

        # --- class weights for diagnostics ---
        n_pos = (y == 1).sum().item()
        n_neg = (y == 0).sum().item()
        N = n_pos + n_neg
        w_pos = N / (2 * max(n_pos, 1))
        w_neg = N / (2 * max(n_neg, 1))
        w_pos = torch.tensor(w_pos, device=device)
        w_neg = torch.tensor(w_neg, device=device)

        print(f"Loss: {self.loss_name}")
        print("Epoch  ||- - - - - - - - Train - - - - - - - -||- - - - - - Validation - - - - - - - || Epoch's | COS θ | Time     ")
        print("Number || Loss   | L-POS   | L-NEG   | MPR    || Loss   | L-POS   | L-NEG   | MPR    || Change  |       | Elapsed  ")
        print("=======||========|=========|=========|========||========|=========|=========|========||=========|=======|==========")

        n_batches = len(df_train) // batch_size + 1
        start_time = time.time()
        for epoch in range(1, epochs + 1):

            old_params = torch.cat([p.data.flatten() for p in self.parameters()])

            self.train()

            perm = torch.randperm(n)

            train_logits = []
            train_labels = []

            batch = 0
            for start in range(0, n, batch_size):
                batch += 1
                print(f"  {epoch:>3}  .... Training: Batch {batch} out of {n_batches} ({batch/n_batches:.2%})       ", end='\r')

                idx = perm[start:start + batch_size]

                u_batch = u[idx].to(device)
                i_batch = i[idx].to(device)
                j_batch = j[idx].to(device)
                x_batch = x[idx].to(device)
                y_batch = y[idx].to(device)

                weights = torch.where(y_batch == 1, w_pos, w_neg)

                optimizer.zero_grad()

                with amp_ctx:
                    logits = self.forward(u_batch, i_batch, j_batch, x_batch)
                    loss = self.compute_loss(logits, y_batch, weights=weights, phase='train')

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_logits.append(logits.detach())
                train_labels.append(y_batch.detach())

            # -------- evaluation --------
            self.eval()

            val_logits = []
            val_labels = []

            with torch.no_grad():

                train_logits = torch.cat(train_logits)
                train_labels = torch.cat(train_labels)
                train_weights = self.get_weights(train_labels, w_pos, w_neg)
                train_loss, train_loss_pos, train_loss_neg = self.compute_loss(
                    logits=train_logits, 
                    label=train_labels, 
                    weights=train_weights, 
                    phase='eval')
                train_mpr = get_mpr(train_logits, train_labels)

                for start in range(0, n_val, batch_size):

                    idx = slice(start, start + batch_size)

                    u_batch = u_val[idx].to(device)
                    i_batch = i_val[idx].to(device)
                    j_batch = j_val[idx].to(device)
                    x_batch = x_val[idx].to(device)
                    y_batch = y_val[idx].to(device)

                    with amp_ctx:
                        logits = self.forward(u_batch, i_batch, j_batch, x_batch)

                    val_logits.append(logits)
                    val_labels.append(y_batch)

            val_logits = torch.cat(val_logits)
            val_labels = torch.cat(val_labels)
            val_weights = self.get_weights(val_labels, w_pos, w_neg)
            val_loss, val_loss_pos, val_loss_neg = self.compute_loss(
                logits=val_logits, 
                label=val_labels, 
                weights=val_weights, 
                phase='eval')
            val_mpr = get_mpr(val_logits, val_labels)

            # ---- diagnostics ----

            elapsed = time.time() - start_time

            # calculate norms of parameters
            new_params = torch.cat([p.data.flatten() for p in self.parameters()])
            step_vec = new_params - old_params
            upd_norm = torch.norm(step_vec).item()

            # cosine similarity between prev_step and current step_vec
            if 'prev_step' in locals():
                cos_sim = (prev_step @ step_vec) / (prev_step.norm() * step_vec.norm() + 1e-12)
                cos_sim = f"{cos_sim:.3f}" if cos_sim > 0 else f"{cos_sim:.2f}"
            else:
                cos_sim = "None "
            prev_step = step_vec.clone()

            print(
                f"{epoch:5d}  "
                f"|| {train_loss:6.4f} | {train_loss_pos:7.4f} | {train_loss_neg:7.4f} | {train_mpr:6.4f} "
                f"|| {val_loss:6.4f} | {val_loss_pos:7.4f} | {val_loss_neg:7.4f} | {val_mpr:6.4f} "
                f"|| {upd_norm:7.4f} | {cos_sim:5} | {elapsed:7.1f}s"
            )
    
    def save(self, path: str, note: str = None):
        """
        Save model parameters and configuration.
        """

        from datetime import datetime

        self.note = note
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        torch.save({
            "state_dict": self.state_dict(),
            "d": self.d,
            "loss_name": self.loss_name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "use_amp": self.use_amp,
            "device": self.device,
            "timestamp": self.timestamp,
            "note": self.note
        }, path)

    def load(self, path: str):
        """
        Load model parameters and configuration.
        """

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.load_state_dict(checkpoint["state_dict"])

        self.loss_name = checkpoint.get("loss_name", None)
        self.lr = checkpoint.get("lr", None)
        self.weight_decay = checkpoint.get("weight_decay", None)
        self.batch_size = checkpoint.get("batch_size", None)
        self.epochs = checkpoint.get("epochs", None)
        self.use_amp = checkpoint.get("use_amp", None)
        self.device = checkpoint.get("device", None)
        self.note = checkpoint.get("note", None)
        self.timestamp = checkpoint.get("timestamp", None)

        print("Loaded OutcomeModel summary:")
        print(f"Model:             OutcomeModel")
        print(f"Embedding dim:     {self.d}")
        print(f"Loss:              {self.loss_name}")
        print(f"Learning rate:     {self.lr}")
        print(f"Weight decay:      {self.weight_decay}")
        print(f"Batch size:        {self.batch_size}")
        print(f"Epochs:            {self.epochs}")
        print(f"Use AMP:           {self.use_amp}")
        print(f"Timestamp:         {self.timestamp}")
        if self.note:
            print(f"Note:              {self.note}")