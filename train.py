import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim, nn
from autoregressive.models import AR_Transcriber
from autoregressive.constants import *
import time
import math
from evaluate import evaluate

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def save_model(save_name, iter, model_state_dict, optimizer, scheduler):
    checkpoint = {
        'epoch': iter,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    torch.save(checkpoint, "./saved_model/" + save_name + ".pt")

class Train:
    def __init__(self, device):
        self.device = device
        self.n_mels = N_MELS
        self.n_notes = 44
        self.model = AR_Transcriber(N_MELS, self.n_notes, 48, 32, device).to(device)
        self.model_name = "tf_07_1_symmetric/"
        self.mode = ""
        self.ckpt = "5"
        self.data_folder = "data/spec_repr/m_1_512/"
        self.fold = "05"

    def get_pairs(self):

        files = os.listdir(self.data_folder)
        train_pairs = []
        train_len = []
        eval_pairs = []
        for i, file in enumerate(files):
            data = np.load(self.data_folder + file, allow_pickle=True)
            if file.split("_")[0] != self.fold:
                train_len.append(len(data["repr"]))
                train_pairs.append((torch.from_numpy(data["repr"]).float(), torch.from_numpy(data["labels"]).long()))
            else:
                eval_pairs.append((torch.from_numpy(data["repr"]).float(), torch.from_numpy(data["labels"]).long()))
        return train_pairs, np.array(train_len) / np.sum(train_len), eval_pairs

    def eval_model(self):

        metrics, loss = evaluate(self.data_folder, self.fold, self.model, self.device)
        f1on = metrics[2]
        f1off = metrics[5]
        f1f = metrics[6]

        return loss, f1on, f1off, f1f

    def train(self, input_tensor, target_tensor, optimizer, criterion):
        target_length = target_tensor.size(1)
        loss = 0
        optimizer.zero_grad()
        output = self.model(input_tensor, target_tensor, symmetric=True)

        for t in range(target_length):
            for note in range(self.n_notes):
                loss += criterion(output[:, t, note], target_tensor[:, t, note])

        loss.backward()
        optimizer.step()

        return loss.item() / target_length

    def trainIters(self, n_iters, print_every=1000, plot_every=100, save_every=500, learning_rate=6e-4):
        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        if not os.path.exists("./saved_model/" + self.model_name):
            os.makedirs("./saved_model/" + self.model_name)

        batch_size = 16
        train_files, norm_train_files_len, eval_files = self.get_pairs()

        train_files_idx = [np.random.choice(range(len(train_files)), p=norm_train_files_len) # randomly select files in
                           for i in range(n_iters * batch_size)]                                     # dataset according to their length

        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        stepLR = StepLR(optimizer, step_size=2000, gamma=0.98)
        prev_iters = 0
        best_eval_loss = np.inf
        if self.mode == "resume_training":
            ckpt = torch.load("./saved_model/" + self.model_name + "ckpt_" + self.ckpt +".pt")
            self.model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            stepLR.load_state_dict(ckpt["scheduler"])
            prev_iters = ckpt["epoch"]

        for iter in range(1, n_iters + 1):
            n_frames = 360
            n_context = 2
            input_tensor = torch.zeros((batch_size, n_frames + n_context, self.n_mels)) # Because of (3,3) paddding
            target_tensor = torch.zeros((batch_size, n_frames, self.n_notes), dtype=torch.long)
            for i in range(batch_size):
                training_pair = train_files[train_files_idx[i + (iter - 1) * batch_size]]
                idx = np.random.randint(0, training_pair[1].shape[0] - n_frames)
                input_tensor[i] = training_pair[0][idx:idx + n_frames + n_context]
                target_tensor[i] = training_pair[1][idx:idx + n_frames]

            loss = self.train(input_tensor.to(self.device), target_tensor.to(self.device), optimizer, criterion)
            stepLR.step()

            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                self.model.eval()
                eval_loss, f1on, f1off, f1f = self.eval_model()
                self.model.train()
                print('%s (%d %d%%) loss: %.4f eval_loss: %.4f Onset: %.4f Offset: %.4f Frame: %.4f' % (timeSince(start, iter / n_iters),
                                                  iter, iter / n_iters * 100, print_loss_avg, eval_loss, f1on, f1off, f1f))

                save_model(self.model_name + "ckpt_" + str((prev_iters + iter) // print_every), prev_iters + iter, self.model.state_dict(), optimizer, stepLR)

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model = self.model.state_dict()
                    save_model(self.model_name + "best_ckpt_1", prev_iters + iter, best_model, optimizer, stepLR)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        return prev_iters + iter, self.model.state_dict(), optimizer, stepLR


train_module = Train()
Train.trainIters(n_iters=15000, print_every=1000)
