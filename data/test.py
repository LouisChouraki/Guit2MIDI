import matplotlib.pyplot as plt
import numpy as np
import torch
from autoregressive.models import AR_Transcriber

def plot_pianoroll(data, pred=None):

    onset_plot = []
    sustain_plot = []
    reonset_plot = []
    offset_plot = []
    for i, step in enumerate(data):
        temp_onset_plot = []
        temp_sustain_plot = []
        temp_reonset_plot = []
        temp_offset_plot = []
        for j, note in enumerate(step):
            if data[i, j] == 1:
                temp_onset_plot.append(j)
            if data[i, j] == 2:
                temp_sustain_plot.append(j)
            if data[i, j] == 3:
                temp_offset_plot.append(j)
            if data[i, j] == 4:
                temp_reonset_plot.append(j)

        onset_plot.append(temp_onset_plot)
        sustain_plot.append(temp_sustain_plot)
        reonset_plot.append(temp_reonset_plot)
        offset_plot.append(temp_offset_plot)

    # plot
    fig, ax = plt.subplots()

    ax.eventplot(onset_plot, orientation="vertical", lineoffsets=range(len(data)), linewidth=4, colors='g')
    ax.eventplot(sustain_plot, orientation="vertical", lineoffsets=range(len(data)), linewidth=4, colors='b')
    ax.eventplot(reonset_plot, orientation="vertical", lineoffsets=range(len(data)), linewidth=4, colors='y')
    ax.eventplot(offset_plot, orientation="vertical", lineoffsets=range(len(data)), linewidth=4, colors='r')
    plt.show()

def compare_pianoroll(data, output=None):

    tp_plot = []
    fn_plot = []
    fp_plot = []
    for i, step in enumerate(output):
        temp_tp_plot = []
        temp_fn_plot = []
        temp_fp_plot = []
        for j, note in enumerate(step):
            if data[i + 1, j] != 0:
                if output[i, j] != 0:
                    temp_tp_plot.append(j)

                elif output[i, j] == 0:
                    temp_fn_plot.append(j)

            if output[i, j] != 0:
                if data[i + 1, j] == 0:
                    temp_fp_plot.append(j)

        tp_plot.append(temp_tp_plot)
        fn_plot.append(temp_fn_plot)
        fp_plot.append(temp_fp_plot)
    #plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()

    ax.eventplot(tp_plot, orientation="vertical", lineoffsets=range(len(output)), linewidth=4, colors='b')
    ax.eventplot(fn_plot, orientation="vertical", lineoffsets=range(len(output)), linewidth=4, colors='g')
    ax.eventplot(fp_plot, orientation="vertical", lineoffsets=range(len(output)), linewidth=4, colors='r')
    #plt.show()

data = np.load("spec_repr/m/05_SS2-88-F_solo.npz", allow_pickle=True)
model = AR_Transcriber(229, 44, 48, 32).cuda()
model.load_state_dict(torch.load("../saved_model/tf_05/ckpt_30.pt")["state_dict"])
model.eval()
output = model(torch.from_numpy(data["repr"]).unsqueeze(0).float().cuda())
output = torch.softmax(output, dim=3)
output = torch.argmax(output, dim=3)
test = output.cpu().numpy()
compare_pianoroll(data["labels"], output.squeeze())
plot_pianoroll(output.squeeze())
plot_pianoroll(data["labels"])
