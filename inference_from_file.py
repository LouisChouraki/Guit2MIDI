import time
import os
import librosa
import numpy as np
import torch
from autoregressive.models import AR_Transcriber
import mido
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt


def remove_simultaneous_onsets(onsets):
    length = len(onsets)
    i = 0
    while i < length:
        if onsets[i] == 0:
            i += 1
        else:
            if i + 1 < length:
                onsets[i + 1] = 0
            if i + 2 < length:
                onsets[i + 2] = 0
            i += 4

    return onsets

def compute_spec(filename):
    sr, data = wavfile.read(filename)
    data = librosa.util.normalize(data.astype(float))
    data = librosa.resample(data, sr, 22050)
    cqt = np.abs(librosa.cqt(data,
                             hop_length=512,
                             sr=22050,
                             n_bins=192,
                             bins_per_octave=24))
    return cqt


def ninos(data):
    pm = 4
    am = 0
    pa = 3
    aa = 1
    thresh = 0.1
    start_idx = max(pm, pa)
    end_idx = max(am, aa)
    sparsity = np.zeros(len(data))
    max_index = int(0.94 * len(data[0]))
    for i in range(len(data)):
        temp = np.sort(data[i])[:max_index]
        sparsity[i] = (np.sum(temp ** 2)) / ((max_index ** 0.25) * np.sum(temp ** 4) ** 0.25)

    on_set = np.zeros(len(data))
    for i in range(start_idx, len(sparsity) - end_idx, 1):
        if sparsity[i] == np.max(sparsity[i - pm: i + am + 1]) and \
                sparsity[i] > (np.mean(sparsity[i - pa: i + aa + 1]) + thresh):
            on_set[i] = 1
        else:
            on_set[i] = 0
    return on_set


def adjust_onsets(midi_notes, onsets, mode):

    new_midi_notes = []
    new_onsets = []

    current = []
    cnt = []

    return midi_notes, new_onsets

def plot_pianoroll(midi_notes, onsets):
    onset_plot = np.empty((len(onsets), 6))
    for i, notes in enumerate(midi_notes):
        j = -1
        if len(notes) > 6:
            print()

        for j, note in enumerate(notes):
            onset_plot[i, j] = note
        for k in range(6 - (j+1)):
            onset_plot[i, j + k + 1] = -1
    plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()

    ax.eventplot(onset_plot, orientation="vertical", lineoffsets=range(len(onsets)), linewidth=4)
    for i, onset in enumerate(onsets):
        if onset:
            ax.axvline(x=i, ymin=0, ymax=1, color='r')
    plt.ylim((5, 25))

def play_midi(all_notes, on_sets):
    different_notes = np.linspace(40, 84, num=44, dtype=int)
    note_on = [None] * 44
    note_off = [None] * 44
    for index, note in enumerate(different_notes):
        note_on[index] = mido.Message('note_on', note=note)
        note_off[index] = mido.Message('note_off', note=note)
    output_port = mido.open_output("IAC Driver Bus 1", virtual=True)

    for i in range(4):
        output_port.send(mido.Message('note_on', note=10, velocity=0))
        time.sleep(0.5)
    prev_midi_notes = [[],[]]
    for i, notes in enumerate(all_notes):
        start = time.time()
        if on_sets[i]:
            for note in notes:
                output_port.send(note_on[note])
                output_port.send(note_off[note])
        prev_midi_notes.pop(0)
        prev_midi_notes.append(notes)
        end = time.time()
        time.sleep(0.023 - (end - start))

    for note in range(len(note_off)):
        output_port.send(note_off[note])

def main(filename, model, load_cqt, PREDICT, mode):
    file_folder = "inference_npy/" + filename

    if PREDICT:
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        cnn = TabCNN()
        cnn.model = build_model(experiment='adapt_06')
        cnn.model.load_weights("/Users/louischouraki/Documents/tab-cnn-master/model/saved/" + model + "/weights.h5")

        if load_cqt:
            cqt = np.load("data/spec_repr/c/" + filename + ".npz")
            cqt = cqt["repr"]
            full_x = np.pad(cqt, [(4, 4), (0, 0)], mode='constant')
        else:
            cqt = compute_cqt("/Users/louischouraki/Documents/rec_guit/" + filename + ".wav")
            cqt = np.swapaxes(cqt, 0, 1)
            full_x = np.pad(cqt, [(4, 4), (0, 0)], mode='constant')

        is_onset = ninos(cqt)
        is_onset = remove_simultaneous_onsets(is_onset)
        all_midi_notes = []
        for frame_idx in range(len(full_x) - 8):
            sample_x = full_x[frame_idx: frame_idx + 9]
            X = np.expand_dims(np.swapaxes(sample_x, 0, 1), [0,-1])
            output = cnn.model.predict(X)
            pitch = np.array(list(map(tab2pitch, output))).squeeze()
            all_midi_notes.append(np.where(pitch)[0])
        np.save(file_folder + "/" + filename + ".npy", all_midi_notes)
        np.save(file_folder + "/" + filename + "_onset.npy", is_onset)

    all_midi_notes = np.load(file_folder + "/" + filename + ".npy", allow_pickle=True)
    is_onset = np.load(file_folder + "/" + filename + "_onset.npy")

    #ninos_notes, ninos_onsets = adjust_onsets(all_midi_notes, is_onset, "ninos_first")
    cnn_notes, cnn_onsets = adjust_onsets(all_midi_notes, is_onset, "cnnv2")
    plot_pianoroll(all_midi_notes, is_onset)
    plot_pianoroll(cnn_notes, cnn_onsets)
    #plot_pianoroll(cnn_notes, cnn_onsets)
    print("Playing")
    play_midi(cnn_notes, cnn_onsets)


if __name__ == "__main__":
    load_cqt = False
    PREDICT = True
    mode = "ninos_first"
    filename = "loop0028"
    model = "best/2"
    main(filename, model, load_cqt, PREDICT, mode)

"""
play_cnt = [cnt+1 for cnt in play_cnt]
            for k, cnt in enumerate(play_cnt):
                if cnt == 10:
                    output_port.send(note_off[current_play[k]])
                    current_play.pop()
                    play_cnt.pop()"""