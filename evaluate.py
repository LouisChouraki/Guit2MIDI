import os
import sys
from collections import defaultdict
import torch
import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
import csv
import jams
from autoregressive.models import AR_Transcriber
from autoregressive.constants import *

eps = sys.float_info.epsilon

def write_to_csv(model_name, metrics, eval_loss):
    with open("saved_model/" + model_name + ".csv", "w", encoding='UTF8') as f:
        writer = csv.writer(f, delimiter=",")
        # write a row to the csv file
        writer.writerow(['Loss: ', str(eval_loss)])
        writer.writerow([' ','Precision', 'Recall', 'F1'])
        writer.writerow(['Onset', str(metrics[0]), str(metrics[1]), str(metrics[2])])
        writer.writerow(['Offset', str(metrics[3]), str(metrics[4]), str(metrics[5])])
        writer.writerow(['Frame', '/', '/', str(metrics[6])])
    return

def get_fold(fold):
    files = []
    for file in os.listdir("data/spec_repr/m_2_2_256/"):
        if file.split("_")[0] == fold:
            files.append(file)
    return files


def extract_ref(data):
    pitch = []
    intervals = []

    for i in range(6):
        for note in data[i]:
            pitch.append(note.value)
            intervals.append([note.time, note.time + note.duration])

    return np.array(pitch), np.array(intervals)


def extract_pred(data):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    pitches = []
    intervals = []
    test = data.cpu().numpy()
    for p in range(data.shape[1]):
        for t in range(data.shape[0]):
            """
            if data[t, p] != 0:
                i = t + 1
                while i < data.shape[0] and data[i, p] != 0:
                    i += 1
                pitches.append(p)
                intervals.append([t, i])
            """
            if data[t, p] in [1, 4]:
                i = t + 1
                while i < data.shape[0] and data[i, p] == 2:
                    i += 1
                if i < data.shape[0] and data[i, p] in [3, 4]:
                    pitches.append(p)
                    intervals.append([t, i])

    return np.array(pitches), np.array(intervals)

def notes_to_frames_ref(data, shape, scaling):
    roll = np.zeros(tuple(shape))
    for i in range(6):
        for note in data[i]:
            pitch = int(round(note.value)) - 40
            onset_idx = int(note.time // scaling)
            dur_idx_length = int(((onset_idx + 1) * scaling + note.duration - note.time) // scaling)
            roll[onset_idx:onset_idx+dur_idx_length+1, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs

def notes_to_frames_pred(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return
    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]
    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs

def evaluate(path, fold, model, symmetry, device, save_path=None):
    metrics = defaultdict(list)
    MIN_MIDI = 40
    files = get_fold(fold)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    total_len = 0
    loss = 0

    for file in files[:10]:
        ref = jams.load("./data/GuitarSet/annotation/" + file[:-3] + "jams").annotations["note_midi"]
        data = np.load(path + file, allow_pickle=True)
        
        input = data["repr"]
        target = torch.from_numpy(data["labels"]).unsqueeze(0).long().to(device)
        with torch.no_grad():
            output = model(torch.from_numpy(input).unsqueeze(0).float().to(device), symmetric=symmetry)
            file_len = target.shape[1]
            total_len += file_len
            for t in range(file_len):
                for note in range(44):
                    loss += criterion(output[:, t, note], target[:, t, note])
            output = output.squeeze()
            output = torch.softmax(output, dim=2)

            tets = target.cpu().numpy()
            output = torch.argmax(output, dim=2)
            test = output.cpu().numpy()

        scaling = HOP_LENGTH / SAMPLE_RATE
        p_ref, i_ref = extract_ref(ref)
        p_est, i_est = extract_pred(output)

        t_ref, f_ref = notes_to_frames_ref(ref, output.shape, scaling)
        t_est, f_est = notes_to_frames_pred(p_est, i_est, output.shape)


        p_ref = np.array([midi_to_hz(midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
        save_pianoroll(label_path, label['onset'], label['frame'])
        pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
        save_pianoroll(pred_path, pred['onset'], pred['frame'])
        midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)
    """
    scores = [np.mean(metrics['metric/note/precision']), np.mean(metrics['metric/note/recall']),
              np.mean(metrics['metric/note/f1']), np.mean(metrics['metric/note-with-offsets/precision']),
              np.mean(metrics['metric/note-with-offsets/recall']), np.mean(metrics['metric/note-with-offsets/f1']),
              np.mean(metrics['metric/frame/f1'])]

    return scores, loss / total_len


if __name__ == '__main__':
    path = "data/spec_repr/m_3_512/"
    fold = "05"
    model_name = "tf_07_3_asymmetric/ckpt_15"
    model = AR_Transcriber(229, 44, 48, 32).cuda()
    model.load_state_dict(torch.load("saved_model/" + model_name + ".pt")["state_dict"])
    model.eval()
    scores, loss = evaluate(path, fold, model)

    print("Note onset: p : %.4f r : %.4f f1 : %.4f" % (scores[0],
                                                      scores[1],
                                                      scores[2]))
    print("Note with offset: p : %.4f r : %.4f f1 : %.4f" % (scores[3],
                                                            scores[4],
                                                            scores[5]))
    print("Frame: f1 : %.4f" % (scores[6]))

    print("Loss:" + str(loss.item()))
    write_to_csv(model_name, scores, loss)