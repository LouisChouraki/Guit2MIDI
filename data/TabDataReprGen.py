import os
import numpy as np
import jams
from scipy.io import wavfile
import sys
import librosa
from librosa import display
from tensorflow.keras.utils import to_categorical
from autoregressive.constants import *
import matplotlib.pyplot as plt

class TabDataReprGen:
    
    def __init__(self, mode="c", n_pad=None):
        # file path to the GuitarSet dataset
        path = "GuitarSet/"
        self.path_audio = path + "audio/audio_mic/"
        self.path_anno = path + "annotation/"
        
        # labeling parameters
        self.string_midi_pitches = [40,45,50,55,59,64]
        
        # prepresentation and its labels storage
        self.output = {}

        self.preproc_mode = mode

        # save file path
        self.save_path = "spec_repr/" + self.preproc_mode + "/"

    def load_rep_and_labels_from_raw_file(self, filename):
        file_audio = self.path_audio + filename + "_mic.wav"
        file_anno = self.path_anno + filename + ".jams"
        jam = jams.load(file_anno)
        self.sr_original, data = wavfile.read(file_audio)

        
        # preprocess audio, store in output dict
        data = np.swapaxes(self.preprocess_audio(data), 0, 1)

        # construct labels
        frame_indices = range(len(data) - 4)
        times = librosa.frames_to_time(frame_indices, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        n_frames = len(times)
        # loop over all strings and sample annotations
        labels = np.zeros((n_frames, 44))  # [onset, sustain, offset, reonset, off]
        labels -= 1
        hop_time = HOP_LENGTH / SAMPLE_RATE
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            # replace midi pitch values with fret numbers

            for note in anno:
                pitch = int(round(note.value)) - 40
                onset_idx = int(note.time // hop_time)
                dur_idx_length = int(((onset_idx + 1) * hop_time + note.duration - note.time) // hop_time)
                if pitch >= 0 and dur_idx_length > -1:
                    if labels[onset_idx, pitch] > -1:
                        labels[onset_idx, pitch] = 4
                    else:
                        labels[onset_idx, pitch] = 1

                    if onset_idx + dur_idx_length >= n_frames - 1:
                        dur_idx_length = n_frames - onset_idx - 1
                    for i in range(1, dur_idx_length + 1):
                        labels[onset_idx + i, pitch] = 2

                    if onset_idx + dur_idx_length + 1 <= n_frames - 1:
                        labels[onset_idx + dur_idx_length + 1, pitch] = 3

        for frame_idx in range(labels.shape[0]):
            for pitch_idx in range(labels.shape[1]):
                if labels[frame_idx, pitch_idx] == -1:
                    labels[frame_idx, pitch_idx] = 0

        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.output["repr"] = data
        self.output["labels"] = labels

        np.savez(save_path + filename + ".npz", **self.output)

        print("done: " + filename + ", " + str(len(labels)) + " frames")

    def preprocess_audio(self, data):
        data = data / 32768
        data = librosa.resample(data, self.sr_original, SAMPLE_RATE)

        data = np.concatenate((np.zeros(PADS[0] * HOP_LENGTH), data, np.zeros(WINDOW_LENGTH*2 + (1 + PADS[0]) * HOP_LENGTH)))
        data = librosa.feature.melspectrogram(data, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                              power=1, win_length=WINDOW_LENGTH, center=False, htk=True,
                                              n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)

        data = np.log10(np.clip(data, a_min=1e-7, a_max=None))

        return data

    def save_data(self, filename):
        np.savez(filename, **self.output)
        
    def get_nth_filename(self, n):
        # returns the filename with no extension
        filenames = np.sort(np.array(os.listdir(self.path_anno)))
        filenames = list(filter(lambda x: x[-5:] == ".jams", filenames))
        return filenames[n][:-5]
    
    def load_and_save_repr_nth_file(self, n):
        # filename has no extenstion
        filename = self.get_nth_filename(n)
        self.load_rep_and_labels_from_raw_file(filename)

        
def main(args):
    n = args[0]
    m = args[1]

    gen = TabDataReprGen(mode='m_1_512')
    gen.load_and_save_repr_nth_file(n)
    
if __name__ == "__main__":
    main(args)



    
            
                                
                    
                    
                
                
    
                
                
                
                
