import multiprocessing as mp
import os
import string
from glob import glob

import mido
import numpy as np
from scipy import sparse
from tqdm import tqdm


class MIDITransformer:
    """
    Loads MIDI files and transform into a sparse matrix in CSR format
    Based on: https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    """

    def __init__(self, midi_file_dir_path):
        self.midi_files_path_list = self._load_midi_files(path=midi_file_dir_path)
        self.files_number = len(self.midi_files_path_list)

    def generate_training_dataset(self, save_path):
        if os.path.exists(save_path):
            print(f"File {save_path} exists - skipping Dataset Generation")
        else:
            try:
                result_array_concatenated = self._generate_training_dataset_multiprocessing()
            except Exception as e:
                print(f"Error running with Multiprocessing: {e}\nRunning on single core instead.")
                result_array_concatenated = self._generate_training_dataset_single_thread()
            sparse.save_npz(save_path, result_array_concatenated)
            print(f"Dataset Generated in file {save_path}")

    @staticmethod
    def _load_midi_files(path):
        files_list = glob(str(path / '*/*.midi'))
        print(f"Loaded {len(files_list)} MIDI files.")
        return files_list

    def _process_file(self, midi_file_path):
        midi_file = mido.MidiFile(midi_file_path, clip=True)
        result_array_sparse = sparse.csr_matrix(self._mid2array(midi_file), dtype=np.uint8)
        return result_array_sparse

    def _generate_training_dataset_single_thread(self):
        result_list = []
        for midi_file_path in tqdm(self.midi_files_path_list, total=self.files_number):
            array_sparse = self._process_file(midi_file_path)
            result_list.append(array_sparse)
        print("Concatenating the results")
        result_array_concatenated = sparse.vstack(result_list, format='csr')
        size_gb = result_array_concatenated.data.nbytes / 1024 ** 3
        print(f"Successfully generated training dataset. Dataset file size: {size_gb:.2f}GB")
        return result_array_concatenated

    def _generate_training_dataset_multiprocessing(self):
        num_processes = mp.cpu_count()
        print(f'Running Multiprocessing on {num_processes} cores')
        with mp.Pool(processes=num_processes) as pool:
            result_list = list(
                tqdm(pool.imap_unordered(self._process_file, self.midi_files_path_list), total=self.files_number)
            )
        print("Concatenating the results")
        result_array_concatenated = sparse.vstack(result_list, format='csr')
        size_gb = result_array_concatenated.data.nbytes / 1024 ** 3
        print(f"Successfully generated training dataset. Dataset file size: {size_gb:.2f}GB")
        return result_array_concatenated

    @staticmethod
    def _msg2dict(msg):
        result = dict()
        if 'note_on' in msg:
            on_ = True
        elif 'note_off' in msg:
            on_ = False
        else:
            on_ = None
        result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
            str.maketrans({a: None for a in string.punctuation})))

        if on_ is not None:
            for k in ['note', 'velocity']:
                result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                    str.maketrans({a: None for a in string.punctuation})))
        return [result, on_]

    @staticmethod
    def _switch_note(last_state, note, velocity, on_=True):
        # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
        result = [0] * 88 if last_state is None else last_state.copy()
        if 21 <= note <= 108:
            result[note - 21] = velocity if on_ else 0
        return result

    def _get_new_state(self, new_msg, last_state):
        new_msg, on_ = self._msg2dict(str(new_msg))
        new_state = self._switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'],
                                      on_=on_) if on_ is not None else last_state
        return [new_state, new_msg['time']]

    def _track2seq(self, track):
        # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
        result = []
        last_state, last_time = self._get_new_state(str(track[0]), [0] * 88)
        for i in range(1, len(track)):
            new_state, new_time = self._get_new_state(track[i], last_state)
            if new_time > 0:
                result += [last_state] * new_time
            last_state, last_time = new_state, new_time
        return result

    def _mid2array(self, mid, min_msg_pct=0.1):
        tracks_len = [len(tr) for tr in mid.tracks]
        min_n_msg = max(tracks_len) * min_msg_pct
        # convert each track to nested list
        all_arrays = []
        for i in range(len(mid.tracks)):
            if len(mid.tracks[i]) > min_n_msg:
                ary_i = self._track2seq(mid.tracks[i])
                all_arrays.append(ary_i)
        # make all nested list the same length
        max_len = max([len(ary) for ary in all_arrays])
        for i in range(len(all_arrays)):
            if len(all_arrays[i]) < max_len:
                all_arrays[i] += [[0] * 88] * (max_len - len(all_arrays[i]))
        all_arrays = np.array(all_arrays)
        all_arrays = all_arrays.max(axis=0)
        # trim: remove consecutive 0s in the beginning and at the end
        sums = all_arrays.sum(axis=1)
        ends = np.where(sums > 0)[0]
        return all_arrays[min(ends): max(ends)]

    @staticmethod
    def _get_midi_length_seconds(mid):
        # get the total number of ticks in the MIDI file
        total_ticks = mid.ticks_per_beat * mid.length

        # get the tempo (in microseconds per tick) from the first tempo change event
        tempo = 500000  # default tempo (in case no tempo events are found)
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        # calculate the length of the MIDI file in seconds
        seconds = mido.tick2second(tick=total_ticks, ticks_per_beat=mid.ticks_per_beat, tempo=tempo)

        # print the length of the MIDI file
        return seconds
