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
        """
        Initialize the MIDITransformer.

        Args:
            midi_file_dir_path (pathlib.Path): Path to the directory containing MIDI files.
        """
        self.midi_files_path_list = self._load_midi_files(path=midi_file_dir_path)
        self.files_number = len(self.midi_files_path_list)

    def generate_training_dataset(self, save_path):
        """
         Generate a training dataset and save it as a sparse matrix in CSR format.

         Args:
             save_path (pathlib.Path): Path to save the generated dataset.
         """
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
        """
        Load MIDI files from a directory.

        Args:
            path (pathlib.Path): Path to the directory containing MIDI files.

        Returns:
            list: List of MIDI file paths.
        """
        files_list = glob(str(path / '*/*.midi'))
        print(f"Loaded {len(files_list)} MIDI files.")
        return files_list

    def _process_file(self, midi_file_path):
        """
        Process a MIDI file and convert it into a sparse matrix.

        Args:
            midi_file_path (pathlib.Path): Path to the MIDI file.

        Returns:
            sparse.csr_matrix: Sparse matrix representation of the MIDI file.
        """
        midi_file = mido.MidiFile(str(midi_file_path), clip=True)
        result_array_sparse = sparse.csr_matrix(self._mid2array(midi_file), dtype=np.uint8)
        return result_array_sparse

    def _generate_training_dataset_single_thread(self):
        """
        Generate the training dataset using a single thread.

        Returns:
            sparse.csr_matrix: Concatenated sparse matrix representing the training dataset.
        """
        result_list = []
        # use tqdm to display progressbar
        for midi_file_path in tqdm(self.midi_files_path_list, total=self.files_number):
            array_sparse = self._process_file(midi_file_path)
            result_list.append(array_sparse)
        print("Concatenating the results")
        # concatenate the list of arrays into one long array
        result_array_concatenated = sparse.vstack(result_list, format='csr')
        size_gb = result_array_concatenated.data.nbytes / 1024 ** 3
        print(f"Successfully generated training dataset. Dataset file size: {size_gb:.2f}GB")
        return result_array_concatenated

    def _generate_training_dataset_multiprocessing(self):
        """
         Generate the training dataset using multiprocessing.

         mp.Pool creates process pool with the specified number of processes (num_processes),
         which is equal to the number of CPU cores on the system.
         This allows for concurrent execution of multiple tasks, improving performance by utilizing multiple CPU cores.

         The pool.imap_unordered() function is used to apply the _process_file method to each MIDI file path in
         self.midi_files_path_list using the worker processes in the pool.
         It returns an iterator that provides the results in an unordered manner as soon as they become available.
         The list() function is then used to collect these results into a list called result_list.

         Returns:
             sparse.csr_matrix: Concatenated sparse matrix representing the training dataset.
         """
        num_processes = mp.cpu_count()
        print(f'Running Multiprocessing on {num_processes} cores')

        with mp.Pool(processes=num_processes) as pool:
            result_list = list(
                tqdm(pool.imap_unordered(self._process_file, self.midi_files_path_list), total=self.files_number)
            )
        print("Concatenating the results")
        # concatenate the list of arrays into one long array
        result_array_concatenated = sparse.vstack(result_list, format='csr')
        size_gb = result_array_concatenated.data.nbytes / 1024 ** 3
        print(f"Successfully generated training dataset. Dataset file size: {size_gb:.2f}GB")
        return result_array_concatenated

    @staticmethod
    def _msg2dict(msg):
        """
        Convert a MIDI message to a dictionary.

        Args:
            msg (str): MIDI message.

        Returns:
            list: A list containing a dictionary with extracted information and a flag indicating note on/off.
        """
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
        """
        Switch the state of a note in the piano roll.

        Args:
            last_state (list): Previous state of the piano roll.
            note (int): Note ID.
            velocity (int): Note velocity.
            on_ (bool): Flag indicating note on/off.

        Returns:
            list: Updated state of the piano roll.
        """
        # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
        result = [0] * 88 if last_state is None else last_state.copy()
        if 21 <= note <= 108:
            result[note - 21] = velocity if on_ else 0
        return result

    def _get_new_state(self, new_msg, last_state):
        """
         Get the new state of the piano roll based on the new MIDI message.

         Args:
             new_msg (str): New MIDI message.
             last_state (list): Previous state of the piano roll.

         Returns:
             list: New state of the piano roll.
         """
        new_msg, on_ = self._msg2dict(str(new_msg))
        new_state = self._switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'],
                                      on_=on_) if on_ is not None else last_state
        return [new_state, new_msg['time']]

    def _track2seq(self, track):
        """
        Convert a MIDI track to a sequence of piano roll states.

        Args:
            track (mido.MidiTrack): MIDI track.

        Returns:
            list: Sequence of piano roll states.
        """
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
        """
        Convert a MIDI file to a numpy array.

        Args:
            mid (mido.MidiFile): MIDI file.
            min_msg_pct (float): Minimum percentage of messages required in a track to consider it for conversion.

        Returns:
            np.ndarray: Numpy array representation of the MIDI file.
        """
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
        """
          Get the length of a MIDI file in seconds.

          Args:
              mid (mido.MidiFile): MIDI file.

          Returns:
              float: Length of the MIDI file in seconds.
          """
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
