import pickle
import glob
from utils import *
from audio import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            if len(text) < hparams.min_text:
                continue
            futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def preprocess_data(wav_files, output_path, mel_path, quant_path, audio_config):
    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(wav_files):
        id_ = path.split('/')[-1][:-4]
        dataset_ids += [id_]
        rtn = True
        rst = process_utterance(path, audio_config)
        if rst is not None:
            m, x = rst
        else:
            rtn = False

        if rtn is not None:
            np.save(f'{mel_path}/{id_}.npy', m)
            np.save(f'{quant_path}/{id_}.npy', x)
            display('%i/%i', (i + 1, len(wav_files)))

    with open(output_path + 'dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids, f)

def process_utterance(path, audio_config):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
    :param path:
    :param audio_config:
    :return:
    """
    try:
        # Load the audio as numpy array
        wav = load_wav(path, sample_rate=audio_config.sample_rate, encode=False)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(path))
        return None

    wav = wav / np.abs(wav).max() * audio_config.rescaling_max

    # M-AILABS extra silence specific
    if audio_config.trim_silence_enable:
        wav = trim_silence(wav, audio_config.trim_top_db, audio_config.trim_fft_size, audio_config.trim_hop_size)

    quant_scale = 2 ** audio_config.bits
    quant = linear_quantize(wav, quant_scale)
    quant = quant.astype(np.int)
    out = quant
    constant_values = int(linear_quantize(0, quant_scale))

    hop_length = int(audio_config.sample_rate * audio_config.hop_time / 1000)
    win_length = int(audio_config.sample_rate * audio_config.win_time / 1000)

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = melspectrogram(wav, sample_rate=audio_config.sample_rate, n_fft=audio_config.n_fft,
                                     hop_length=hop_length, win_length=win_length, num_mels=audio_config.num_mels,
                                     fmin=audio_config.fmin, min_level_db=audio_config.min_level_db).astype(np.float32)

    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > audio_config.max_mel_frames and audio_config.clip_mels_length_enable:
        return None

    # Ensure time resolution adjustement between audio and mel-spectrogram
    fft_size = win_length
    l, r = pad_lr(wav, fft_size, hop_length)

    # Zero pad for quantized signal
    out_l = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    assert len(out_l) >= mel_frames * hop_length

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out_l[:mel_frames * hop_length]
    assert len(out) % hop_length == 0

    # Return a tuple describing this training example
    return mel_spectrogram, out


def get_files(path, extension='.wav'):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames

