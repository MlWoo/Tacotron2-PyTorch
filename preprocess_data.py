import argparse
import os
from datasets.preprocessor import get_files, preprocess_data
import json
from utils import _json_object_hook
from multiprocessing import cpu_count


def main():
    parser = argparse.ArgumentParser(description='Tacotron2 Training')
    parser.add_argument('--data_dir', metavar='DIR', default='/home/lynn/dataset/mandarin/sdpz',
                        help='path to dataset')
    parser.add_argument('--base_dir', metavar='DIR', default='/home/lynn/workspace/wumenglin/WaveRNN_pytorch',
                        help='path to dataset')
    parser.add_argument('--output_dir', metavar='DIR', default='dataset_mandarin/',
                        help='path to dataset')
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument("--num_workers", type=int, help='cpu workers to help preprocess')

    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data, object_hook=_json_object_hook)
    audio_config = config.audio_config
    output_path = os.path.join(args.base_dir, args.output_dir)
    mel_path = os.path.join(output_path, 'mels')
    txt_path = os.path.join(output_path, 'txt')
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(txt_path, exist_ok=True)
    num_workers = cpu_count() if config.num_workers is None else int(config.num_workers)

    wav_files = get_files(args.data_dir)
    preprocess_data(wav_files, output_path=output_path, mel_path=mel_path, txt_path=txt_path,
                    audio_config=audio_config, num_workers=num_workers)


if __name__ == '__main__':
    main()
