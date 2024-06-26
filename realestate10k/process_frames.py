import argparse
import logging
import os
import random
import shutil
from utils import get_logger

import numpy as np
from multiprocessing import Pool
from subprocess import DEVNULL, STDOUT, check_call


def create_parser():
    parser = argparse.ArgumentParser(description='Process the raw videos in RealEstate10k.')

    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Input directory (should contain raw/ and metadata/)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--num-workers', '-n',
        type=int,
        default=8,
        help='Number of workers to download the dataset'
    )
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default='',
        required=False,
        help='Log file path'
    )
    return parser


def process_seq(args):
    seq_meta_path, raw_dir, output_dir = args
    metadata = np.load(seq_meta_path, allow_pickle=True)

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) == len(metadata['timestamps']):
            logging.getLogger("RE10kP").info(f'Skipping, dir exists: {output_dir}')
            return
        else:
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=False)

    vid = metadata['vid']
    video_path = os.path.join(raw_dir, f'{vid}.mp4')
    timestamps = metadata['timestamps']

    logging.getLogger("RE10kP").info(f'Start processing {args[0]} with {len(timestamps)} frames.')

    for i, timestamp in enumerate(timestamps):
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill

        frame_path = os.path.join(output_dir, f'{i:04d}.png')

        cmd = ['ffmpeg', '-y', '-ss', str_timestamp, '-i', video_path, '-vframes', '1', '-f', 'image2', frame_path]
        try:
            check_call(cmd, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            logging.getLogger("RE10kP").error(f'Error processing {seq_meta_path}: {e}')
            shutil.rmtree(output_dir)
            break


def main():
    args = create_parser().parse_args()
    _ = get_logger('RE10kP', args.log_file, logging.INFO)

    raw_video_dir = os.path.join(args.input_dir, 'raw')
    metadata_dir = os.path.join(args.input_dir, 'metadata')
    full_list = open(os.path.join(args.input_dir, 'full_list.txt')).read().splitlines()
    full_list = [f.split(' ')[0] for f in full_list]

    os.makedirs(args.output_dir, exist_ok=True)
    process_seq_args = []
    for seq in full_list:
        seq_meta_path = os.path.join(metadata_dir, f'{seq}.npz')
        process_seq_args.append((seq_meta_path, raw_video_dir, os.path.join(args.output_dir, seq)))
    random.shuffle(process_seq_args)

    with Pool(args.num_workers) as p:
        p.map(process_seq, process_seq_args)

    os.sync()
    logging.shutdown()


if __name__ == '__main__':
    main()
