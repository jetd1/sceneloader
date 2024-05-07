import argparse
import logging
import os
import shutil
import sys
import time

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
        help='Input directory (should contain raw/ and metadata/)'
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

    vid = metadata['vid']
    video_path = os.path.join(raw_dir, f'{vid}.mp4')
    timestamps = metadata['timestamps']

    logging.getLogger("RE10kP").info(f'Start processing {args[0]} with {len(timestamps)} frames.')

    os.makedirs(output_dir, exist_ok=True)

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


def get_logger(name, logfile=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s'
    )
    c_handler = logging.StreamHandler(stream=sys.stdout)
    c_handler.setFormatter(log_formatter)
    c_handler.setLevel(log_level)
    logger.addHandler(c_handler)
    if logfile:
        f_handler = logging.FileHandler(logfile)
        f_handler.setFormatter(log_formatter)
        f_handler.setLevel(log_level)
        logger.addHandler(f_handler)
    logger.setLevel(log_level)
    return logger


def main():
    args = create_parser().parse_args()
    log_file = args.log_file if args.log_file else None
    logger = get_logger('RE10kP', log_file, logging.INFO)

    raw_video_dir = os.path.join(args.input_dir, 'raw')
    metadata_dir = os.path.join(args.input_dir, 'metadata')
    full_list = open(os.path.join(args.input_dir, 'full_list.txt')).read().splitlines()
    full_list = [f.split(' ')[0] for f in full_list]

    os.makedirs(args.output_dir, exist_ok=True)

    process_seq_args = []
    for seq in full_list:
        seq_meta_path = os.path.join(metadata_dir, f'{seq}.npz')
        raw_video_path = os.path.join(raw_video_dir, f'{seq}.mp4')
        process_seq_args.append((seq_meta_path, raw_video_dir, os.path.join(args.output_dir, seq)))

    with Pool(args.num_workers) as p:
        p.map(process_seq, process_seq_args)


if __name__ == '__main__':
    main()
