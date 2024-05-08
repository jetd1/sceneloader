import os
import random

from utils import get_logger

from multiprocessing import Pool
from pytube import YouTube
import argparse
import logging
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser(description='Download the RealEstate10k dataset.')
    parser.add_argument(
        '--source',
        type=str,
        default='https://s3-haosu.nrp-nautilus.io/jet-public/RealEstate10K.tar.gz',
        help='Local dir to the list / URL to download the list'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='all',
        help='Split to download (train, test, all)',
        choices=['train', 'test', 'all']
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
    parser.add_argument(
        '--max-files', '-m',
        type=int,
        default=99999999,
        required=False,
        help='Max number of files to download'
    )
    return parser


def process_meta(meta):
    with open(meta) as f:
        seq_id = os.path.basename(meta).split('.')[0]

        lines = f.readlines()
        vid = lines[0].rstrip().split('=')[-1]

        lines = lines[1:]
        length = len(lines)
        timestamps = np.zeros(length, dtype=np.int64)
        intrinsics = np.zeros((length, 4), dtype=np.float32)
        poses = np.zeros((length, 12), dtype=np.float32)

        for i, l in enumerate(lines):
            line = l.split(' ')
            timestamps[i] = int(line[0])
            intrinsics[i] = [float(i) for i in line[1:5]]
            poses[i] = [float(i) for i in line[7:19]]

        ret = {
            'seq_id': seq_id,
            'vid': vid,
            'length': length,
            'timestamps': timestamps,
            'intrinsics': intrinsics,
            'poses': poses
        }

        return ret


def download_video(video_url, save_path):
    yt = YouTube(video_url)
    streams = yt.streams

    perfect_streams = streams.filter(adaptive=True, res='720p', file_extension='mp4')
    if perfect_streams:
        stream = perfect_streams.first()
    else:
        logging.getLogger("RE10kD").info(f'No perfect stream for {video_url}, using the first one...')
        stream = streams.first()

    stream.download(filename=save_path, max_retries=5)


def download_workers(args):
    vid, raw_video_dir = args
    v_url = f'https://www.youtube.com/watch?v={vid}'
    save_path = os.path.join(raw_video_dir, f'{vid}.mp4')

    if os.path.exists(save_path):
        logging.getLogger("RE10kD").info(f'Skipping, already downloaded: {save_path}')
        return

    try:
        download_video(v_url, save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        logging.getLogger("RE10kD").error(f'Failed to download {v_url}: {e}')


def download(list_dir, output_dir, num_workers, max_files=99999999):
    metadata = os.listdir(list_dir)
    metadata = [os.path.join(list_dir, x) for x in metadata if x.endswith('.txt')]
    logging.getLogger("RE10kD").info(f'# metadata: {len(metadata)}')

    metadata = [process_meta(x) for i, x in enumerate(metadata) if i < max_files]
    logging.getLogger("RE10kD").info(f'Processed {len(metadata)} metadata...')

    raw_video_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_video_dir, exist_ok=True)

    vids = list(set([x['vid'] for x in metadata]))

    logging.getLogger("RE10kD").info(f'Downloading {len(vids)} videos with {num_workers} workers...')
    args = list(zip(vids, [raw_video_dir] * len(vids)))
    random.shuffle(args)
    with Pool(num_workers) as p:
        p.map(download_workers, args)

    meta_dir = os.path.join(output_dir, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)

    logging.getLogger("RE10kD").info(f'Saving {len(metadata)} metadata...')
    f = open(os.path.join(output_dir, 'full_list.txt'), 'w')
    for meta in metadata:
        if not os.path.exists(os.path.join(raw_video_dir, f'{meta["vid"]}.mp4')):
            logging.getLogger("RE10kD").info(f'Video {meta["vid"]} not downloaded, skipping seq {meta["seq_id"]}...')
            continue
        f.write(f'{meta["seq_id"]} {meta["vid"]}\n')
        np.savez_compressed(os.path.join(meta_dir, f'{meta["seq_id"]}.npz'), **meta)
    f.close()


def main():
    args = create_parser().parse_args()
    logger = get_logger('RE10kD', args.log_file, logging.INFO)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source.startswith('http'):
        logger.info(f'Downloading lists from: {args.source}')

        list_dir = os.path.join(args.output_dir, 'lists')
        os.makedirs(list_dir, exist_ok=True)

        assert os.system(f'wget {args.source} -O {list_dir}/RealEstate10K.tar.gz') == 0
        assert os.system(f'tar -xf {list_dir}/RealEstate10K.tar.gz -C {list_dir}') == 0
        os.remove(f'{list_dir}/RealEstate10K.tar.gz')

        list_dir = os.path.join(list_dir, 'RealEstate10K')
    else:
        list_dir = args.source

    assert os.path.exists(list_dir)

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    for split in splits:
        logger.info(f'Processing split: {split}')
        split_list_dir = os.path.join(list_dir, split)
        split_output_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        download(split_list_dir, split_output_dir, args.num_workers, max_files=args.max_files)

    os.sync()
    logging.shutdown()


if __name__ == '__main__':
    main()
