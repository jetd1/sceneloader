import argparse
import os
import numpy as np


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
    return parser


def process_seq(seq_meta_path, raw_dir, output_dir):
    metadata = np.load(seq_meta_path, allow_pickle=True)

    vid = metadata['vid']
    video_path = os.path.join(raw_dir, f'{vid}.mp4')
    timestamps = metadata['timestamps']
    intrinsics = metadata['intrinsics']
    poses = metadata['poses']

    os.makedirs(output_dir, exist_ok=True)

    for i, timestamp in enumerate(timestamps):
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill

        frame_path = os.path.join(output_dir, f'{i:04d}.png')

        cmd = f'ffmpeg -y -ss {str_timestamp} -i {video_path} -vframes 1 -f image2 {frame_path}'
        assert os.system(cmd) == 0


if __name__ == '__main__':
    args = create_parser().parse_args()

    raw_video_dir = os.path.join(args.input_dir, 'raw')
    metadata_dir = os.path.join(args.input_dir, 'metadata')
    full_list = open(os.path.join(args.input_dir, 'full_list.txt')).read().splitlines()
    full_list = [f.split(' ')[0] for f in full_list]

    os.makedirs(args.output_dir, exist_ok=True)
    for seq in full_list:
        seq_meta_path = os.path.join(metadata_dir, f'{seq}.npz')
        raw_video_path = os.path.join(raw_video_dir, f'{seq}.mp4')
        process_seq(seq_meta_path, raw_video_dir, os.path.join(args.output_dir, seq))
