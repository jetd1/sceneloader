import logging
import os

import numpy as np
import json
import argparse
from PIL import Image

from utils import get_logger
from multiprocessing import Pool


def create_parser():
    parser = argparse.ArgumentParser(description='Export the dataset to adobe json format.')

    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Input directory (should contain full_list.txt metadata/ and images_org/)'
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
    parser.add_argument(
        '--max-seq', '-m',
        type=int,
        default=99999999,
        required=False,
        help='Max number of sequences to include'
    )
    return parser


def export_adobe_json_from_npz(args):
    npz_path, json_path, img_dir = args

    try:
        logging.getLogger("RE10kE_ADOBE").info(f'Processing {npz_path}')
        data = np.load(npz_path)
        seq_id = str(data['seq_id'])

        img_sample = Image.open(os.path.join(img_dir, seq_id, '0000.png'))
        w, h = img_sample.size

        frames = []
        for i in range(data['length']):
            fx, fy, cx, cy = data['intrinsics'][i][:4]
            fx *= w
            fy *= h
            cx *= w
            cy *= h

            pose = data['poses'][i]
            w2c = np.eye(4)
            w2c[:3] = pose.reshape(3, 4)

            frame = {
                'w': w,
                'h': h,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'w2c': w2c.tolist(),
                'file_path': os.path.abspath(os.path.join(img_dir, seq_id, f'{i:04d}.png'))
            }
            frames.append(frame)

        with open(json_path, 'w') as f:
            json.dump({'frames': frames}, f, indent=4)

    except Exception as e:
        logging.getLogger("RE10kE_ADOBE").error(f'Error processing {npz_path}: {e}')


def main():
    args = create_parser().parse_args()
    _ = get_logger("RE10kE_ADOBE", args.log_file, logging.INFO)

    metadata_dir = os.path.join(args.input_dir, 'metadata')
    full_list = open(os.path.join(args.input_dir, 'full_list.txt')).read().splitlines()
    full_list = [f.split(' ')[0] for f in full_list]

    os.makedirs(args.output_dir, exist_ok=True)
    process_seq_args = []
    for seq in full_list:
        npz_path = os.path.join(metadata_dir, f'{seq}.npz')
        json_path = os.path.join(args.output_dir, f'{seq}.json')
        img_dir = os.path.join(args.input_dir, 'images_org')
        process_seq_args.append((npz_path, json_path, img_dir))

    process_seq_args = process_seq_args[:args.max_seq]
    with Pool(args.num_workers) as p:
        p.map(export_adobe_json_from_npz, process_seq_args)

    try:
        os.sync()
        json_list = [os.path.abspath(j[1]) for j in process_seq_args]
        with open(os.path.join(args.output_dir, 'manifest.txt'), 'w') as f:
            for j in json_list:
                if os.path.exists(j):
                    f.write(f'{j}\n')

    except Exception as e:
        logging.getLogger("RE10kE_ADOBE").error(f'Error syncing and writing manifest: {e}')

    finally:
        os.sync()
        logging.shutdown()


if __name__ == '__main__':
    main()
