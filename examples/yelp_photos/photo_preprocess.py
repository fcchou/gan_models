"""
Preprocess the yelp photos. Make the photos into squares and down-size to 64 * 64

Example command:
    python -m examples.yelp_photos.photo_preprocess --path-to-photos=./yelp_photos
"""
import argparse
import glob
from multiprocessing.pool import ThreadPool
import os

from PIL import Image
import numpy as np


OUTPUT_SHAPE = (64, 64)


def process_img(fn):
    img = Image.open(fn)
    half_w = img.width / 2
    half_h = img.height / 2
    new_size_half = min(half_w, half_h)
    img_crop = img.crop(
        (
            int(half_w - new_size_half),
            int(half_h - new_size_half),
            int(half_w + new_size_half),
            int(half_h + new_size_half),
        )
    )
    return np.asarray(img_crop.resize(OUTPUT_SHAPE, resample=Image.HAMMING))


def main():
    parser = argparse.ArgumentParser(description='Preprocess the Yelp photo data')
    parser.add_argument('--path-to-photos', required=True, help='Path to the yelp photos')
    parser.add_argument(
        '--output-path',
        type=str,
        default='processed_photos.npy',
        help='Path for the output file',
    )
    args = parser.parse_args()
    all_photo_files = glob.glob(os.path.join(args.path_to_photos, '*.jpg'))
    assert all_photo_files

    pool = ThreadPool(16)
    outputs = pool.map(process_img, all_photo_files)
    np.save(args.output_path, outputs)


if __name__ == '__main__':
    main()
