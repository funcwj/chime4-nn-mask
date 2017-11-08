#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.7

import argparse
import json
import os
import numpy as np

def run(args):
    with open(args.json_to_split, 'r') as f:
        ori_data = json.load(f)
    items_num = len(ori_data)
    print('number of items in {}: {}'.format(args.json_to_split, items_num))
    assert args.num_of_parts >= 2, "expect to be splited into more than two parts"
    items_per_part = int(items_num / args.num_of_parts)
    basename = os.path.basename(args.json_to_split).split('.')[0]
    for i in range(args.num_of_parts):
        ibeg = i * items_per_part
        iend = items_num if i == args.num_of_parts - 1 else ibeg + items_per_part
        filename = '{}_{}.json'.format(basename, i + 1)
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(ori_data[ibeg: iend], f, indent=4)
            print('dump {} items into {}...'.format(iend - ibeg, filepath))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split json file into several parts')
    parser.add_argument('json_to_split', type=str, help="location of .json file to be splited")
    parser.add_argument('num_of_parts', type=int, help="number of parts to split")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="directory to dump the splited json file")
    args = parser.parse_args()
    run(args)
