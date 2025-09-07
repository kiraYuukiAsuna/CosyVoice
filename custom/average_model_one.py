# Copyright (c) 2020 Mobvoi Inc (Di Wu)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import glob

import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    path = args.src_path
    print('Processing {}'.format(path))
    states = torch.load(path, map_location=torch.device('cpu'))
    avg = {}
    for k in states.keys():
        if k not in ['step', 'epoch']:
            if k not in avg.keys():
                avg[k] = states[k].clone()
            else:
                avg[k] += states[k]

    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == '__main__':
    main()
