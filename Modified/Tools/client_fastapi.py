import argparse
import logging
import os

import requests


def save_response(response, output_path):
    response.raise_for_status()
    with open(output_path, 'wb') as output_file:
        output_file.write(response.content)
    logging.info('save response to %s', output_path)


def post_request(url, params, prompt_wav=None):
    if prompt_wav is None:
        return requests.post(url, params=params)

    with open(prompt_wav, 'rb') as prompt_file:
        files = {
            'prompt_wav': ('prompt_wav', prompt_file, 'application/octet-stream')
        }
        return requests.post(url, params=params, files=files)


def main():
    url = "http://{}:{}/inference_{}".format(args.host, args.port, args.mode)

    if args.mode == 'sft':
        response = post_request(url, {
            'tts_text': args.tts_text,
            'spk_id': args.spk_id,
            'format': args.format
        })
    elif args.mode == 'zero_shot':
        response = post_request(url, {
            'tts_text': args.tts_text,
            'prompt_text': args.prompt_text,
            'format': args.format
        }, prompt_wav=args.prompt_wav)
    elif args.mode == 'cross_lingual':
        response = post_request(url, {
            'tts_text': args.tts_text,
            'format': args.format
        }, prompt_wav=args.prompt_wav)
    elif args.mode == 'instruct':
        response = post_request(url, {
            'tts_text': args.tts_text,
            'spk_id': args.spk_id,
            'instruct_text': args.instruct_text,
            'format': args.format
        })
    else:
        response = post_request(url, {
            'tts_text': args.tts_text,
            'instruct_text': args.instruct_text,
            'format': args.format
        }, prompt_wav=args.prompt_wav)

    save_response(response, args.output)
    logging.info('get response')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument(
        '--mode',
        default='sft',
        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct', 'instruct2'],
        help='request mode')
    parser.add_argument(
        '--tts_text',
        type=str,
        default='你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？')
    parser.add_argument('--spk_id', type=str, default='中文女')
    parser.add_argument('--prompt_text', type=str, default='希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav', type=str, default='../../../asset/zero_shot_prompt.wav')
    parser.add_argument(
        '--instruct_text',
        type=str,
        default='Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--format', type=str, default='wav', choices=['wav', 'mp3', 'flac', 'ogg'])
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    if args.output == '':
        args.output = os.path.abspath('demo.{}'.format(args.format))

    main()
