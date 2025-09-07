import argparse
from functools import partial
import logging
import glob
import os
import re
from tqdm import tqdm
from cosyvoice.utils.frontend_utils import spell_out_number, split_paragraph, contains_chinese, replace_blank, remove_bracket, replace_corner_mark
import inflect
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

logger = logging.getLogger()

zh_tn_model = ZhNormalizer(
    remove_erhua=False, full_to_half=False, overwrite_cache=True)
en_tn_model = EnNormalizer()
inflect_parser = inflect.engine()

def main():
    wavs = list(glob.glob('{}/*wav'.format(args.src_dir)))

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        txt = wav.replace('.wav', '.normalized.txt')
        if not os.path.exists(txt):
            logger.warning('{} do not exsist'.format(txt))
            continue
        with open(txt) as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())

        text = content.strip()
        if contains_chinese(text):
            text = zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
        else:
            text = en_tn_model.normalize(text)
            text = spell_out_number(text, inflect_parser)

        utt = os.path.basename(wav).replace('.wav', '')
        spk = utt.split('_')[0]
        utt2wav[utt] = wav
        utt2text[utt] = text
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()
