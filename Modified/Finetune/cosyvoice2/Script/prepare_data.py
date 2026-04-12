import argparse
import glob
import logging
import os
import re

logger = logging.getLogger(__name__)
NORMALIZER_AVAILABLE = False

try:
    from cosyvoice.utils.frontend_utils import (
        contains_chinese,
        replace_blank,
        replace_corner_mark,
        remove_bracket,
        spell_out_number,
    )
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    import inflect

    zh_tn_model = ZhNormalizer(
        remove_erhua=False, full_to_half=False, overwrite_cache=True
    )
    en_tn_model = EnNormalizer()
    inflect_parser = inflect.engine()
    NORMALIZER_AVAILABLE = True
except ImportError as exc:
    logger.warning("text normalizer unavailable, fallback to raw text: %s", exc)


def normalize_text(text):
    text = text.strip()
    if not NORMALIZER_AVAILABLE:
        return text

    if contains_chinese(text):
        text = zh_tn_model.normalize(text)
        text = text.replace("\n", "")
        text = replace_blank(text)
        text = replace_corner_mark(text)
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r"[，,、]+$", "。", text)
        return text

    text = en_tn_model.normalize(text)
    return spell_out_number(text, inflect_parser)

def main():
    wavs = sorted(glob.glob(os.path.join(args.src_dir, "*.wav")))

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in wavs:
        txt = wav.replace(".wav", ".normalized.txt")
        if not os.path.exists(txt):
            logger.warning("%s does not exist", txt)
            continue

        with open(txt, "r", encoding="utf-8") as f:
            content = "".join(line.rstrip("\n") for line in f)

        text = normalize_text(content)

        utt = os.path.basename(wav).replace(".wav", "")
        spk = utt.split("_")[0]
        utt2wav[utt] = wav
        utt2text[utt] = text
        utt2spk[utt] = spk
        spk2utt.setdefault(spk, []).append(utt)

    os.makedirs(args.des_dir, exist_ok=True)

    with open(os.path.join(args.des_dir, "wav.scp"), "w", encoding="utf-8") as f:
        for k, v in utt2wav.items():
            f.write(f"{k} {v}\n")
    with open(os.path.join(args.des_dir, "text"), "w", encoding="utf-8") as f:
        for k, v in utt2text.items():
            f.write(f"{k} {v}\n")
    with open(os.path.join(args.des_dir, "utt2spk"), "w", encoding="utf-8") as f:
        for k, v in utt2spk.items():
            f.write(f"{k} {v}\n")
    with open(os.path.join(args.des_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for k, v in spk2utt.items():
            f.write(f"{k} {' '.join(v)}\n")
    if args.instruct:
        with open(os.path.join(args.des_dir, "instruct"), "w", encoding="utf-8") as f:
            for k in utt2text:
                f.write(f"{k} {args.instruct}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--des_dir", type=str, required=True)
    parser.add_argument("--instruct", type=str, default="")
    args = parser.parse_args()
    main()
