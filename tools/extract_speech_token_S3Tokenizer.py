#!/usr/bin/env python3
# Copyright (c) 2026
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
import argparse
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
from tqdm import tqdm


MODEL_NAME_MAP = {
    "speech_tokenizer_v1.onnx": "speech_tokenizer_v1",
    "speech_tokenizer_v1_25hz.onnx": "speech_tokenizer_v1_25hz",
    "speech_tokenizer_v1.batch.onnx": "speech_tokenizer_v1",
    "speech_tokenizer_v2.onnx": "speech_tokenizer_v2_25hz",
    "speech_tokenizer_v2.batch.onnx": "speech_tokenizer_v2_25hz",
    "speech_tokenizer_v3.onnx": "speech_tokenizer_v3_25hz",
    "speech_tokenizer_v3.batch.onnx": "speech_tokenizer_v3_25hz",
}
TRAIN_SAMPLE_RATE = 24000
TOKENIZER_SAMPLE_RATE = 16000
FRAME_ALIGN_SAMPLES = 960


def load_s3tokenizer():
    try:
        import s3tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `s3tokenizer`. Install it from "
            "https://github.com/xingchensong/S3Tokenizer or `pip install s3tokenizer`."
        ) from exc
    return s3tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Accelerated speech-token extraction using S3Tokenizer."
    )
    parser.add_argument("--dir", required=True, type=str, help="Kaldi-style data dir containing wav.scp")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="S3Tokenizer model name, e.g. speech_tokenizer_v3_25hz",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="",
        help="Optional compatibility arg. If set, infer the S3Tokenizer model name from the ONNX filename.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per process")
    parser.add_argument(
        "--num_thread",
        type=int,
        default=8,
        help="Number of worker threads for audio loading and mel extraction",
    )
    parser.add_argument(
        "--keep_rank_outputs",
        action="store_true",
        help="Keep per-rank temporary pt files after merge",
    )
    parser.add_argument(
        "--no_strict_length_check",
        action="store_false",
        dest="strict_length_check",
        help="Keep mismatched utterances by truncating to the shorter length instead of skipping them.",
    )
    parser.set_defaults(strict_length_check=True)
    return parser.parse_args()


def resolve_model_name(model_name, onnx_path):
    if onnx_path and os.path.isfile(onnx_path):
        # Prefer the local ONNX checkpoint to avoid an unnecessary download.
        return onnx_path
    if model_name:
        return model_name
    if onnx_path:
        basename = os.path.basename(onnx_path)
        if basename in MODEL_NAME_MAP:
            return MODEL_NAME_MAP[basename]
    raise ValueError(
        "Unable to resolve S3Tokenizer model name. Pass --model explicitly or use a known --onnx_path filename."
    )


def read_wav_scp(data_dir):
    wav_scp = Path(data_dir) / "wav.scp"
    utt2wav = []
    with wav_scp.open() as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            utt, wav_path = line.split(maxsplit=1)
            utt2wav.append((utt, wav_path))
    return utt2wav


def resolve_device(device_arg):
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if device_arg == "cuda":
            device_index = int(os.environ.get("LOCAL_RANK", 0))
        else:
            device_index = int(device_arg.split(":", maxsplit=1)[1])
        torch.cuda.set_device(device_index)
        return f"cuda:{device_index}"
    return "cpu"


def maybe_init_dist(device):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1 and dist.is_initialized() is False:
        backend = "nccl" if device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def load_mel(s3tokenizer, item):
    utt, wav_path = item
    try:
        audio, sample_rate = torchaudio.load(wav_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != TRAIN_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sample_rate, TRAIN_SAMPLE_RATE)(audio)
        padded_len = int(math.ceil(audio.size(1) / FRAME_ALIGN_SAMPLES) * FRAME_ALIGN_SAMPLES)
        if padded_len != audio.size(1):
            audio = torch.cat([audio, audio.new_zeros(1, padded_len - audio.size(1))], dim=1)
        expected_code_len = padded_len // FRAME_ALIGN_SAMPLES
        if TRAIN_SAMPLE_RATE != TOKENIZER_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(TRAIN_SAMPLE_RATE, TOKENIZER_SAMPLE_RATE)(audio)
        mel = s3tokenizer.log_mel_spectrogram(audio.squeeze(0))
        return {"utt": utt, "mel": mel, "expected_code_len": expected_code_len, "error": None}
    except Exception as exc:  # pragma: no cover - defensive path for corrupt audio/input
        return {"utt": utt, "mel": None, "expected_code_len": None, "error": str(exc)}


def extract_codes(s3tokenizer, tokenizer, entries, device, batch_size, num_thread, rank, strict_length_check):
    total_batches = math.ceil(len(entries) / batch_size) if entries else 0
    results = {}
    stats = {"total": 0, "passed": 0, "failed": 0}
    executor = ThreadPoolExecutor(max_workers=num_thread) if num_thread > 1 else None
    try:
        progress = tqdm(
            batch_iter(entries, batch_size),
            total=total_batches,
            desc=f"rank{rank}",
            disable=rank != 0,
        )
        for batch in progress:
            if executor is None:
                loaded = [load_mel(s3tokenizer, item) for item in batch]
            else:
                loaded = list(executor.map(lambda item: load_mel(s3tokenizer, item), batch))
            stats["total"] += len(loaded)
            valid = []
            for item in loaded:
                if item["error"] is not None:
                    stats["failed"] += 1
                    logging.warning("Skip %s due to preprocessing error: %s", item["utt"], item["error"])
                    continue
                valid.append(item)
            if len(valid) == 0:
                continue
            utts = [item["utt"] for item in valid]
            mels = [item["mel"] for item in valid]
            expected_code_lens = [item["expected_code_len"] for item in valid]
            mels, mels_lens = s3tokenizer.padding(mels)
            with torch.inference_mode():
                codes, codes_lens = tokenizer.quantize(
                    mels.to(device),
                    mels_lens.to(device),
                )
            codes = codes.cpu()
            codes_lens = codes_lens.cpu()
            for idx, utt in enumerate(utts):
                actual_len = codes_lens[idx].item()
                expected_len = expected_code_lens[idx]
                if actual_len != expected_len:
                    message = (
                        f"Token length mismatch for {utt}: expected {expected_len}, got {actual_len}. "
                        "This would reintroduce token/mel misalignment during flow training."
                    )
                    if strict_length_check:
                        stats["failed"] += 1
                        logging.warning("Skip %s", message)
                        continue
                    logging.warning(message)
                final_len = min(actual_len, expected_len)
                results[utt] = codes[idx, :final_len].tolist()
                stats["passed"] += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return results, stats


def merge_stats(stats, device, world_size):
    stats_tensor = torch.tensor(
        [stats["total"], stats["passed"], stats["failed"]],
        device=device,
        dtype=torch.long,
    )
    if world_size > 1:
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    return {
        "total": stats_tensor[0].item(),
        "passed": stats_tensor[1].item(),
        "failed": stats_tensor[2].item(),
    }


def save_outputs(output_dir, rank_results, rank, world_size, keep_rank_outputs):
    output_dir = Path(output_dir)
    final_path = output_dir / "utt2speech_token.pt"
    if world_size == 1:
        torch.save(rank_results, final_path)
        return

    part_path = output_dir / f"utt2speech_token.rank{rank}.pt"
    torch.save(rank_results, part_path)
    dist.barrier()
    if rank == 0:
        merged = {}
        for idx in range(world_size):
            this_part = output_dir / f"utt2speech_token.rank{idx}.pt"
            merged.update(torch.load(this_part, map_location="cpu"))
        torch.save(merged, final_path)
        if keep_rank_outputs is False:
            for idx in range(world_size):
                (output_dir / f"utt2speech_token.rank{idx}.pt").unlink(missing_ok=True)
    dist.barrier()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    s3tokenizer = load_s3tokenizer()
    model_name = resolve_model_name(args.model, args.onnx_path)
    device = resolve_device(args.device)
    rank, world_size = maybe_init_dist(device)

    entries = read_wav_scp(args.dir)
    entries = entries[rank::world_size]
    logging.info("Loading S3Tokenizer model %s on %s, rank=%s world_size=%s", model_name, device, rank, world_size)
    tokenizer = s3tokenizer.load_model(model_name).to(device)
    if hasattr(tokenizer, "eval"):
        tokenizer.eval()
    if hasattr(tokenizer, "freeze"):
        tokenizer.freeze()

    rank_results, local_stats = extract_codes(
        s3tokenizer=s3tokenizer,
        tokenizer=tokenizer,
        entries=entries,
        device=device,
        batch_size=args.batch_size,
        num_thread=args.num_thread,
        rank=rank,
        strict_length_check=args.strict_length_check,
    )
    save_outputs(args.dir, rank_results, rank, world_size, args.keep_rank_outputs)
    stats = merge_stats(local_stats, device, world_size)
    if rank == 0:
        logging.info("Saved utt2speech_token.pt to %s", args.dir)
        logging.info(
            "Extraction summary: total=%s passed=%s failed=%s",
            stats["total"],
            stats["passed"],
            stats["failed"],
        )


if __name__ == "__main__":
    main()
