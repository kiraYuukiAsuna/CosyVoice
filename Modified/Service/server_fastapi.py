import argparse
import io
import logging
import os
import sys
import traceback

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.getLogger('matplotlib').setLevel(logging.WARNING)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(ROOT_DIR, '..', '..'))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, 'pretrained_models', 'Fun-CosyVoice3-0.5B')

sys.path.append(REPO_ROOT)
sys.path.append(os.path.normpath(os.path.join(REPO_ROOT, 'third_party', 'Matcha-TTS')))

from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

logger = logging.getLogger("cosyvoice_api")
logging.basicConfig(level=logging.INFO)

cosyvoice = None
END_OF_PROMPT = '<|endofprompt|>'
COSYVOICE3_ZERO_SHOT_PREFIX = f'You are a helpful assistant.{END_OF_PROMPT}'
COSYVOICE3_INSTRUCT_PREFIX = 'You are a helpful assistant. '


def is_cosyvoice3():
    return cosyvoice is not None and cosyvoice.__class__.__name__ == 'CosyVoice3'


def normalize_zero_shot_prompt_text(text):
    if not is_cosyvoice3() or text == '':
        return text

    if text.startswith(COSYVOICE3_ZERO_SHOT_PREFIX):
        return text

    if text.endswith(END_OF_PROMPT):
        text = text[:-len(END_OF_PROMPT)]

    if END_OF_PROMPT in text:
        return text

    return COSYVOICE3_ZERO_SHOT_PREFIX + text


def normalize_cross_lingual_text(text):
    if not is_cosyvoice3() or text == '':
        return text

    if text.startswith(COSYVOICE3_ZERO_SHOT_PREFIX):
        return text

    if text.endswith(END_OF_PROMPT):
        text = text[:-len(END_OF_PROMPT)]

    if END_OF_PROMPT in text:
        return text

    return COSYVOICE3_ZERO_SHOT_PREFIX + text


def normalize_instruct_text(text):
    if text == '':
        return text

    if is_cosyvoice3():
        if text.startswith(COSYVOICE3_INSTRUCT_PREFIX) and text.endswith(END_OF_PROMPT):
            return text

        normalized_text = text
        if normalized_text.startswith(COSYVOICE3_ZERO_SHOT_PREFIX):
            normalized_text = normalized_text[len(COSYVOICE3_ZERO_SHOT_PREFIX):]
        elif normalized_text.startswith(COSYVOICE3_INSTRUCT_PREFIX):
            normalized_text = normalized_text[len(COSYVOICE3_INSTRUCT_PREFIX):]

        normalized_text = normalized_text.replace(END_OF_PROMPT, '')
        return f'{COSYVOICE3_INSTRUCT_PREFIX}{normalized_text}{END_OF_PROMPT}'

    if text.endswith(END_OF_PROMPT):
        return text

    return f'{text}{END_OF_PROMPT}'


def process_model_output(model_output):
    """Collect model output into a single PCM16 byte array."""
    buffer = io.BytesIO()

    if hasattr(model_output, '__iter__') and not isinstance(model_output, (list, dict, bytes, bytearray)):
        for item in model_output:
            if isinstance(item, dict) and 'tts_speech' in item:
                speech_data = item['tts_speech']
                if hasattr(speech_data, 'numpy'):
                    buffer.write((speech_data.numpy() * (2 ** 15)).astype(np.int16).tobytes())
            elif hasattr(item, 'numpy'):
                buffer.write((item.numpy() * (2 ** 15)).astype(np.int16).tobytes())
            elif isinstance(item, (bytes, bytearray)):
                buffer.write(item)
    else:
        if isinstance(model_output, dict) and 'tts_speech' in model_output:
            speech_data = model_output['tts_speech']
            if hasattr(speech_data, 'numpy'):
                buffer.write((speech_data.numpy() * (2 ** 15)).astype(np.int16).tobytes())
        elif hasattr(model_output, 'numpy'):
            buffer.write((model_output.numpy() * (2 ** 15)).astype(np.int16).tobytes())
        elif isinstance(model_output, (bytes, bytearray)):
            buffer.write(model_output)

    buffer.seek(0)
    return buffer.read()


def add_wav_header(pcm_data, num_channels=1, bits_per_sample=16):
    data_size = len(pcm_data)
    header = bytearray()

    sample_rate = cosyvoice.sample_rate

    header.extend(b'RIFF')
    header.extend((data_size + 36).to_bytes(4, 'little'))
    header.extend(b'WAVE')

    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))
    header.extend((1).to_bytes(2, 'little'))
    header.extend(num_channels.to_bytes(2, 'little'))
    header.extend(sample_rate.to_bytes(4, 'little'))
    bytes_per_second = sample_rate * num_channels * bits_per_sample // 8
    header.extend(bytes_per_second.to_bytes(4, 'little'))
    block_align = num_channels * bits_per_sample // 8
    header.extend(block_align.to_bytes(2, 'little'))
    header.extend(bits_per_sample.to_bytes(2, 'little'))

    header.extend(b'data')
    header.extend(data_size.to_bytes(4, 'little'))

    return bytes(header) + pcm_data


def build_audio_response(model_output, format_name):
    audio_bytes = process_model_output(model_output)
    normalized_format = format_name.lower()
    audio_data = add_wav_header(audio_bytes) if normalized_format == "wav" else audio_bytes
    content_type = "audio/wav" if normalized_format == "wav" else f"audio/{normalized_format}"

    return Response(
        content=bytes(audio_data) if isinstance(audio_data, bytearray) else audio_data,
        headers={
            "Content-Disposition": f"inline; filename=\"speech.{normalized_format}\"",
            "Content-Length": str(len(audio_data)),
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
        media_type=content_type,
    )


def error_response(message_prefix, exc):
    logger.error(f"{message_prefix}: {str(exc)}")
    logger.error(traceback.format_exc())
    return Response(
        content=f"{message_prefix}: {str(exc)}".encode('utf-8'),
        status_code=500,
        media_type="text/plain",
    )


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    spk_id: str = Query(..., description="说话人ID"),
    format: str = Query("wav", description="音频格式，如 wav、mp3 等")
):
    try:
        model_output = cosyvoice.inference_sft(tts_text, spk_id)
        return build_audio_response(model_output, format)
    except Exception as exc:
        return error_response("处理 TTS 请求时发生错误", exc)


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    prompt_text: str = Query(..., description="提示文本"),
    prompt_wav: UploadFile = File(..., description="提示音频文件"),
    format: str = Query("wav", description="音频格式，如 wav、mp3 等")
):
    try:
        prompt_text = normalize_zero_shot_prompt_text(prompt_text)
        model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav.file)
        return build_audio_response(model_output, format)
    except Exception as exc:
        return error_response("处理 Zero-shot TTS 请求时发生错误", exc)


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    prompt_wav: UploadFile = File(..., description="提示音频文件"),
    format: str = Query("wav", description="音频格式，如 wav、mp3 等")
):
    try:
        tts_text = normalize_cross_lingual_text(tts_text)
        model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_wav.file)
        return build_audio_response(model_output, format)
    except Exception as exc:
        return error_response("处理 Cross-lingual TTS 请求时发生错误", exc)


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    spk_id: str = Query(..., description="说话人ID"),
    instruct_text: str = Query(..., description="指令文本"),
    format: str = Query("wav", description="音频格式，如 wav、mp3 等")
):
    try:
        instruct_text = normalize_instruct_text(instruct_text)
        model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
        return build_audio_response(model_output, format)
    except Exception as exc:
        return error_response("处理 Instruct TTS 请求时发生错误", exc)


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    instruct_text: str = Query(..., description="指令文本"),
    prompt_wav: UploadFile = File(..., description="提示音频文件"),
    format: str = Query("wav", description="音频格式，如 wav、mp3 等")
):
    try:
        instruct_text = normalize_instruct_text(instruct_text)
        model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav.file)
        return build_audio_response(model_output, format)
    except Exception as exc:
        return error_response("处理 Instruct2 TTS 请求时发生错误", exc)


def start_server(host='0.0.0.0', port=50000, model_dir=DEFAULT_MODEL_DIR):
    global cosyvoice
    cosyvoice = AutoModel(model_dir=model_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument(
        '--model_dir',
        type=str,
        default=DEFAULT_MODEL_DIR,
        help='local path or modelscope repo id')
    args = parser.parse_args()

    start_server(port=args.port, model_dir=args.model_dir)
