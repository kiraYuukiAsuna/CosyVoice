import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Query, Response, UploadFile, Form, File
from fastapi import FastAPI, Query, Response
import traceback
import io
import os
import sys
import argparse
import logging
import uuid
logging.getLogger('matplotlib').setLevel(logging.WARNING)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/..'.format(ROOT_DIR))
sys.path.append('{}/../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 设置日志记录器
logger = logging.getLogger("cosyvoice_api")
logging.basicConfig(level=logging.INFO)


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)
                     ).astype(np.int16).tobytes()
        yield tts_audio


def process_model_output(model_output):
    """收集所有音频数据到一个字节数组中"""
    buffer = io.BytesIO()

    # 处理生成器类型的输出
    if hasattr(model_output, '__iter__') and not isinstance(model_output, (list, dict, bytes, bytearray)):
        for item in model_output:
            if isinstance(item, dict) and 'tts_speech' in item:
                speech_data = item['tts_speech']
                if hasattr(speech_data, 'numpy'):
                    # PCM 16位有符号整数，范围是 [-32768, 32767]
                    tts_audio = (speech_data.numpy() * (2 ** 15)
                                 ).astype(np.int16).tobytes()
                    buffer.write(tts_audio)
            elif hasattr(item, 'numpy'):
                tts_audio = (item.numpy() * (2 ** 15)
                             ).astype(np.int16).tobytes()
                buffer.write(tts_audio)
            elif isinstance(item, (bytes, bytearray)):
                buffer.write(item)
    else:
        # 处理非生成器类型
        if isinstance(model_output, dict) and 'tts_speech' in model_output:
            speech_data = model_output['tts_speech']
            if hasattr(speech_data, 'numpy'):
                tts_audio = (speech_data.numpy() * (2 ** 15)
                             ).astype(np.int16).tobytes()
                buffer.write(tts_audio)
        elif hasattr(model_output, 'numpy'):
            tts_audio = (model_output.numpy() * (2 ** 15)
                         ).astype(np.int16).tobytes()
            buffer.write(tts_audio)
        elif isinstance(model_output, (bytes, bytearray)):
            buffer.write(model_output)

    # 将指针移回开始位置
    buffer.seek(0)
    return buffer.read()


def add_wav_header(pcm_data, num_channels=1, bits_per_sample=16):
    """为PCM数据添加WAV文件头"""
    # 计算数据大小
    data_size = len(pcm_data)

    # WAV文件头
    header = bytearray()

    sample_rate = cosyvoice.sample_rate

    # RIFF header
    header.extend(b'RIFF')
    header.extend((data_size + 36).to_bytes(4, 'little'))  # 文件大小 - 8
    header.extend(b'WAVE')

    # fmt chunk
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))  # fmt chunk大小
    header.extend((1).to_bytes(2, 'little'))   # 音频格式 (1 = PCM)
    header.extend(num_channels.to_bytes(2, 'little'))  # 通道数
    header.extend(sample_rate.to_bytes(4, 'little'))  # 采样率
    bytes_per_second = sample_rate * num_channels * bits_per_sample // 8
    header.extend(bytes_per_second.to_bytes(4, 'little'))  # 每秒字节数
    block_align = num_channels * bits_per_sample // 8
    header.extend(block_align.to_bytes(2, 'little'))  # 块对齐
    header.extend(bits_per_sample.to_bytes(2, 'little'))  # 每个样本的位数

    # data chunk
    header.extend(b'data')
    header.extend(data_size.to_bytes(4, 'little'))  # 数据大小

    # 合并头和数据，返回bytes类型而不是bytearray
    return bytes(header) + pcm_data


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    spk_id: str = Query(..., description="说话人ID"),
    format: str = Query("wav", description="音频格式，如wav, mp3等")
):
    try:
        print(
            f"Received TTS request: tts_text={tts_text}, spk_id={spk_id}, format={format}")
        # 调用模型生成音频
        model_output = cosyvoice.inference_sft(
            tts_text, spk_id, stream=False, speed=1, text_frontend=True)

        # 处理模型输出，合并所有音频数据
        audio_bytes = process_model_output(model_output)

        # 如果是WAV格式，需要添加WAV文件头
        if format.lower() == "wav":
            audio_data = add_wav_header(audio_bytes)
        else:
            audio_data = audio_bytes

        # 确定正确的Content-Type
        content_type = "audio/wav" if format.lower(
        ) == "wav" else f"audio/{format}"

        # 构建响应头
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{format}\"",
            "Content-Length": str(len(audio_data)),
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }

        # 明确转换为bytes类型，因为FastAPI的Response期望字节而不是bytearray
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)

        # 返回音频数据
        return Response(
            content=audio_data,
            headers=headers,
            media_type=content_type
        )
    except Exception as e:
        logger.error(f"处理TTS请求时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            content=f"处理TTS请求时发生错误: {str(e)}".encode('utf-8'),
            status_code=500,
            media_type="text/plain"
        )


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    prompt_text: str = Query(..., description="提示文本"),
    prompt_wav: UploadFile = File(..., description="提示音频文件"),
    format: str = Query("wav", description="音频格式，如wav, mp3等")
):
    try:
        print(
            f"Received Zero-shot TTS request: tts_text={tts_text}, prompt_text={prompt_text}, format={format}")
        # 加载提示音频
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)

        # 调用模型生成音频
        model_output = cosyvoice.inference_zero_shot(
            tts_text, prompt_text, prompt_speech_16k)

        # 处理模型输出，合并所有音频数据
        audio_bytes = process_model_output(model_output)

        # 如果是WAV格式，需要添加WAV文件头
        if format.lower() == "wav":
            audio_data = add_wav_header(audio_bytes)
        else:
            audio_data = audio_bytes

        # 确定正确的Content-Type
        content_type = "audio/wav" if format.lower(
        ) == "wav" else f"audio/{format}"

        # 构建响应头
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{format}\"",
            "Content-Length": str(len(audio_data)),
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }

        # 明确转换为bytes类型
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)

        # 返回音频数据
        return Response(
            content=audio_data,
            headers=headers,
            media_type=content_type
        )
    except Exception as e:
        logger.error(f"处理Zero-shot TTS请求时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            content=f"处理Zero-shot TTS请求时发生错误: {str(e)}".encode('utf-8'),
            status_code=500,
            media_type="text/plain"
        )


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    prompt_wav: UploadFile = File(..., description="提示音频文件"),
    format: str = Query("wav", description="音频格式，如wav, mp3等")
):
    try:
        print(
            f"Received Cross-lingual TTS request: tts_text={tts_text}, format={format}")
        # 加载提示音频
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)

        # 调用模型生成音频
        model_output = cosyvoice.inference_cross_lingual(
            tts_text, prompt_speech_16k)

        # 处理模型输出，合并所有音频数据
        audio_bytes = process_model_output(model_output)

        # 如果是WAV格式，需要添加WAV文件头
        if format.lower() == "wav":
            audio_data = add_wav_header(audio_bytes)
        else:
            audio_data = audio_bytes

        # 确定正确的Content-Type
        content_type = "audio/wav" if format.lower(
        ) == "wav" else f"audio/{format}"

        # 构建响应头
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{format}\"",
            "Content-Length": str(len(audio_data)),
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }

        # 明确转换为bytes类型
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)

        # 返回音频数据
        return Response(
            content=audio_data,
            headers=headers,
            media_type=content_type
        )
    except Exception as e:
        logger.error(f"处理Cross-lingual TTS请求时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            content=f"处理Cross-lingual TTS请求时发生错误: {str(e)}".encode('utf-8'),
            status_code=500,
            media_type="text/plain"
        )


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Query(..., description="要转换为语音的文本"),
    instruct_text: str = Query(..., description="指令文本"),
    prompt_wav: UploadFile = File(..., description="提示音频文件"),
    format: str = Query("wav", description="音频格式，如wav, mp3等")
):
    try:
        print(
            f"Received Instruct2 TTS request: tts_text={tts_text}, instruct_text={instruct_text}, format={format}")
        # 加载提示音频
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)

        # 调用模型生成音频
        model_output = cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_speech_16k)

        # 处理模型输出，合并所有音频数据
        audio_bytes = process_model_output(model_output)

        # 如果是WAV格式，需要添加WAV文件头
        if format.lower() == "wav":
            audio_data = add_wav_header(audio_bytes)
        else:
            audio_data = audio_bytes

        # 确定正确的Content-Type
        content_type = "audio/wav" if format.lower(
        ) == "wav" else f"audio/{format}"

        # 构建响应头
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{format}\"",
            "Content-Length": str(len(audio_data)),
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }

        # 明确转换为bytes类型
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)

        # 返回音频数据
        return Response(
            content=audio_data,
            headers=headers,
            media_type=content_type
        )
    except Exception as e:
        logger.error(f"处理Instruct2 TTS请求时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            content=f"处理Instruct2 TTS请求时发生错误: {str(e)}".encode('utf-8'),
            status_code=500,
            media_type="text/plain"
        )
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='/home/seele/Dev/DeepLearning/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(
                args.model_dir, load_jit=False, load_trt=False, fp16=False)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
