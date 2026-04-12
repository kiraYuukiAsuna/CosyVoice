import time
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm  # libsora 基于 soundfile，可以直接使用
from cosyvoice.cli.cosyvoice import CosyVoice2
import os
import sys
from cosyvoice.utils.file_utils import load_wav
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))


max_val = 0.8


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(
        1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B',
                       load_jit=False, load_trt=False, fp16=False)

# 保存音频到文件
audio_data = []
idx = 0
# for i in cosyvoice.inference_sft("<|jp|>アキモナカバヲスギテイタ。ヘヤノナカニイルト、サスガニスズシイガ、ヒナタニデルト、ネムクナルホドアタタカイ。ニワノキギハオオカタアカクイロヅキ、ナカニハハヲオトシハジメタモノモアル。", "1", stream=False, speed=1, text_frontend=True):
for i in cosyvoice.inference_sft("""再说，如果只是住在隔壁就能变得亲近的话，那些迷恋她的男生们就不用那么辛苦了。顺带一提，感受到异性魅力未必等同于怀有恋爱感情。""", "1", stream=False, speed=1, text_frontend=True):
    audio_data.append(i['tts_speech'].numpy().flatten())
    # torchaudio.save(f'output_audio_{idx}.wav',
    #                 i['tts_speech'], cosyvoice.sample_rate)
    idx += 1
# prompt_wav = "/home/seele/Dev/DeepLearning/CosyVoice/examples/libritts/cosyvoice2/Dataset/elysia_train/1_1.wav"
# prompt_sr = 16000
# prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))

# for i in cosyvoice.inference_cross_lingual("アキモナカバヲスギテイタ。ヘヤノナカニイルト、サスガニスズシイガ、ヒナタニデルト、ネムクナルホドアタタカイ。ニワノキギハオオカタアカクイロヅキ、ナカニハハヲオトシハジメタモノモアル。", prompt_speech_16k, stream=False, speed=1):
#     audio_data.append(i['tts_speech'].numpy().flatten())

# 将音频拼接成单一的 NumPy 数组
audio_array = np.concatenate(audio_data)

# 保存为 WAV 文件（16-bit PCM 格式）
output_file = "output_audio.wav"
sf.write(output_file, audio_array, samplerate=cosyvoice.sample_rate)

print(f"生成的音频已保存为文件: {output_file}")


# audio_data = []
# prompt_wav = "/home/CosyVoice/examples/libritts/cosyvoice2/cosyvoice2Dataset/elysia_test/1_1069.wav"
# prompt_sr = 16000
# prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
# for i in cosyvoice.inference_zero_shot("所谓的天使当然只是一种比喻，但椎名真昼这名少女就是如此美丽可爱，使得这个比喻就像真的一样。", "这你就问对人啦。他是个非常温柔，又特别有趣的男孩。虽然话不多，心思却格外细腻呢。",
#                                        prompt_speech_16k, stream=False, speed=1, text_frontend=True):
#     audio_data.append(i['tts_speech'].numpy().flatten())

# # for i, j in enumerate(cosyvoice.inference_instruct2("睡觉吧，闭上眼睛，再睁开后就天亮了。", '用悲伤的语气说这句话', prompt_speech_16k, stream=False)):
# #    audio_data.append(j['tts_speech'].numpy().flatten())

# # 将音频拼接成单一的 NumPy 数组
# audio_array = np.concatenate(audio_data)

# # 保存为 WAV 文件（16-bit PCM 格式）
# output_file = "output_audio.wav"
# sf.write(output_file, audio_array, samplerate=cosyvoice.sample_rate)

# print(f"生成的音频已保存为文件: {output_file}")


# 保存音频到文件
# audio_data = []

# prompt_wav = "/home/CosyVoice/examples/libritts/cosyvoice2/cosyvoice2Dataset/elysia_train/1_1.wav"
# prompt_sr = 16000
# prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))

# for i in cosyvoice.inference_cross_lingual("所谓的天使当然只是一种比喻，但椎名真昼这名少女就是如此美丽可爱，使得这个比喻就像真的一样。", prompt_speech_16k, stream=False, speed=1):
#     audio_data.append(i['tts_speech'].numpy().flatten())

# # 将音频拼接成单一的 NumPy 数组
# audio_array = np.concatenate(audio_data)

# # 保存为 WAV 文件（16-bit PCM 格式）
# output_file = "output_audio_abc.wav"
# sf.write(output_file, audio_array, samplerate=cosyvoice.sample_rate)

# print(f"生成的音频已保存为文件: {output_file}")

# max_val = 0.8
