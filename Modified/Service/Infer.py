import io
import os
import sys
import time
import traceback
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RepoRootPath = os.path.normpath(os.path.join(ROOT_DIR, '..', '..'))
sys.path.append(ROOT_DIR)
sys.path.append(RepoRootPath)
sys.path.append(os.path.normpath(os.path.join(ROOT_DIR, '..', '..', 'third_party', 'Matcha-TTS')))
import base64
import argparse
import soundfile as sf
from io import BytesIO
import numpy as np
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2
from ExternalConfig import *
from PythonWorker import PythonWorkerBase

class TTSWorker(PythonWorkerBase):
    """TTS模型Worker示例"""

    cosyvoice2: CosyVoice2

    def initialize(self):
        pretrainedModelPath = os.path.join(
            RepoRootPath, "pretrained_models", "CosyVoice2-0.5B")
        ExternalConfig.WorkingDirectoryRootPath = self.get_global(
            "__WorkingDirectoryRootPath__", ExternalConfig.WorkingDirectoryRootPath)
        ExternalConfig.VoiceSourceModelRootPath = pretrainedModelPath
        ExternalConfig.TTSModelType = "ZeroShot"
        ExternalConfig.TTSModelName = ""

        if self.get_global('__ExternalConfig_VoiceSourceModelRootPath__') is not None:
            print(
                f"Using custom voice source model path: {self.get_global('__ExternalConfig_VoiceSourceModelRootPath__')}")
            ExternalConfig.VoiceSourceModelRootPath = self.get_global(
                '__ExternalConfig_VoiceSourceModelRootPath__')
        if self.get_global('__ExternalConfig_TTSModelType__') is not None:
            print(
                f"Using custom TTS model type: {self.get_global('__ExternalConfig_TTSModelType__')}")
            ExternalConfig.TTSModelType = self.get_global(
                '__ExternalConfig_TTSModelType__')
        if self.get_global('__ExternalConfig_TTSModelName__') is not None:
            print(
                f"Using custom TTS model name: {self.get_global('__ExternalConfig_TTSModelName__')}")
            ExternalConfig.TTSModelName = self.get_global(
                '__ExternalConfig_TTSModelName__')

        if ExternalConfig.VoiceSourceModelRootPath == "":
            ExternalConfig.VoiceSourceModelRootPath = os.path.join(
                ROOT_DIR, "pretrained_models", "CosyVoice2-0.5B")
        if ExternalConfig.TTSModelType == "":
            ExternalConfig.TTSModelType = "ZeroShot"
        if ExternalConfig.TTSModelName == "":
            ExternalConfig.TTSModelName = ""
            ExternalConfig.TTSModelType = "ZeroShot"

        testPath = os.path.join(
            ExternalConfig.VoiceSourceModelRootPath, ExternalConfig.TTSModelName)
        if not os.path.exists(testPath):
            print(
                f"Warning: The specified model path does not exist: {testPath}")
            print("Please check the configuration or download the required model.")
            print("Using default model path instead.")
            ExternalConfig.TTSModelName = ""
            ExternalConfig.TTSModelType = "ZeroShot"

        if ExternalConfig.TTSModelName == "":
            ExternalConfig.VoiceSourceModelRootPath = pretrainedModelPath
            ExternalConfig.TTSModelType = "ZeroShot"

        llmPath = os.path.join(
            ExternalConfig.VoiceSourceModelRootPath, ExternalConfig.TTSModelName, "llm.pt")
        flowPath = os.path.join(
            ExternalConfig.VoiceSourceModelRootPath, ExternalConfig.TTSModelName, "flow.pt")
        spkInfoPath = os.path.join(
            ExternalConfig.VoiceSourceModelRootPath, ExternalConfig.TTSModelName, "spk2info.pt")

        print(f"Using LLM path: {llmPath}")
        print(f"Using Flow path: {flowPath}")
        print(f"Using Speaker Info path: {spkInfoPath}")

        try:
            self.cosyvoice2 = CosyVoice2(
                model_dir=pretrainedModelPath, voiceModelPath=os.path.join(ExternalConfig.VoiceSourceModelRootPath, ExternalConfig.TTSModelName))
        except Exception:
            raise TypeError('no valid model_type!')

    def encode(self, sampling_rate, audio, format):
        with BytesIO() as f:
            if format.upper() == 'OGG':
                sf.write(f, audio, sampling_rate, format="ogg")
                return BytesIO(f.getvalue())
            elif format.upper() == 'MP3':
                sf.write(f, audio, sampling_rate, format="mp3")
                return BytesIO(f.getvalue())
            elif format.upper() == 'WAV':
                sf.write(f, audio, sampling_rate, format="wav")
                return BytesIO(f.getvalue())
            elif format.upper() == 'FLAC':
                sf.write(f, audio, sampling_rate, format="flac")
                return BytesIO(f.getvalue())
            else:
                raise ValueError(f"Unsupported format:{format}")

    def process_model_output(self, model_output):
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

    def InferenceAudioFromText(self, mode, tts_text, spk_id, speed, prompt_text, prompt_audio_path):
        if mode == "ZeroShot":
            prompt_speech_16k = load_wav(prompt_audio_path, 16000)
            print("......开始生成音频......")
            t1 = time.time()
            model_output = self.cosyvoice2.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k)
            audio_bytes = self.process_model_output(model_output)
            t2 = time.time()

            # 根据 PCM16（每采样2字节，默认单声道）计算音频时长
            audio_length = (len(audio_bytes) / 2) / self.cosyvoice2.sample_rate
            print(f"生成音频完成，耗时 {t2 - t1:.2f} 秒，音频时长 {audio_length:.2f} 秒")
            rtf = (t2 - t1) / audio_length
            print(f"RTF: {rtf:.4f}")
            audios = self.encode(self.cosyvoice2.sample_rate, np.frombuffer(
                audio_bytes, dtype=np.int16), format="WAV")
            return audios, t1, t2
        elif mode == "SFT":
            print("......开始生成音频......")
            t1 = time.time()
            model_output = self.cosyvoice2.inference_sft(
                tts_text, spk_id, speed)
            audio_bytes = self.process_model_output(model_output)
            t2 = time.time()

            # 根据 PCM16（每采样2字节，默认单声道）计算音频时长
            audio_length = (len(audio_bytes) / 2) / self.cosyvoice2.sample_rate
            print(f"生成音频完成，耗时 {t2 - t1:.2f} 秒，音频时长 {audio_length:.2f} 秒")
            rtf = (t2 - t1) / audio_length
            print(f"RTF: {rtf:.4f}")
            audios = self.encode(self.cosyvoice2.sample_rate, np.frombuffer(
                audio_bytes, dtype=np.int16), format="WAV")
            return audios, t1, t2
        else:
            return None, None, None

    def save_bytesio_to_file(self, bytesio, path):
        with open(path, 'wb') as f:
            f.write(bytesio.getvalue())


    def exposed_GenerateTTSAudio(self, mode, tts_text, spk_id, speed, prompt_text, prompt_audio_path) -> str:
        audio_bytesIO, t1, t2 = self.InferenceAudioFromText(
            mode, tts_text, spk_id, speed, prompt_text, prompt_audio_path)
        # 将音频字节转换为Base64字符串以便传输
        audio_base64 = base64.b64encode(audio_bytesIO.getvalue()).decode('utf-8')
        return audio_base64


if __name__ == "__main__":
    # audio, t1, t2 = InferenceAudioFromText("SFT", "你好！新的一天，从一场美妙的邂逅开始♪ 你有什么想和我说的吗？", "1", 1.0, "", "")
    # save_bytesio_to_file(audio, "output_sft.wav")
    # audio, t1, t2 = InferenceAudioFromText("ZeroShot", "你好！新的一天，从一场美妙的邂逅开始♪ 你有什么想和我说的吗？", "1", 1.0, "「闲人闲来赏花草，忙人忙得不得了…」", R"D:\Dataset\处理后文件\BertVits2TrainDataset\原神\胡桃\AudioRaw\vo_BZLQ001_6_hutao_01.wav")
    # save_bytesio_to_file(audio, "output_sft.wav")

    # worker = TTSWorker()
    # worker.exposed_GenerateTTSAudio("SFT", "测试一下这个TTS服务是否正常工作。", "1", 1.0, "", "")


    parser = argparse.ArgumentParser(description='Python Worker Server')
    parser.add_argument('--port', type=int, default=0,
                        help='Port number (0 for auto)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host address')
    args = parser.parse_args()

    worker = TTSWorker(host=args.host, port=args.port)

    try:
        worker.start(blocking=True)
    except KeyboardInterrupt:
        print("\n\n[INFO] Worker stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Worker error: {e}")
        traceback.print_exc()
        sys.exit(1)
