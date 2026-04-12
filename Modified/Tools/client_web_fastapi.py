import os
import tempfile

import gradio as gr
import requests


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50000


def build_api_url(host, port, endpoint):
    normalized_host = str(host).strip()
    normalized_port = int(port)
    return f"http://{normalized_host}:{normalized_port}/{endpoint}"


def save_response_to_temp_file(response, format_name):
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_name}") as temp_file:
        temp_file.write(response.content)
        return temp_file.name


def post_request(host, port, endpoint, params, prompt_wav=None):
    url = build_api_url(host, port, endpoint)
    if prompt_wav is None:
        return requests.post(url, params=params, timeout=300)

    with open(prompt_wav, 'rb') as prompt_file:
        files = {
            'prompt_wav': ('prompt_wav', prompt_file, 'application/octet-stream')
        }
        return requests.post(url, params=params, files=files, timeout=300)


def handle_response(response, format_name):
    try:
        output_path = save_response_to_temp_file(response, format_name)
        return output_path, ""
    except requests.HTTPError:
        error_text = response.text if response.text else f"HTTP {response.status_code}"
        return None, error_text


def inference_sft(host, port, text, speaker_id, format_name):
    try:
        response = post_request(host, port, "inference_sft", {
            "tts_text": text,
            "spk_id": speaker_id,
            "format": format_name
        })
        return handle_response(response, format_name)
    except Exception as exc:
        return None, str(exc)


def inference_zero_shot(host, port, text, prompt_text, prompt_audio, format_name):
    if not prompt_audio:
        return None, "Error: No reference audio provided"

    try:
        response = post_request(host, port, "inference_zero_shot", {
            "tts_text": text,
            "prompt_text": prompt_text,
            "format": format_name
        }, prompt_wav=prompt_audio)
        return handle_response(response, format_name)
    except Exception as exc:
        return None, str(exc)


def inference_cross_lingual(host, port, text, prompt_audio, format_name):
    if not prompt_audio:
        return None, "Error: No reference audio provided"

    try:
        response = post_request(host, port, "inference_cross_lingual", {
            "tts_text": text,
            "format": format_name
        }, prompt_wav=prompt_audio)
        return handle_response(response, format_name)
    except Exception as exc:
        return None, str(exc)


def inference_instruct(host, port, text, speaker_id, instruct_text, format_name):
    try:
        response = post_request(host, port, "inference_instruct", {
            "tts_text": text,
            "spk_id": speaker_id,
            "instruct_text": instruct_text,
            "format": format_name
        })
        return handle_response(response, format_name)
    except Exception as exc:
        return None, str(exc)


def inference_instruct2(host, port, text, instruct_text, prompt_audio, format_name):
    if not prompt_audio:
        return None, "Error: No reference audio provided"

    try:
        response = post_request(host, port, "inference_instruct2", {
            "tts_text": text,
            "instruct_text": instruct_text,
            "format": format_name
        }, prompt_wav=prompt_audio)
        return handle_response(response, format_name)
    except Exception as exc:
        return None, str(exc)


with gr.Blocks(title="CosyVoice TTS Client") as demo:
    gr.Markdown("# CosyVoice TTS Demo")

    with gr.Row():
        server_host = gr.Textbox(label="Server Host", value=DEFAULT_HOST)
        server_port = gr.Number(label="Server Port", value=DEFAULT_PORT, precision=0)

    with gr.Tabs():
        with gr.TabItem("SFT"):
            with gr.Row():
                with gr.Column():
                    sft_text = gr.Textbox(label="Text", lines=5)
                    sft_speaker = gr.Textbox(label="Speaker ID", value="1")
                    sft_format = gr.Dropdown(label="Output Format", choices=["wav", "mp3", "flac", "ogg"], value="wav")
                    sft_submit = gr.Button("Generate Speech")
                with gr.Column():
                    sft_output = gr.Audio(label="Generated Speech")
                    sft_error = gr.Textbox(label="Error", lines=4)

            sft_submit.click(
                fn=inference_sft,
                inputs=[server_host, server_port, sft_text, sft_speaker, sft_format],
                outputs=[sft_output, sft_error]
            )

        with gr.TabItem("Zero-shot"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("参考提示：CosyVoice1/2 直接填写参考音频对应文本；CosyVoice3 无需手动写 `You are a helpful assistant.<|endofprompt|>`，服务端会自动补。")
                    zs_text = gr.Textbox(label="Text", lines=5)
                    zs_prompt_text = gr.Textbox(label="Prompt Text", lines=3)
                    zs_prompt_audio = gr.Audio(label="Reference Audio", type="filepath")
                    zs_format = gr.Dropdown(label="Output Format", choices=["wav", "mp3", "flac", "ogg"], value="wav")
                    zs_submit = gr.Button("Generate Speech")
                with gr.Column():
                    zs_output = gr.Audio(label="Generated Speech")
                    zs_error = gr.Textbox(label="Error", lines=4)

            zs_submit.click(
                fn=inference_zero_shot,
                inputs=[server_host, server_port, zs_text, zs_prompt_text, zs_prompt_audio, zs_format],
                outputs=[zs_output, zs_error]
            )

        with gr.TabItem("Cross-lingual"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("参考提示：CosyVoice3 跨语种文本前缀会由服务端自动补；直接输入目标语言文本即可。")
                    cl_text = gr.Textbox(label="Text", lines=5)
                    cl_prompt_audio = gr.Audio(label="Reference Audio", type="filepath")
                    cl_format = gr.Dropdown(label="Output Format", choices=["wav", "mp3", "flac", "ogg"], value="wav")
                    cl_submit = gr.Button("Generate Speech")
                with gr.Column():
                    cl_output = gr.Audio(label="Generated Speech")
                    cl_error = gr.Textbox(label="Error", lines=4)

            cl_submit.click(
                fn=inference_cross_lingual,
                inputs=[server_host, server_port, cl_text, cl_prompt_audio, cl_format],
                outputs=[cl_output, cl_error]
            )

        with gr.TabItem("Instruct"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("参考提示：CosyVoice1 的 instruct 末尾需要 `<|endofprompt|>`；这里不用手动写，服务端会自动补。CosyVoice3 会自动转成 `You are a helpful assistant. 指令<|endofprompt|>`。")
                    instruct_text = gr.Textbox(label="Text", lines=5)
                    instruct_speaker = gr.Textbox(label="Speaker ID", value="1")
                    instruct_prompt = gr.Textbox(label="Instruction", lines=3)
                    instruct_format = gr.Dropdown(label="Output Format", choices=["wav", "mp3", "flac", "ogg"], value="wav")
                    instruct_submit = gr.Button("Generate Speech")
                with gr.Column():
                    instruct_output = gr.Audio(label="Generated Speech")
                    instruct_error = gr.Textbox(label="Error", lines=4)

            instruct_submit.click(
                fn=inference_instruct,
                inputs=[server_host, server_port, instruct_text, instruct_speaker, instruct_prompt, instruct_format],
                outputs=[instruct_output, instruct_error]
            )

        with gr.TabItem("Instruct2"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("参考提示：CosyVoice2 的 instruct2 末尾需要 `<|endofprompt|>`；这里不用手动写。CosyVoice3 会自动转成 `You are a helpful assistant. 指令<|endofprompt|>`。")
                    instruct2_text = gr.Textbox(label="Text", lines=5)
                    instruct2_prompt = gr.Textbox(label="Instruction", lines=3)
                    instruct2_audio = gr.Audio(label="Reference Audio", type="filepath")
                    instruct2_format = gr.Dropdown(label="Output Format", choices=["wav", "mp3", "flac", "ogg"], value="wav")
                    instruct2_submit = gr.Button("Generate Speech")
                with gr.Column():
                    instruct2_output = gr.Audio(label="Generated Speech")
                    instruct2_error = gr.Textbox(label="Error", lines=4)

            instruct2_submit.click(
                fn=inference_instruct2,
                inputs=[server_host, server_port, instruct2_text, instruct2_prompt, instruct2_audio, instruct2_format],
                outputs=[instruct2_output, instruct2_error]
            )


if __name__ == "__main__":
    launch_host = os.environ.get("COSYVOICE_WEB_HOST", "127.0.0.1")
    launch_port = int(os.environ.get("COSYVOICE_WEB_PORT", "7860"))
    demo.launch(server_name=launch_host, server_port=launch_port, share=False)
