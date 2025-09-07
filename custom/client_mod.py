import gradio as gr
import requests
import numpy as np
import io
import wave
import os
from pydub import AudioSegment
import tempfile

# API endpoint configuration
API_HOST = "http://localhost:50000"  # Update with your actual server address

def get_wav_info(wav_bytes):
    """Extract sample rate and audio data from WAV bytes"""
    with io.BytesIO(wav_bytes) as wav_io:
        with wave.open(wav_io, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
            sample_width = wav_file.getsampwidth()
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit audio
                dtype = np.int16
            elif sample_width == 4:  # 32-bit audio
                dtype = np.int32
            else:
                dtype = np.uint8
                
            audio_data = np.frombuffer(audio_bytes, dtype=dtype)
            
    return sample_rate, audio_data

def inference_sft(text, speaker_id, format_type):
    """Call the SFT inference API"""
    try:
        response = requests.post(
            f"{API_HOST}/inference_sft",
            params={
                "tts_text": text,
                "spk_id": speaker_id,
                "format": format_type
            }
        )
        
        if response.status_code == 200:
            # Save the response to a temp file for gradio to play
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_type}") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            return temp_path, None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def inference_zero_shot(text, prompt_text, prompt_audio, format_type):
    """Call the Zero-shot inference API"""
    try:
        if prompt_audio is None:
            return None, "Error: No reference audio provided"
        
        # Extract audio data from gradio component
        audio_path = prompt_audio
        
        files = {'prompt_wav': open(audio_path, 'rb')}
        
        response = requests.post(
            f"{API_HOST}/inference_zero_shot",
            params={
                "tts_text": text,
                "prompt_text": prompt_text,
                "format": format_type
            },
            files=files
        )
        
        if response.status_code == 200:
            # Save the response to a temp file for gradio to play
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_type}") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            return temp_path, None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def inference_cross_lingual(text, prompt_audio, format_type):
    """Call the Cross-lingual inference API"""
    try:
        if prompt_audio is None:
            return None, "Error: No reference audio provided"
        
        # Extract audio data from gradio component
        audio_path = prompt_audio
        
        files = {'prompt_wav': open(audio_path, 'rb')}
        
        response = requests.post(
            f"{API_HOST}/inference_cross_lingual",
            params={
                "tts_text": text,
                "format": format_type
            },
            files=files
        )
        
        if response.status_code == 200:
            # Save the response to a temp file for gradio to play
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_type}") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            return temp_path, None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def inference_instruct2(text, instruct_text, prompt_audio, format_type):
    """Call the Instruct2 inference API"""
    try:
        if prompt_audio is None:
            return None, "Error: No reference audio provided"
        
        # Extract audio data from gradio component
        audio_path = prompt_audio
        
        files = {'prompt_wav': open(audio_path, 'rb')}
        
        response = requests.post(
            f"{API_HOST}/inference_instruct2",
            params={
                "tts_text": text,
                "instruct_text": instruct_text,
                "format": format_type
            },
            files=files
        )
        
        if response.status_code == 200:
            # Save the response to a temp file for gradio to play
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_type}") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            return temp_path, None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="CosyVoice TTS Client") as demo:
    gr.Markdown("# CosyVoice TTS Demo")
    
    with gr.Tabs():
        # SFT Tab
        with gr.TabItem("Standard Fine-tuned TTS"):
            with gr.Row():
                with gr.Column():
                    sft_text = gr.Textbox(
                        label="Text to convert to speech",
                        placeholder="Enter text here...",
                        lines=5
                    )
                    sft_speaker = gr.Dropdown(
                        label="Speaker ID",
                        choices=["1"],  # Replace with actual available speakers
                        value="1"
                    )
                    sft_format = gr.Dropdown(
                        label="Output Format",
                        choices=["wav", "mp3"],
                        value="wav"
                    )
                    sft_submit = gr.Button("Generate Speech")
                
                with gr.Column():
                    sft_output = gr.Audio(label="Generated Speech")
                    sft_error = gr.Textbox(label="Error Message", visible=True)
            
            sft_submit.click(
                fn=inference_sft,
                inputs=[sft_text, sft_speaker, sft_format],
                outputs=[sft_output, sft_error]
            )
        
        # Zero-shot Tab
        with gr.TabItem("Zero-shot TTS"):
            with gr.Row():
                with gr.Column():
                    zs_text = gr.Textbox(
                        label="Text to convert to speech",
                        placeholder="Enter text here...",
                        lines=5
                    )
                    zs_prompt_text = gr.Textbox(
                        label="Prompt Text",
                        placeholder="Enter reference text that matches the reference audio...",
                        lines=3
                    )
                    zs_prompt_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath"  # Using filepath instead of binary
                    )
                    zs_format = gr.Dropdown(
                        label="Output Format",
                        choices=["wav", "mp3"],
                        value="wav"
                    )
                    zs_submit = gr.Button("Generate Speech")
                
                with gr.Column():
                    zs_output = gr.Audio(label="Generated Speech")
                    zs_error = gr.Textbox(label="Error Message", visible=True)
            
            zs_submit.click(
                fn=inference_zero_shot,
                inputs=[zs_text, zs_prompt_text, zs_prompt_audio, zs_format],
                outputs=[zs_output, zs_error]
            )
        
        # Cross-lingual Tab
        with gr.TabItem("Cross-lingual TTS"):
            with gr.Row():
                with gr.Column():
                    cl_text = gr.Textbox(
                        label="Text to convert to speech (any language)",
                        placeholder="Enter text here in any language...",
                        lines=5
                    )
                    cl_prompt_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath"  # Using filepath instead of binary
                    )
                    cl_format = gr.Dropdown(
                        label="Output Format",
                        choices=["wav", "mp3"],
                        value="wav"
                    )
                    cl_submit = gr.Button("Generate Speech")
                
                with gr.Column():
                    cl_output = gr.Audio(label="Generated Speech")
                    cl_error = gr.Textbox(label="Error Message", visible=True)
            
            cl_submit.click(
                fn=inference_cross_lingual,
                inputs=[cl_text, cl_prompt_audio, cl_format],
                outputs=[cl_output, cl_error]
            )
        
        # Instruct2 Tab
        with gr.TabItem("Instruct2 TTS"):
            with gr.Row():
                with gr.Column():
                    ins_text = gr.Textbox(
                        label="Text to convert to speech",
                        placeholder="Enter text here...",
                        lines=5
                    )
                    ins_instruct = gr.Textbox(
                        label="Instruction",
                        placeholder="E.g., speak with a happy tone, speak fast, etc.",
                        lines=2
                    )
                    ins_prompt_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath"  # Using filepath instead of binary
                    )
                    ins_format = gr.Dropdown(
                        label="Output Format",
                        choices=["wav", "mp3"],
                        value="wav"
                    )
                    ins_submit = gr.Button("Generate Speech")
                
                with gr.Column():
                    ins_output = gr.Audio(label="Generated Speech")
                    ins_error = gr.Textbox(label="Error Message", visible=True)
            
            ins_submit.click(
                fn=inference_instruct2,
                inputs=[ins_text, ins_instruct, ins_prompt_audio, ins_format],
                outputs=[ins_output, ins_error]
            )

    gr.Markdown("""
    ## CosyVoice TTS System
    
    This demo showcases various capabilities of CosyVoice TTS:
    
    - **Standard Fine-tuned TTS**: Generate speech from predefined voice models
    - **Zero-shot TTS**: Clone a voice with just a short audio sample
    - **Cross-lingual TTS**: Generate speech in different languages using a reference voice
    - **Instruct2 TTS**: Control speech attributes with natural language instructions
    
    For more information, visit the [CosyVoice project page](https://github.com/thuhcsi/CosyVoice).
    """)

if __name__ == "__main__":
    demo.launch(share=True)
