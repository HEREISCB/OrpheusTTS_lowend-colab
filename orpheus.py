import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import time
import os


model = None

def load_model(model_name="canopylabs/orpheus-tts-0.1-finetune-prod"):
    global model
    # Initialize the Orpheus TTS model according to documentation
    model = OrpheusModel(model_name=model_name)

def generate_speech(prompt, voice, temperature, top_p, repetition_penalty, max_tokens):
    if model is None:
        load_model()
    
    # Start timing
    start_time = time.monotonic()
    
    # Generate speech from the provided text
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens
    )
    
    # Create a unique output filename to avoid overwriting previous generations
    output_filename = f"output_{int(time.time())}.wav"
    
    # Write the audio to a WAV file
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        
        duration = total_frames / wf.getframerate()
    
    # Calculate processing time
    end_time = time.monotonic()
    processing_time = end_time - start_time
    
    # Prepare result message
    result_message = f"Generated {duration:.2f} seconds of audio in {processing_time:.2f} seconds"
    
    return output_filename, result_message

# Create the Gradio interface
with gr.Blocks(title="OrpheusTTS-WebUI") as demo:
    # Use HTML h1 tag for bigger title without hashtag
    gr.Markdown("<div align='center'><h1>OrpheusTTS-WebUI</h1></div>")
    
    # Main description without links
    gr.Markdown("""<div align='center'>Generate realistic speech from text using the OrpheusTTS model.
    
**Available voices:** tara, jess, leo, leah, dan, mia, zac, zoe (in order of conversational realism)
    
**Available emotive tags:** `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
    
**Note:** Increasing repetition_penalty and temperature makes the model speak faster. Increasing Max Tokens extends the maximum duration of genrated audio.
</div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Text input area
            prompt = gr.Textbox(
                label="Text Input", 
                placeholder="Enter text to convert to speech...",
                lines=5
            )
            
            with gr.Row():
                # Voice selection dropdown
                voice = gr.Dropdown(
                    choices=["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"],
                    label="Voice",
                    value="tara"
                )
                
            with gr.Row():
                # Generation parameters
                temperature = gr.Slider(
                    minimum=0.1, 
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P"
                )
                rep_penalty = gr.Slider(
                    minimum=1.1,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Repetition Penalty"
                )
                max_tokens = gr.Slider(
                    minimum=1200,
                    maximum=3600,
                    value=1200,
                    step=100,
                    label="Max Tokens"
                )
            
            # Generate button
            submit_btn = gr.Button("Generate Speech")
            
            # Example prompts
            gr.Examples(
                examples=[
                    "Man, the way social media has, um, completely changed how we interact is just wild, right?",
                    "I just got back from my vacation <sigh> and I'm already feeling stressed about work.",
                    "Did you hear what happened at the party last night? <laugh> It was absolutely ridiculous!",
                    "I've been working on this project for hours <yawn> and I still have so much to do.",
                    "The concert was amazing! <gasp> You should have seen the light show!"
                ],
                inputs=prompt,
                label="Example Prompts"
            )
            
        with gr.Column(scale=1):
            # Audio output
            audio_output = gr.Audio(label="Generated Speech")
            # Generation statistics
            result_text = gr.Textbox(label="Generation Stats", interactive=False)
    
    # Connect the generate_speech function to the interface
    submit_btn.click(
        fn=generate_speech,
        inputs=[prompt, voice, temperature, top_p, rep_penalty, max_tokens],
        outputs=[audio_output, result_text]
    )

    # Clean up function to remove generated audio files (won't work in all deployments)
    def cleanup():
        for file in os.listdir():
            if file.startswith("output_") and file.endswith(".wav"):
                try:
                    os.remove(file)
                except:
                    pass
    
    # Register cleanup for when the interface closes
    demo.load(cleanup)

    # Add footer with links as a separate element at the bottom
    gr.Markdown("""<div align='center' style='margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;'>
<a href="https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained" target="_blank">Hugging Face</a> | 
<a href="https://github.com/Saganaki22/OrpheusTTS-WebUI" target="_blank">WebUI GitHub</a> | 
<a href="https://github.com/canopyai/Orpheus-TTS" target="_blank">Official GitHub</a>
</div>""")

# Only run when orpheus.py is executed directly, not when imported
# Comment this out when using the wrapper
"""
if __name__ == "__main__":
    demo.launch(share=False)  # Set share=False to disable public URL
"""
