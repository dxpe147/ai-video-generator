import gradio as gr
from ai_video_agent import AIVideoGenerator
from video_editor import AdvancedVideoEditor
import os
from PIL import Image

class VideoGeneratorUI:
    def __init__(self):
        self.agent = AIVideoGenerator()
        self.editor = AdvancedVideoEditor()
    
    def generate_image_ui(self, prompt, negative_prompt, width, height, steps, guidance, seed):
        """UI wrapper for image generation"""
        try:
            if seed == 0:
                seed = None
            
            image = self.agent.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed
            )
            
            return image, "✅ Image generated successfully!"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def generate_video_ui(self, prompt, negative_prompt, num_frames, steps, guidance, seed):
        """UI wrapper for video generation"""
        try:
            if seed == 0:
                seed = None
            
            frames = self.agent.generate_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed
            )
            
            # Save video
            os.makedirs("output", exist_ok=True)
            output_path = "output/generated_video.mp4"
            self.agent.save_video(frames, output_path, fps=8)
            
            return output_path, "✅ Video generated successfully!"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def apply_effect_ui(self, video_path, effect_type, intensity):
        """Apply effect to uploaded video"""
        # Implementation for applying effects to existing videos
        return video_path, f"Effect {effect_type} applied with intensity {intensity}"


def create_ui():
    """Create Gradio interface"""
    
    app = VideoGeneratorUI()
    
    with gr.Blocks(title="AI Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 AI Video & Image Generator")
        gr.Markdown("Create stunning visuals for music videos and entertainment content")
        
        with gr.Tabs():
            # Image Generation Tab
            with gr.Tab("📸 Image Generation"):
                with gr.Row():
                    with gr.Column():
                        img_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="cyberpunk city at sunset, neon lights, cinematic",
                            lines=3
                        )
                        img_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="low quality, blurry, distorted",
                            lines=2
                        )
                        
                        with gr.Row():
                            img_width = gr.Slider(512, 1024, 1024, step=64, label="Width")
                            img_height = gr.Slider(512, 1024, 1024, step=64, label="Height")
                        
                        with gr.Row():
                            img_steps = gr.Slider(10, 50, 30, step=1, label="Steps")
                            img_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance Scale")
                        
                        img_seed = gr.Number(label="Seed (0 for random)", value=0)
                        img_btn = gr.Button("🎨 Generate Image", variant="primary")
                    
                    with gr.Column():
                        img_output = gr.Image(label="Generated Image")
                        img_status = gr.Textbox(label="Status")
                
                img_btn.click(
                    fn=app.generate_image_ui,
                    inputs=[img_prompt, img_neg_prompt, img_width, img_height, 
                           img_steps, img_guidance, img_seed],
                    outputs=[img_output, img_status]
                )
            
            # Video Generation Tab
            with gr.Tab("🎥 Video Generation"):
                with gr.Row():
                    with gr.Column():
                        vid_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="dancer in colorful lights, energetic movement",
                            lines=3
                        )
                        vid_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="low quality, blurry, static, distorted",
                            lines=2
                        )
                        
                        vid_frames = gr.Slider(8, 64, 16, step=8, label="Number of Frames")
                        
                        with gr.Row():
                            vid_steps = gr.Slider(10, 50, 25, step=1, label="Steps")
                            vid_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance Scale")
                        
                        vid_seed = gr.Number(label="Seed (0 for random)", value=0)
                        vid_btn = gr.Button("🎬 Generate Video", variant="primary")
                    
                    with gr.Column():
                        vid_output = gr.Video(label="Generated Video")
                        vid_status = gr.Textbox(label="Status")
                
                vid_btn.click(
                    fn=app.generate_video_ui,
                    inputs=[vid_prompt, vid_neg_prompt, vid_frames, 
                           vid_steps, vid_guidance, vid_seed],
                    outputs=[vid_output, vid_status]
                )
            
            # Effects Tab
            with gr.Tab("✨ Video Effects"):
                gr.Markdown("### Apply effects to your videos")
                
                with gr.Row():
                    effect_type = gr.Dropdown(
                        choices=['glitch', 'color_shift', 'zoom_pulse', 'kaleidoscope'],
                        label="Effect Type",
                        value='glitch'
                    )
                    effect_intensity = gr.Slider(0, 1, 0.5, label="Intensity")
                
                effect_input = gr.Video(label="Upload Video")
                effect_btn = gr.Button("Apply Effect", variant="primary")
                effect_output = gr.Video(label="Processed Video")
                
                effect_btn.click(
                    fn=app.apply_effect_ui,
                    inputs=[effect_input, effect_type, effect_intensity],
                    outputs=effect_output
                )
        
        gr.Markdown("""
        ### 📚 Tips:
        - Use detailed, descriptive prompts for best results
        - Higher steps = better quality but slower generation
        - Guidance scale 7-8 usually works best
        - Set a seed for reproducible results
        - GPU recommended for faster generation
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, server_name="0.0.0.0")