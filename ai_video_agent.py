import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
    DPMSolverMultistepScheduler
)
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class AIVideoGenerator:
    """
    Main AI Agent for video and image generation
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"🚀 Initializing AI Video Generator on {device}...")
        
        # Model storage
        self.image_pipe = None
        self.video_pipe = None
        self.depth_pipe = None
        
    def load_image_generator(self, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        """Load Stable Diffusion XL for image generation"""
        print("📸 Loading image generator...")
        
        self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        self.image_pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.image_pipe.enable_model_cpu_offload()
            self.image_pipe.enable_vae_slicing()
        
        print("✅ Image generator loaded!")
        
    def load_video_generator(self):
        """Load AnimateDiff for video generation"""
        print("🎬 Loading video generator...")
        
        try:
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.video_pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.video_pipe.scheduler = DDIMScheduler.from_config(
                self.video_pipe.scheduler.config,
                beta_schedule="linear",
                steps_offset=1
            )
            
            self.video_pipe.to(self.device)
            
            if self.device == "cuda":
                self.video_pipe.enable_model_cpu_offload()
                self.video_pipe.enable_vae_slicing()
            
            print("✅ Video generator loaded!")
        except Exception as e:
            print(f"⚠️ Video generator loading failed: {e}")
            print("Will use image-to-video fallback method")
    
    def load_depth_estimator(self):
        """Load depth estimation model"""
        print("🔍 Loading depth estimator...")
        
        self.depth_pipe = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=0 if self.device == "cuda" else -1
        )
        
        print("✅ Depth estimator loaded!")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate image from text prompt"""
        
        if self.image_pipe is None:
            self.load_image_generator()
        
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        print(f"🎨 Generating image: '{prompt[:50]}...'")
        
        image = self.image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, static",
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Generate video frames from text prompt"""
        
        if self.video_pipe is None:
            self.load_video_generator()
        
        if self.video_pipe is None:
            # Fallback: generate multiple images
            return self._generate_video_fallback(prompt, num_frames, seed)
        
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        print(f"🎥 Generating video: '{prompt[:50]}...'")
        
        output = self.video_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return output.frames[0]
    
    def _generate_video_fallback(self, prompt, num_frames, seed):
        """Fallback method using multiple images with transitions"""
        print("Using image-based video generation...")
        
        frames = []
        for i in range(num_frames):
            # Add temporal variation to prompt
            frame_prompt = f"{prompt}, frame {i}, cinematic motion"
            frame_seed = (seed + i) if seed else None
            
            image = self.generate_image(
                frame_prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                seed=frame_seed
            )
            frames.append(image)
        
        return frames
    
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Generate depth map from image"""
        
        if self.depth_pipe is None:
            self.load_depth_estimator()
        
        print("🔍 Estimating depth...")
        
        depth = self.depth_pipe(image)
        depth_map = np.array(depth['depth'])
        
        # Normalize to 0-255
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        
        return depth_map
    
    def create_depth_video(self, frames: List[Image.Image]) -> List[np.ndarray]:
        """Create depth maps for video frames"""
        
        depth_frames = []
        for i, frame in enumerate(frames):
            print(f"Processing depth for frame {i+1}/{len(frames)}...")
            depth_map = self.estimate_depth(frame)
            depth_frames.append(depth_map)
        
        return depth_frames
    
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 8,
        add_music: Optional[str] = None
    ):
        """Save frames as video file"""
        
        import imageio
        
        print(f"💾 Saving video to {output_path}...")
        
        # Convert PIL images to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        
        # Save video
        imageio.mimsave(output_path, frame_arrays, fps=fps)
        
        # Add music if provided
        if add_music:
            self._add_audio_to_video(output_path, add_music)
        
        print("✅ Video saved successfully!")
    
    def _add_audio_to_video(self, video_path: str, audio_path: str):
        """Combine video with audio"""
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        try:
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Trim or loop audio to match video duration
            if audio.duration < video.duration:
                audio = audio.audio_loop(duration=video.duration)
            else:
                audio = audio.subclip(0, video.duration)
            
            final = video.set_audio(audio)
            output_path = video_path.replace('.mp4', '_with_audio.mp4')
            final.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            print(f"✅ Video with audio saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to add audio: {e}")


# Example usage functions

def example_music_video_generation():
    """Example: Generate a music video"""
    
    agent = AIVideoGenerator()
    
    # Music video concept
    scenes = [
        "synthwave sunset over neon city, retro 80s aesthetic",
        "dancer in colorful lights, energetic movement, club atmosphere",
        "abstract geometric patterns pulsing with rhythm",
        "silhouette against colorful background, emotional performance"
    ]
    
    all_frames = []
    
    for i, scene in enumerate(scenes):
        print(f"\n🎬 Scene {i+1}/{len(scenes)}")
        frames = agent.generate_video(
            prompt=scene,
            num_frames=16,
            seed=42 + i
        )
        all_frames.extend(frames)
    
    # Save the complete video
    os.makedirs("output", exist_ok=True)
    agent.save_video(all_frames, "output/music_video.mp4", fps=8)
    
    return all_frames


def example_depth_video():
    """Example: Generate video with depth maps"""
    
    agent = AIVideoGenerator()
    
    # Generate video
    prompt = "astronaut floating in space, cosmic background, cinematic"
    frames = agent.generate_video(prompt, num_frames=16, seed=123)
    
    # Create depth maps
    depth_frames = agent.create_depth_video(frames)
    
    # Save both
    os.makedirs("output", exist_ok=True)
    agent.save_video(frames, "output/space_video.mp4", fps=8)
    
    # Save depth video
    depth_images = [Image.fromarray(d) for d in depth_frames]
    agent.save_video(depth_images, "output/space_depth.mp4", fps=8)
    
    print("✅ Video and depth maps saved!")


if __name__ == "__main__":
    print("🎥 AI Video Generator - Ready!")
    print("\nExample 1: Music Video Generation")
    example_music_video_generation()
    
    print("\n" + "="*50)
    print("Example 2: Depth Video Generation")
    example_depth_video()