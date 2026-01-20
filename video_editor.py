import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple
import os

class AdvancedVideoEditor:
    """
    Video editing and effects for AI-generated content
    """
    
    def __init__(self):
        self.effects = {
            'beat_sync': self.beat_sync_effect,
            'color_shift': self.color_shift_effect,
            'glitch': self.glitch_effect,
            'zoom_pulse': self.zoom_pulse_effect,
            'kaleidoscope': self.kaleidoscope_effect
        }
    
    def apply_transition(
        self,
        frame1: Image.Image,
        frame2: Image.Image,
        transition_type: str = 'fade',
        steps: int = 10
    ) -> List[Image.Image]:
        """Create transition between two frames"""
        
        frames = []
        
        if transition_type == 'fade':
            for i in range(steps):
                alpha = i / steps
                blended = Image.blend(frame1, frame2, alpha)
                frames.append(blended)
        
        elif transition_type == 'slide':
            w, h = frame1.size
            for i in range(steps):
                offset = int((i / steps) * w)
                new_frame = Image.new('RGB', (w, h))
                new_frame.paste(frame1, (-offset, 0))
                new_frame.paste(frame2, (w - offset, 0))
                frames.append(new_frame)
        
        elif transition_type == 'dissolve':
            for i in range(steps):
                # Add noise-based dissolve
                alpha = i / steps
                arr1 = np.array(frame1)
                arr2 = np.array(frame2)
                noise = np.random.random(arr1.shape[:2]) > alpha
                result = arr1.copy()
                result[noise] = arr2[noise]
                frames.append(Image.fromarray(result))
        
        return frames
    
    def beat_sync_effect(self, frame: Image.Image, intensity: float) -> Image.Image:
        """Flash/brightness effect synced to beat"""
        
        enhancer = ImageEnhance.Brightness(frame)
        brightened = enhancer.enhance(1 + intensity * 0.5)
        
        return brightened
    
    def color_shift_effect(self, frame: Image.Image, hue_shift: float) -> Image.Image:
        """Shift colors for psychedelic effect"""
        
        arr = np.array(frame)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(rgb)
    
    def glitch_effect(self, frame: Image.Image, intensity: float) -> Image.Image:
        """Digital glitch effect"""
        
        arr = np.array(frame).copy()
        h, w = arr.shape[:2]
        
        # Random horizontal shifts
        for _ in range(int(intensity * 10)):
            y = np.random.randint(0, h)
            shift = np.random.randint(-50, 50)
            if shift > 0:
                arr[y, shift:] = arr[y, :-shift]
            else:
                arr[y, :shift] = arr[y, -shift:]
        
        # Color channel separation
        if np.random.random() < intensity:
            offset = int(intensity * 20)
            arr[:, :, 0] = np.roll(arr[:, :, 0], offset, axis=1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -offset, axis=1)
        
        return Image.fromarray(arr)
    
    def zoom_pulse_effect(self, frame: Image.Image, intensity: float) -> Image.Image:
        """Zoom in/out pulse effect"""
        
        w, h = frame.size
        scale = 1 + intensity * 0.2
        
        new_w, new_h = int(w * scale), int(h * scale)
        zoomed = frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Crop to center
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        result = zoomed.crop((left, top, left + w, top + h))
        
        return result
    
    def kaleidoscope_effect(self, frame: Image.Image, segments: int = 8) -> Image.Image:
        """Create kaleidoscope mirror effect"""
        
        arr = np.array(frame)
        h, w = arr.shape[:2]
        
        # Create kaleidoscope
        result = np.zeros_like(arr)
        angle_step = 360 / segments
        
        for i in range(segments):
            angle = i * angle_step
            matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(arr, matrix, (w, h))
            
            # Create wedge mask
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([
                [w//2, h//2],
                [w, 0],
                [w, h]
            ], np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
            result = cv2.bitwise_or(result, cv2.bitwise_and(rotated, rotated, mask=mask))
        
        return Image.fromarray(result)
    
    def add_text_overlay(
        self,
        frame: Image.Image,
        text: str,
        position: Tuple[int, int] = (50, 50),
        font_size: int = 40,
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """Add text overlay to frame"""
        
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(frame)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Add shadow
        shadow_offset = 2
        draw.text(
            (position[0] + shadow_offset, position[1] + shadow_offset),
            text,
            fill=(0, 0, 0),
            font=font
        )
        
        # Add text
        draw.text(position, text, fill=color, font=font)
        
        return frame
    
    def create_music_visualizer(
        self,
        frames: List[Image.Image],
        audio_file: str,
        output_path: str
    ):
        """Create music visualization overlay"""
        
        print("🎵 Creating music visualization...")
        
        # This is a placeholder - would need audio analysis library
        # like librosa for actual beat detection
        
        processed_frames = []
        for i, frame in enumerate(frames):
            # Simulate beat intensity
            beat_intensity = abs(np.sin(i * 0.5)) * 0.5
            
            # Apply beat-synced effects
            if beat_intensity > 0.3:
                frame = self.beat_sync_effect(frame, beat_intensity)
            
            processed_frames.append(frame)
        
        return processed_frames


# Example usage
def create_music_video_with_effects():
    """Complete example with effects"""
    
    from ai_video_agent import AIVideoGenerator
    
    agent = AIVideoGenerator()
    editor = AdvancedVideoEditor()
    
    # Generate base video
    prompt = "neon cityscape at night, cyberpunk aesthetic, vibrant colors"
    frames = agent.generate_video(prompt, num_frames=32, seed=999)
    
    # Apply effects
    processed_frames = []
    for i, frame in enumerate(frames):
        # Cycle through effects
        effect_cycle = i % 4
        
        if effect_cycle == 0:
            frame = editor.color_shift_effect(frame, i / len(frames))
        elif effect_cycle == 1:
            frame = editor.zoom_pulse_effect(frame, abs(np.sin(i * 0.3)))
        elif effect_cycle == 2:
            frame = editor.glitch_effect(frame, 0.3)
        else:
            frame = editor.kaleidoscope_effect(frame, 6)
        
        processed_frames.append(frame)
    
    # Add transitions between scenes
    final_frames = []
    for i in range(len(processed_frames) - 1):
        final_frames.append(processed_frames[i])
        if i % 8 == 7:  # Transition every 8 frames
            transition = editor.apply_transition(
                processed_frames[i],
                processed_frames[i + 1],
                'fade',
                steps=4
            )
            final_frames.extend(transition)
    
    # Save
    os.makedirs("output", exist_ok=True)
    agent.save_video(final_frames, "output/effects_video.mp4", fps=12)
    
    print("✅ Effects video created!")


if __name__ == "__main__":
    create_music_video_with_effects()