from PIL import Image
import os

def extract_frames(gif_path, output_dir, num_frames=10):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the GIF
    with Image.open(gif_path) as gif:
        total_frames = gif.n_frames
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        extracted_frames = []
        for frame_index in frame_indices:
            gif.seek(frame_index)
            frame = gif.copy()
            frame_filename = os.path.join(output_dir, f"frame_{frame_index}.png")
            frame.save(frame_filename)
            extracted_frames.append(frame_filename)
            print(f"Saved frame {frame_index} as {frame_filename}")
        
    return extracted_frames


gif_path = "component_3_freeBody.gif"  
output_dir = "extracted_frames_body"
extract_frames(gif_path, output_dir)
