import subprocess

input_file = "480.mp4"
output_file = "output_240p.mp4"

subprocess.run([
    "ffmpeg",
    "-i", input_file,
    "-vf", "scale=426:240",
    "-c:v", "libx264",
    "-crf", "28",          
    "-preset", "medium",
    "-c:a", "aac",
    "-b:a", "64k",
    output_file
], check=True)

print("240p video created successfully!")