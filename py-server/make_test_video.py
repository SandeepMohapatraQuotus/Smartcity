"""
make_test_video.py
------------------
Generates a synthetic 30-second test video (test_video.mp4) in the current
directory so you can test the MJPEG stream without a physical webcam.

Run:
    python make_test_video.py

Then hit:
    http://localhost:8000/stream/mjpeg?source=/home/quotus/Downloads/traffic1.mp4
"""
import cv2
import numpy as np
import os
import math

OUTPUT   = os.path.join(os.path.dirname(__file__), "test_video.mp4")
WIDTH    = 1280
HEIGHT   = 720
FPS      = 25
DURATION = 60          # seconds — long enough to keep stream running

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

total_frames = FPS * DURATION
print(f"Writing {total_frames} frames → {OUTPUT}")

for i in range(total_frames):
    t   = i / FPS
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Gradient background that shifts over time
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        b = int(20  + 30  * math.sin(t * 0.3 + ratio * 2))
        g = int(10  + 15  * math.sin(t * 0.2))
        r = int(5   + 10  * ratio)
        img[y, :] = [max(0,min(255,b)), max(0,min(255,g)), max(0,min(255,r))]

    # Moving "car" rectangle (simulates vehicle for detector)
    cx = int((WIDTH  * 0.1) + (WIDTH  * 0.8) * ((t * 0.15) % 1.0))
    cy = int(HEIGHT * 0.65)
    cv2.rectangle(img, (cx - 60, cy - 25), (cx + 60, cy + 25), (0, 180, 220), -1)
    cv2.rectangle(img, (cx - 40, cy - 45), (cx + 40, cy - 25), (0, 140, 180), -1)
    cv2.putText(img, "AB12 CD34", (cx - 45, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Second "car" going opposite direction
    cx2 = int(WIDTH - (WIDTH * 0.1) - (WIDTH * 0.8) * ((t * 0.10) % 1.0))
    cy2 = int(HEIGHT * 0.75)
    cv2.rectangle(img, (cx2 - 55, cy2 - 22), (cx2 + 55, cy2 + 22), (220, 80, 40), -1)
    cv2.rectangle(img, (cx2 - 35, cy2 - 38), (cx2 + 35, cy2 - 22), (180, 60, 30), -1)

    # Bouncing "person" ellipse
    px = int(WIDTH  * 0.5 + 200 * math.sin(t * 0.7))
    py = int(HEIGHT * 0.5 + 80  * math.sin(t * 1.1))
    cv2.ellipse(img, (px, py), (18, 36), 0, 0, 360, (200, 160, 120), -1)
    cv2.circle( img, (px, py - 44), 16, (200, 160, 120), -1)

    # Road markings
    for lane_x in range(100, WIDTH, 200):
        cv2.line(img, (lane_x, HEIGHT - 60), (lane_x + 80, HEIGHT - 60), (200, 200, 200), 3)

    # HUD overlay
    cv2.rectangle(img, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(img, f"SYNTHETIC TEST FEED   t={t:.1f}s", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2)

    writer.write(img)
    if i % (FPS * 5) == 0:
        print(f"  {i}/{total_frames} frames written ({t:.0f}s)...")

writer.release()
print(f"\nDone!  {OUTPUT}")
print(f"\nStream URL:")
print(f"  http://localhost:8000/stream/mjpeg?source={OUTPUT}")
