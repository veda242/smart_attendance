"""
Quick camera diagnostic script
"""
import cv2
import sys

print("[INFO] Testing camera access...")
print("=" * 50)

# Test default camera (0)
for camera_index in range(5):
    print(f"\n[TEST] Trying camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if cap.isOpened():
        print(f"  ✓ Camera {camera_index} found!")
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Resolution: {int(width)}x{int(height)}")
        print(f"  FPS: {fps}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Successfully read frame: {frame.shape}")
        else:
            print(f"  ✗ Failed to read frame")
        
        cap.release()
    else:
        print(f"  ✗ Camera {camera_index} not available")

print("\n" + "=" * 50)
print("[SUMMARY]")
print("If no camera was found, try:")
print("  1. Check if camera is connected")
print("  2. Check if camera is used by another app")
print("  3. Check device manager for camera drivers")
print("  4. Restart the computer")
