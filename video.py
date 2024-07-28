import argparse
import cv2
import matplotlib
import numpy as np
import torch
import os
import time
import platform

from depth_anything_v2.dpt import DepthAnythingV2

def resize_image(image, patch_size=14):
    h, w = image.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    return cv2.resize(image, (new_w, new_h))

def get_camera():
    if platform.system() == 'Darwin':  # macOS
        cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION + 0)
    else:
        cap = cv2.VideoCapture(0)
    
    # Set a lower resolution for the camera capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    return cap

def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 with Camera Input')
    
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cpu'
    torch.set_num_threads(4)  # Adjust this based on your CPU cores
    
    print(f"Using device: {DEVICE}")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    cap = get_camera()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create smaller windows
    cv2.namedWindow('Camera Capture', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Anything Output', cv2.WINDOW_NORMAL)
    
    # Resize windows to a smaller size
    cv2.resizeWindow('Camera Capture', 320, 240)
    cv2.resizeWindow('Depth Anything Output', 320, 240)

    last_capture_time = time.time() - 0.5  # Ensure first frame is captured immediately

    while True:
        current_time = time.time()
        
        if current_time - last_capture_time >= 0.5:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera.")
                break
            
            try:
                frame = resize_image(frame)
                
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                frame_tensor = frame_tensor.to(DEVICE)
                
                with torch.no_grad():
                    depth = depth_anything(frame_tensor)
                
                depth = depth.squeeze().cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                
                if args.grayscale:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                cv2.imshow('Camera Capture', frame)
                cv2.imshow('Depth Anything Output', depth)
                
                last_capture_time = current_time
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                print("Skipping this frame.")
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()