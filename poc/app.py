# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import cv2
import time
import sys

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--query_frame",
        type=int,
        default=0,
        help="Frame to start tracking from",
    )
    parser.add_argument(
        "--cam_id",
        type=int,
        default=0,
        help="Camera ID to use (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width for processing (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height for processing (default: 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target FPS (default: 15)",
    )
    parser.add_argument(
        "--point_x",
        type=int,
        default=50,
        help="X coordinate of point to track (default: 50)",
    )
    parser.add_argument(
        "--point_y",
        type=int,
        default=50,
        help="Y coordinate of point to track (default: 50)",
    )

    args = parser.parse_args()

    # Initialize webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera ID {args.cam_id}")
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Calculate the frame interval needed for target FPS
    frame_interval = 1.0 / args.fps

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    # Create a real-time visualizer with custom settings
    vis = Visualizer(
        save_dir=None, 
        pad_value=120, 
        linewidth=3,
        mode="rainbow",  # Use rainbow mode for better visibility
        tracks_leave_trace=5  # Show a trace of the track movement
    )
    
    window_frames = []
    has_initialized_tracking = False
    has_selected_point = False
    
    def _process_step(window_frames, is_first_step, query_frame, point_x, point_y):
        B = 1  # Batch size
        
        # Get video dimensions for the first frame
        if is_first_step:
            H, W = window_frames[0].shape[:2]
            # Create a single query point at the specified coordinates
            # Format is [frame_idx, x, y] 
            single_point = torch.tensor([[[query_frame, float(point_x), float(point_y)]]], device=DEFAULT_DEVICE)
            print(f"Initializing tracking at point [{point_x}, {point_y}]")
        
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        
        if is_first_step:
            return model(
                video_chunk,
                is_first_step=is_first_step,
                queries=single_point,
                grid_size=0,  # Don't use grid
                grid_query_frame=query_frame,
            )
        else:
            return model(
                video_chunk,
                is_first_step=is_first_step,
                grid_size=0,  # Don't use grid
                grid_query_frame=query_frame,
            )

    # Custom visualization function to draw tracks directly on the frame
    def draw_tracks_on_frame(frame, tracks, visibility):
        """Draw tracks directly on a frame for better visualization."""
        if tracks is None or visibility is None:
            return frame
        
        # Get the most recent track position
        if tracks.shape[2] > 0:  # Check if we have any tracks
            for i in range(tracks.shape[2]):  # For each track point
                if visibility[0, -1, i]:  # If point is visible in the last frame
                    # Get coordinates
                    x, y = int(tracks[0, -1, i, 0].item()), int(tracks[0, -1, i, 1].item())
                    
                    # Draw circle
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Big red dot
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # White center
                    
                    # Draw track history if available
                    for t in range(1, min(10, tracks.shape[1])):
                        if t < tracks.shape[1] and visibility[0, -t, i]:
                            prev_x = int(tracks[0, -t, i, 0].item())
                            prev_y = int(tracks[0, -t, i, 1].item())
                            # Draw line with fading color based on age
                            alpha = 1.0 - (t / 10.0)
                            color = (0, int(255 * alpha), int(255 * (1-alpha)))
                            cv2.line(frame, (x, y), (prev_x, prev_y), color, 2)
                            x, y = prev_x, prev_y
        
        return frame

    # Set up display window
    cv2.namedWindow("CoTracker Result", cv2.WINDOW_NORMAL)
    
    # Initialize variables
    is_first_step = True
    frame_count = 0
    last_result_frame = None
    preview_mode = True
    
    # Define callback for mouse click during preview mode
    def mouse_callback(event, x, y, flags, param):
        global has_selected_point, preview_mode
        if event == cv2.EVENT_LBUTTONDOWN and preview_mode:
            args.point_x = x
            args.point_y = y
            print(f"Selected point: [{x}, {y}]")
            has_selected_point = True
            # Start tracking immediately after selecting a point
            preview_mode = False
    
    # Register callback
    cv2.setMouseCallback("CoTracker Result", mouse_callback)
    
    print("Live preview mode. Click on video to select a point and start tracking. Press ESC to exit.")
    
    # For tracking FPS
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    
    # Start with preview mode - show live feed before tracking
    while preview_mode:
        # FPS calculation
        fps_counter += 1
        if fps_counter >= 10:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_display = int(current_fps)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break
        
        # Add instructions and display
        preview_frame = frame.copy()
        cv2.putText(preview_frame, "Click anywhere to select point and start tracking", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(preview_frame, f"FPS: {fps_display}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("CoTracker Result", preview_frame)
        
        # Collect frames for tracking
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        window_frames.append(frame_rgb)
        
        # Keep window_frames buffer at a manageable size
        if len(window_frames) > model.step * 2:
            window_frames.pop(0)
        
        # Check for ESC key
        key = cv2.waitKey(int(1000 / args.fps))
        if key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
    
    # Main tracking loop
    print(f"Starting webcam tracking at point [{args.point_x}, {args.point_y}]. Press ESC to exit, 'r' to reset.")
    last_time = time.time()
    
    while True:
        # FPS calculation
        fps_counter += 1
        if fps_counter >= 10:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_display = int(current_fps)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Maintain fixed FPS
        current_time = time.time()
        elapsed = current_time - last_time
        
        if elapsed < frame_interval:
            # We're ahead of schedule, wait a bit
            time.sleep(frame_interval - elapsed)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break
        
        # Preprocess frame - make RGB from BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        window_frames.append(frame_rgb)
        
        # Keep window_frames buffer at a manageable size
        if len(window_frames) > model.step * 2:
            window_frames.pop(0)
        
        # Process frames once we have enough for a step
        if len(window_frames) >= model.step * 2:
            if is_first_step or (frame_count % model.step == 0 and frame_count > 0):
                # Process a window of frames
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    query_frame=0 if is_first_step else args.query_frame,
                    point_x=args.point_x,
                    point_y=args.point_y
                )
                
                if is_first_step:
                    is_first_step = False
                    has_initialized_tracking = True
                
                # Print tracking coordinates for debugging
                if pred_tracks is not None and pred_tracks.shape[2] > 0:
                    last_pos = pred_tracks[0, -1, 0].cpu().numpy()
                    print(f"Current track position: [{last_pos[0]:.1f}, {last_pos[1]:.1f}]")
                
                # Use our custom visualization instead of the visualizer
                if pred_tracks is not None and pred_visibility is not None:
                    # Get the last frame from the window
                    display_frame = frame.copy()
                    
                    # Draw the tracks directly on the frame
                    display_frame = draw_tracks_on_frame(display_frame, pred_tracks.cpu(), pred_visibility.cpu())
                    
                    # Add FPS display
                    cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    last_result_frame = display_frame
            
            # Always display the most recent result
            if last_result_frame is not None:
                cv2.imshow("CoTracker Result", last_result_frame)
            else:
                # Only show original frame if no result is available yet
                cv2.putText(frame, "Initializing tracking...", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("CoTracker Result", frame)
        else:
            # Just display a message while waiting for enough frames
            cv2.putText(frame, "Collecting frames...", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("CoTracker Result", frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Update last_time for FPS control
        last_time = current_time
        
        # Check for key press
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == 114 or key == 82:  # 'r' key to reset tracking
            is_first_step = True
            has_initialized_tracking = False
            preview_mode = True
            has_selected_point = False
            print("Tracking reset. Click to select new tracking point.")
            
            # Return to preview mode
            break
    
    # If we exited the tracking loop but need to go back to preview mode
    if preview_mode:
        os.execv(sys.executable, ['python'] + sys.argv)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam tracking stopped")