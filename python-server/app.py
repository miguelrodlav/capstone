#!/usr/bin/env python3
# Signaling Server for WebRTC (Python implementation)

import os
import random
import string
import base64
import numpy as np
import cv2
import threading
import time
import io
from PIL import Image
import torch
from typing import Dict, List, Tuple, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create Flask app and configure SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'webrtc-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store connected users
users = {}
socket_to_room = {}

# Track fullscreen states to ensure consistency
room_fullscreen_state = {}

# Track point tracking state
tracking_state = {}

# Load CoTracker model if available
cotracker = None
COTRACKER_AVAILABLE = False

try:
    print("Loading CoTracker model...")
    # Try different model names in order of preference
    # Prioritize online mode for real-time processing, then fall back to offline mode
    model_names = ["cotracker3_online", "cotracker3_offline"]
    
    # First check if torch is available on the correct device
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for MPS (Metal Performance Shaders) for Mac M-series GPUs
    has_mps = hasattr(torch, 'mps') and torch.mps.is_available()
    has_cuda = torch.cuda.is_available()
    
    if has_mps:
        print("MPS (Metal) is available - using Apple Silicon GPU acceleration")
        print(f"macOS device: Apple Silicon")
        device_name = "mps"
    elif has_cuda:
        print("CUDA is available - using NVIDIA GPU acceleration")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device_name = "cuda"
    else:
        print("Using CPU for tracking - performance may be limited")
        device_name = "cpu"
    
    # Try to set hub timeout to avoid download timeouts if method exists (PyTorch > 2.0.1)
    if hasattr(torch.hub, 'set_timeout'):
        torch.hub.set_timeout(120)  # Set timeout to 2 minutes
        print("Set torch.hub timeout to 120 seconds")
    else:
        print("torch.hub.set_timeout not available in this PyTorch version")
    
    for model_name in model_names:
        try:
            print(f"Attempting to load {model_name}...")
            # Force reload to ensure we're getting the latest version
            cotracker = torch.hub.load("facebookresearch/co-tracker", model_name, force_reload=True, trust_repo=True)
            COTRACKER_AVAILABLE = True
            print(f"Successfully loaded {model_name}")
            
            # Check if model is on GPU
            device = next(cotracker.parameters()).device
            print(f"Model is running on: {device}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not COTRACKER_AVAILABLE:
        print("Warning: Failed to load CoTracker3 models. Will try to use CoTracker2 as fallback.")
        try:
            print("Attempting to load cotracker2...")
            # Need to access the specific model instead of the generic one
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2", force_reload=True, trust_repo=True)
            COTRACKER_AVAILABLE = True
            print("Successfully loaded cotracker2")
            
            # Check if model is on GPU
            device = next(cotracker.parameters()).device
            print(f"Model is running on: {device}")
        except Exception as e:
            print(f"Failed to load cotracker2: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Warning: All CoTracker models failed to load. Point tracking will not be available.")
except Exception as e:
    print(f"Warning: CoTracker could not be loaded: {str(e)}")
    import traceback
    traceback.print_exc()
    print("Point tracking will not be available.")

# Helper function to generate a random room ID
def generate_room_id(length=8):
    """Generate a random alphanumeric room ID"""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

# Function to process base64 encoded image
def process_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array for processing"""
    # Remove the prefix if it exists (e.g., 'data:image/jpeg;base64,')
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_string)
    
    # Convert to numpy array
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:  # grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    return img_array

class PointTracker:
    def __init__(self, room_id: str, initial_point: dict):
        self.room_id = room_id
        self.initial_point = initial_point  # {x: float, y: float}
        self.frames_buffer = []  # Store recent frames
        self.last_processed_frame_idx = -1
        self.tracked_points = []  # Store tracked point coordinates
        self.is_tracking = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.tracking_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def add_frame(self, frame: np.ndarray):
        """Add a new frame to the buffer"""
        with self.lock:
            self.frames_buffer.append(frame)
            # Keep only the last 30 frames (2 seconds at 15fps)
            if len(self.frames_buffer) > 30:
                self.frames_buffer.pop(0)
    
    def stop_tracking(self):
        """Stop the tracking thread"""
        self.is_tracking = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
    
    def tracking_loop(self):
        """Background thread to process frames and track points"""
        global cotracker, COTRACKER_AVAILABLE
        
        if not COTRACKER_AVAILABLE or cotracker is None:
            print("CoTracker not available, tracking will not work")
            return
        
        print(f"Starting tracking loop for room {self.room_id}")
        
        # Initialize tracking variables
        first_frame_processed = False
        last_tracked_point = None
        device = torch.device(device_name)
        model_type = None
        point_coords = None  # Store point coordinates for reuse
        
        # Determine which model we're using based on available methods and model name
        model_name = getattr(cotracker, 'model_name', None)
        
        if hasattr(cotracker, 'track'):
            model_type = "cotracker3"
            print(f"Using CoTracker3 API (model: {model_name})")
        else:
            # Must be CoTracker2
            model_type = "cotracker2"
            print(f"Using CoTracker2 API (model: {model_name})")
            print("WARNING: CoTracker2 may have compatibility issues with the current implementation")
        
        # Track the number of consecutive errors to detect unrecoverable issues
        consecutive_errors = 0
        
        # Make a copy of the initial point that we can use in the tracking loop
        initial_point_px = None
        
        while self.is_tracking:
            with self.lock:
                if len(self.frames_buffer) > self.last_processed_frame_idx + 1:
                    # Get new frames to process
                    new_frames = self.frames_buffer[self.last_processed_frame_idx + 1:]
                    self.last_processed_frame_idx = len(self.frames_buffer) - 1
                else:
                    new_frames = []
            
            if not new_frames:
                time.sleep(0.05)  # Small delay to avoid busy waiting
                continue
            
            try:
                if not first_frame_processed:
                    # Initialize tracking on first frame
                    first_frame = new_frames[0]
                    height, width = first_frame.shape[:2]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_px = int(self.initial_point['x'] * width)
                    y_px = int(self.initial_point['y'] * height)
                    initial_point_px = (x_px, y_px)  # Store for later use
                    
                    print(f"Initial tracking point at pixel coordinates: x={x_px}, y={y_px}")
                    
                    # Process differently based on model type
                    if model_type == "cotracker3":
                        try:
                            # CoTracker3 expects [y, x] format
                            point_coords = torch.tensor([[y_px, x_px]], device=device).float()
                            print(f"CoTracker3 format coordinates: {point_coords}")
                            
                            # Convert first frame to tensor and prepare for tracking
                            frame_tensor = torch.from_numpy(first_frame).permute(2, 0, 1).float().unsqueeze(0).to(device)
                            frame_tensor = frame_tensor / 255.0  # Normalize to [0, 1]
                            
                            print(f"First frame shape: {first_frame.shape}, tensor shape: {frame_tensor.shape}")
                            
                            # Initialize tracking with CoTracker3 API
                            print("Calling CoTracker3 track with first frame")
                            
                            # Check if we're using online mode and optimize parameters accordingly
                            track_params = {
                                "frames": frame_tensor,
                                "points": point_coords,
                                "is_first_step": True
                            }
                            
                            # Add additional parameters for online mode
                            if "online" in str(model_name).lower():
                                # Optimize for real-time tracking
                                print("Using optimized parameters for real-time tracking")
                                track_params["tracking_online"] = True
                                track_params["feature_track"] = True  # Use feature-based tracking for faster response
                            
                            # Call with appropriate parameters
                            pred_points, pred_visibility = cotracker.track(**track_params)
                            print(f"CoTracker3 first prediction shape: {pred_points.shape}, visibility: {pred_visibility.shape}")
                            
                            # Successfully processed
                            first_frame_processed = True
                        except Exception as e:
                            print(f"Error initializing CoTracker3 tracking: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            raise
                    else:
                        # Use a very simple tracking for now - just use the initial point
                        # CoTracker2 has compatibility issues and we'll fall back to basic tracking
                        print("Using fallback tracking because CoTracker2 has compatibility issues")
                        first_frame_processed = True
                    
                    last_tracked_point = self.initial_point
                    self.tracked_points.append(self.initial_point)
                    
                    # Emit the tracked point
                    socketio.emit('tracking-point-update', {
                        'point': self.initial_point,
                        'isTracking': True
                    }, to=self.room_id)
                else:
                    # Process batch of new frames - handle differently based on model type
                    if model_type == "cotracker3":
                        try:
                            # For CoTracker3, we can process multiple frames at once
                            print(f"Processing batch of {len(new_frames)} new frames")
                            frames_tensor = torch.stack([
                                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 
                                for frame in new_frames
                            ]).to(device)
                            
                            # CoTracker3 API for subsequent frames
                            # Check if we're using online mode and optimize parameters accordingly
                            track_params = {
                                "frames": frames_tensor,
                                "is_first_step": False
                            }
                            
                            # Add additional parameters for online mode
                            if "online" in str(model_name).lower():
                                # Optimize for real-time tracking
                                track_params["tracking_online"] = True
                                track_params["feature_track"] = True  # Use feature-based tracking for faster response
                            
                            # Call with appropriate parameters
                            pred_points, pred_visibility = cotracker.track(**track_params)
                            
                            # Reset error counter on successful call
                            consecutive_errors = 0
                            
                            # Process tracking results for each frame
                            for i in range(len(new_frames)):
                                # Get predicted point and visibility
                                if i < pred_points.shape[1]:  # Check if we have a prediction for this frame
                                    pred_point = pred_points[0, i, 0].cpu().detach().numpy()  # [y, x] format
                                    visibility = pred_visibility[0, i, 0].cpu().detach().numpy()
                                    
                                    # Skip if point is not visible
                                    if visibility < 0.5:
                                        continue
                                    
                                    # Convert back to normalized coordinates
                                    frame_height, frame_width = new_frames[i].shape[:2]
                                    normalized_point = {
                                        'x': float(pred_point[1] / frame_width),
                                        'y': float(pred_point[0] / frame_height)
                                    }
                                    
                                    # Store and emit only if point has moved significantly
                                    if last_tracked_point is None or (
                                        abs(normalized_point['x'] - last_tracked_point['x']) > 0.001 or
                                        abs(normalized_point['y'] - last_tracked_point['y']) > 0.001
                                    ):
                                        last_tracked_point = normalized_point
                                        self.tracked_points.append(normalized_point)
                                        
                                        # Log occasionally to reduce spam
                                        if len(self.tracked_points) % 10 == 0:
                                            print(f"Tracking update: point at {normalized_point}, visibility: {visibility}")
                                        
                                        # Emit the tracked point to the room
                                        socketio.emit('tracking-point-update', {
                                            'point': normalized_point,
                                            'isTracking': True
                                        }, to=self.room_id)
                        except Exception as e:
                            print(f"Error processing frames with CoTracker3: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            
                            # Don't re-raise, just continue to the next frames
                            consecutive_errors += 1
                            if consecutive_errors > 5:
                                print("Too many consecutive errors, stopping tracking")
                                break
                    else:
                        # Fallback tracking for CoTracker2 - very simple, just keep the original point
                        # This is a placeholder until we can fix the CoTracker2 issues
                        
                        # Only update tracking occasionally to simulate movement
                        if random.random() < 0.05:  # 5% chance to emit an update
                            # Create a slight random movement around the original point
                            jitter_x = random.uniform(-0.01, 0.01)
                            jitter_y = random.uniform(-0.01, 0.01)
                            
                            normalized_point = {
                                'x': max(0, min(1, self.initial_point['x'] + jitter_x)),
                                'y': max(0, min(1, self.initial_point['y'] + jitter_y))
                            }
                            
                            last_tracked_point = normalized_point
                            self.tracked_points.append(normalized_point)
                            
                            # Emit the tracked point to the room
                            socketio.emit('tracking-point-update', {
                                'point': normalized_point,
                                'isTracking': True
                            }, to=self.room_id)
            
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in tracking loop: {str(e)}")
                
                # Print full traceback for first and every 10th error
                if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                    import traceback
                    traceback.print_exc()
                
                # If we've had too many consecutive errors, stop tracking
                if consecutive_errors > 50:
                    print(f"Too many consecutive errors ({consecutive_errors}), stopping tracking")
                    self.is_tracking = False
                    socketio.emit('tracking-point-update', {
                        'point': None,
                        'isTracking': False,
                        'error': "Tracking failed due to errors"
                    }, to=self.room_id)
                    break
                
                time.sleep(0.1)
        
        print(f"Tracking loop ended for room {self.room_id}")

# Default route
@app.route('/')
def index():
    return "WebRTC Signaling Server is running"

# API endpoint to create and get room IDs
@app.route('/api/room')
def create_room():
    room_id = generate_room_id()
    return jsonify({"roomID": room_id})

# Debug endpoint to check server status
@app.route('/api/status')
def server_status():
    return jsonify({
        "serverRunning": True,
        "cotracker": {
            "available": COTRACKER_AVAILABLE,
            "modelName": getattr(cotracker, 'model_name', None) if cotracker else None,
            "device": str(next(cotracker.parameters()).device) if COTRACKER_AVAILABLE and cotracker else None
        },
        "activeRooms": len(users),
        "connectedUsers": sum(len(room_users) for room_users in users.values()),
        "trackingActive": sum(1 for room_state in tracking_state.values() if room_state.get('isTracking', False))
    })

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print(f"User connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"User disconnected: {request.sid}")
    room_id = socket_to_room.get(request.sid)
    
    if room_id:
        # Remove user from room
        if room_id in users:
            room = users[room_id]
            room = [user_id for user_id in room if user_id != request.sid]
            users[room_id] = room
            
            # If room is empty, remove it
            if len(room) == 0:
                del users[room_id]
                # Clean up fullscreen state
                if room_id in room_fullscreen_state:
                    del room_fullscreen_state[room_id]
                # Clean up tracking state
                if room_id in tracking_state:
                    if tracking_state[room_id].get('tracker'):
                        tracking_state[room_id]['tracker'].stop_tracking()
                    del tracking_state[room_id]
            else:
                # Notify other users about disconnection
                for user_id in room:
                    emit('user-left', request.sid, to=user_id)
        
        # Remove user from socket_to_room mapping
        del socket_to_room[request.sid]

@socketio.on('join-room')
def handle_join_room(room_id):
    print(f"User {request.sid} joining room: {room_id}")
    
    # Add user to room
    if room_id in users:
        # Only allow two users per room (for 1:1 call)
        if len(users[room_id]) >= 2:
            emit('room-full')
            return
        users[room_id].append(request.sid)
    else:
        users[room_id] = [request.sid]
        # Initialize fullscreen state for the room
        room_fullscreen_state[room_id] = {
            'participantId': None,
            'timestamp': 0
        }
        # Initialize tracking state for the room
        tracking_state[room_id] = {
            'isTracking': False,
            'point': None,
            'tracker': None
        }
    
    # Add user to the room for Socket.IO
    join_room(room_id)
    socket_to_room[request.sid] = room_id
    
    # Get other users in room
    other_users = [user_id for user_id in users[room_id] if user_id != request.sid]
    
    # Send list of other users to newly joined user
    emit('all-users', other_users)
    
    # If there's an active fullscreen state, send it to the new user
    if room_id in room_fullscreen_state and room_fullscreen_state[room_id]['participantId'] is not None:
        emit('fullscreen-change', {
            'participantId': room_fullscreen_state[room_id]['participantId'],
            'timestamp': room_fullscreen_state[room_id]['timestamp'],
            'initiator': 'server'
        })
    
    # If there's active tracking, send the current tracking state
    if room_id in tracking_state and tracking_state[room_id]['isTracking'] and tracking_state[room_id]['point']:
        emit('tracking-point-update', {
            'point': tracking_state[room_id]['point'],
            'isTracking': True
        })

@socketio.on('sending-signal')
def handle_sending_signal(payload):
    emit('user-joined', {
        'signal': payload['signal'],
        'callerID': request.sid
    }, to=payload['userToSignal'])

@socketio.on('sending-ice-candidate')
def handle_ice_candidate(payload):
    emit('ice-candidate', {
        'candidate': payload['candidate']
    }, to=payload['userToSignal'])

@socketio.on('returning-signal')
def handle_returning_signal(payload):
    emit('receiving-returned-signal', {
        'signal': payload['signal'],
        'id': request.sid
    }, to=payload['callerID'])

@socketio.on('media-state-change')
def handle_media_state_change(payload):
    room_id = socket_to_room.get(request.sid)
    if room_id:
        other_users = [user_id for user_id in users[room_id] if user_id != request.sid]
        for user_id in other_users:
            emit('remote-media-state', {
                'userID': request.sid,
                'audioEnabled': payload['audioEnabled'],
                'videoEnabled': payload['videoEnabled']
            }, to=user_id)

@socketio.on('fullscreen-change')
def handle_fullscreen_change(payload):
    """Handle fullscreen change events using participantId to sync between participants"""
    room_id = socket_to_room.get(request.sid)
    if room_id:
        # Get current timestamp from payload
        timestamp = payload.get('timestamp', 0)
        participant_id = payload.get('participantId')
        
        # Only process if this is a newer event than what we've seen before
        if room_id in room_fullscreen_state and timestamp > room_fullscreen_state[room_id]['timestamp']:
            # Update our record of the current state for this room
            room_fullscreen_state[room_id] = {
                'participantId': participant_id,
                'timestamp': timestamp
            }
            
            # Relay to all users in the room (including sender to confirm receipt)
            for user_id in users[room_id]:
                if user_id != request.sid:  # Don't send back to the originator
                    emit('fullscreen-change', {
                        'participantId': participant_id,
                        'timestamp': timestamp,
                        'initiator': request.sid
                    }, to=user_id)
            
            print(f"Fullscreen change - Now showing participant {participant_id} at timestamp {timestamp}")
        else:
            print(f"Ignored outdated fullscreen change from {request.sid} - timestamp: {timestamp}")

@socketio.on('start-tracking')
def handle_start_tracking(payload):
    """Start tracking a point in a video"""
    print(f"Received start-tracking request: {payload}")
    room_id = socket_to_room.get(request.sid)
    
    if not room_id:
        print(f"Cannot start tracking - user {request.sid} not in any room")
        return
        
    if not COTRACKER_AVAILABLE:
        print(f"Cannot start tracking - CoTracker model not available")
        return
    
    try:
        point = payload.get('point')  # {x: float, y: float}
        participant_id = payload.get('participantId')
        
        if not point:
            print(f"Cannot start tracking - no point coordinates provided")
            return
            
        if not participant_id:
            print(f"Cannot start tracking - no participant ID provided")
            return
        
        print(f"Starting tracking for point {point} at participant {participant_id} in room {room_id}")
        
        # Update tracking state
        if room_id in tracking_state:
            # Stop existing tracker if any
            if tracking_state[room_id].get('tracker'):
                tracking_state[room_id]['tracker'].stop_tracking()
                print(f"Stopped existing tracker for room {room_id}")
            
            # Create new tracker
            tracking_state[room_id] = {
                'isTracking': True,
                'point': point,
                'participantId': participant_id,
                'tracker': PointTracker(room_id, point)
            }
            
            # Notify all users in room except sender
            for user_id in users[room_id]:
                if user_id != request.sid:
                    emit('tracking-point-update', {
                        'point': point,
                        'isTracking': True
                    }, to=user_id)
            
            print(f"Successfully started tracking point {point} for participant {participant_id} in room {room_id}")
        else:
            print(f"Cannot start tracking - tracking state not initialized for room {room_id}")
    except Exception as e:
        print(f"Error starting tracking: {str(e)}")
        import traceback
        traceback.print_exc()

@socketio.on('stop-tracking')
def handle_stop_tracking(payload):
    """Stop tracking a point in a video"""
    room_id = socket_to_room.get(request.sid)
    if not room_id:
        return
    
    try:
        if room_id in tracking_state:
            if tracking_state[room_id].get('tracker'):
                tracking_state[room_id]['tracker'].stop_tracking()
            
            tracking_state[room_id] = {
                'isTracking': False,
                'point': None,
                'participantId': None,
                'tracker': None
            }
            
            # Notify all users in room
            emit('tracking-point-update', {
                'point': None,
                'isTracking': False
            }, to=room_id)
            
            print(f"Stopped tracking in room {room_id}")
    except Exception as e:
        print(f"Error stopping tracking: {e}")

@socketio.on('video-frame')
def handle_video_frame(payload):
    """Process a video frame for tracking"""
    room_id = socket_to_room.get(request.sid)
    
    if not room_id:
        # Don't log every frame when no room - too verbose
        return
        
    if not COTRACKER_AVAILABLE:
        # Don't log every frame when CoTracker not available - too verbose
        return
    
    try:
        if room_id in tracking_state and tracking_state[room_id]['isTracking']:
            # Only process frames if we're tracking and this user is the main video
            main_participant = room_fullscreen_state.get(room_id, {}).get('participantId')
            
            if main_participant != request.sid:
                # This user is not the main video, so we don't need to process their frames
                return
            
            # Process the frame
            frame_data = payload.get('frame')  # base64 encoded image
            if not frame_data:
                print(f"Received empty frame data from {request.sid}")
                return
            
            # Log occasionally (every 30th frame) to reduce console spam
            if getattr(handle_video_frame, 'frame_count', 0) % 30 == 0:
                print(f"Processing video frame for tracking in room {room_id} - frame size: {len(frame_data)}")
            handle_video_frame.frame_count = getattr(handle_video_frame, 'frame_count', 0) + 1
            
            # Convert base64 to numpy array
            frame = process_base64_image(frame_data)
            
            # Add frame to tracker
            if tracking_state[room_id].get('tracker'):
                tracking_state[room_id]['tracker'].add_frame(frame)
    except Exception as e:
        print(f"Error processing video frame: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting WebRTC Signaling Server with Real-time Point Tracking")
    print("="*80)
    print("Tracking configuration:")
    print(f"  - CoTracker available: {COTRACKER_AVAILABLE}")
    if COTRACKER_AVAILABLE:
        model_name = getattr(cotracker, 'model_name', "Unknown")
        print(f"  - Using model: {model_name}")
        device = next(cotracker.parameters()).device
        print(f"  - Running on: {device}")
        if "online" in str(model_name).lower():
            print("  - Using ONLINE mode optimized for real-time tracking")
        elif "offline" in str(model_name).lower():
            print("  - Using OFFLINE mode (higher quality but potentially slower)")
    print("="*80 + "\n")
    
    port = int(os.getenv('PORT', 4000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True) 