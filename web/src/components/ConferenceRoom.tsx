"use client";

import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import VideoPlayer from "./VideoPlayer";
import ControlBar from "./ControlBar";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";

export default function ConferenceRoom() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const roomID = searchParams.get("room");

  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [audioEnabled, setAudioEnabled] = useState<boolean>(true);
  const [videoEnabled, setVideoEnabled] = useState<boolean>(true);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>(
    []
  );
  const [currentCamera, setCurrentCamera] = useState<string>("");
  const [connected, setConnected] = useState<boolean>(false);
  const [waiting, setWaiting] = useState<boolean>(false);
  const [roomLink, setRoomLink] = useState<string>("");
  const [mainVideo, setMainVideo] = useState<string | null>(null);
  const [fullscreenTimestamp, setFullscreenTimestamp] = useState<number>(0);

  // New state for tracking points
  const [trackingPoint, setTrackingPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [isTracking, setIsTracking] = useState<boolean>(false);
  const [canSelectPoint, setCanSelectPoint] = useState<boolean>(false);

  // New state for video frame capture
  const [isCapturingFrames, setIsCapturingFrames] = useState<boolean>(false);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const frameCapturerRef = useRef<number | null>(null);

  const socketRef = useRef<Socket | null>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const otherUser = useRef<string | null>(null);
  const userStream = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const localVideoFrameRef = useRef<HTMLDivElement>(null);

  // Initialize Socket.io connection and media streams
  useEffect(() => {
    // If no roomID is provided, create one and redirect
    if (!roomID) {
      fetch("http://localhost:4000/api/room")
        .then((res) => res.json())
        .then((data) => {
          router.push(`/conference?room=${data.roomID}`);
        });
      return;
    }

    // Set room link for sharing
    setRoomLink(`${window.location.origin}/conference?room=${roomID}`);
    setWaiting(true);

    // Connect to signaling server
    socketRef.current = io("http://localhost:4000");

    // Get user media
    navigator.mediaDevices
      .getUserMedia({
        video: {
          frameRate: { ideal: 15, max: 15 },
        },
        audio: true,
      })
      .then((stream) => {
        setLocalStream(stream);
        userStream.current = stream;

        // Get available cameras
        navigator.mediaDevices.enumerateDevices().then((devices) => {
          const cameras = devices.filter(
            (device) => device.kind === "videoinput"
          );
          setAvailableCameras(cameras);

          // Set current camera
          const videoTrack = stream.getVideoTracks()[0];
          if (videoTrack) {
            setCurrentCamera(videoTrack.getSettings().deviceId || "");
          }
        });

        // Join room after getting media
        if (socketRef.current) {
          socketRef.current.emit("join-room", roomID);
        }
      })
      .catch((err) => {
        console.error("Error getting user media:", err);
      });

    return () => {
      // Clean up
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (peerRef.current) {
        peerRef.current.close();
      }
      if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [roomID, router]);

  // Set up socket event listeners
  useEffect(() => {
    if (!socketRef.current) return;

    // When we are the first one in the room
    socketRef.current.on("all-users", (users: string[]) => {
      // If there are no other users, we're waiting for someone to join
      if (users.length === 0) {
        setWaiting(true);
        return;
      }

      // If there's another user, initiate call
      setWaiting(false);
      const otherUserID = users[0];
      otherUser.current = otherUserID;

      // Create peer connection and initiate call
      createPeerConnection(otherUserID, true);
    });

    // When someone joins our room
    socketRef.current.on(
      "user-joined",
      (payload: { signal: any; callerID: string }) => {
        setWaiting(false);
        otherUser.current = payload.callerID;

        // Create peer connection and handle the incoming offer
        if (payload.signal && payload.signal.type === "offer") {
          createPeerConnection(payload.callerID, false, payload.signal);
        }
      }
    );

    // When we receive ICE candidates from the other peer
    socketRef.current.on(
      "ice-candidate",
      (incoming: { candidate: RTCIceCandidateInit }) => {
        if (peerRef.current) {
          peerRef.current
            .addIceCandidate(new RTCIceCandidate(incoming.candidate))
            .catch((e) =>
              console.error("Error adding received ice candidate", e)
            );
        }
      }
    );

    // When we receive the returned answer
    socketRef.current.on(
      "receiving-returned-signal",
      (payload: { signal: RTCSessionDescriptionInit; id: string }) => {
        if (peerRef.current) {
          peerRef.current
            .setRemoteDescription(new RTCSessionDescription(payload.signal))
            .catch((e) =>
              console.error("Error setting remote description:", e)
            );
        }
      }
    );

    // When a user leaves the room
    socketRef.current.on("user-left", () => {
      setConnected(false);
      setRemoteStream(null);
      otherUser.current = null;
      setWaiting(true);

      if (peerRef.current) {
        peerRef.current.close();
        peerRef.current = null;
      }
    });

    // When remote media state changes
    socketRef.current.on(
      "remote-media-state",
      (payload: { audioEnabled: boolean; videoEnabled: boolean }) => {
        // This could be used to show UI indicators for remote mute state
        console.log("Remote media state:", payload);
      }
    );

    // When room is full
    socketRef.current.on("room-full", () => {
      alert("This room is full. Please try another room.");
      router.push("/");
    });

    // Update the fullscreen sync event handler
    socketRef.current.on(
      "fullscreen-change",
      (payload: {
        participantId: string | null;
        timestamp: number;
        initiator: string;
      }) => {
        console.log("Received fullscreen change:", payload);

        // Only update if this is a newer event than what we already processed
        if (payload.timestamp > fullscreenTimestamp) {
          setFullscreenTimestamp(payload.timestamp);
          setMainVideo(payload.participantId);
        }
      }
    );

    // Event for tracking point updates
    socketRef.current.on(
      "tracking-point-update",
      (payload: {
        point: { x: number; y: number } | null;
        isTracking: boolean;
      }) => {
        setTrackingPoint(payload.point);
        setIsTracking(payload.isTracking);
      }
    );

    return () => {
      if (socketRef.current) {
        socketRef.current.off("all-users");
        socketRef.current.off("user-joined");
        socketRef.current.off("ice-candidate");
        socketRef.current.off("receiving-returned-signal");
        socketRef.current.off("user-left");
        socketRef.current.off("remote-media-state");
        socketRef.current.off("room-full");
        socketRef.current.off("fullscreen-change");
        socketRef.current.off("tracking-point-update");
      }
    };
  }, [router, fullscreenTimestamp]);

  // Create RTCPeerConnection and handle signaling
  const createPeerConnection = (
    userID: string,
    isInitiator: boolean,
    incomingOffer?: RTCSessionDescriptionInit
  ) => {
    const peer = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.stunprotocol.org" },
        { urls: "stun:stun.l.google.com:19302" },
      ],
    });

    peerRef.current = peer;

    // Add local tracks to the connection
    if (userStream.current) {
      userStream.current.getTracks().forEach((track) => {
        if (userStream.current) {
          peer.addTrack(track, userStream.current);
        }
      });
    }

    // Listen for remote tracks
    peer.ontrack = (event) => {
      setRemoteStream(event.streams[0]);
      setConnected(true);
    };

    // Handle ICE candidates
    peer.onicecandidate = (event) => {
      if (event.candidate && socketRef.current && otherUser.current) {
        socketRef.current.emit("sending-ice-candidate", {
          userToSignal: otherUser.current,
          candidate: event.candidate,
        });
      }
    };

    // Handle connection state changes
    peer.onconnectionstatechange = () => {
      switch (peer.connectionState) {
        case "connected":
          console.log("WebRTC peers connected!");
          setConnected(true);
          break;
        case "disconnected":
        case "failed":
          console.log("WebRTC connection failed or disconnected");
          setConnected(false);
          break;
        default:
          break;
      }
    };

    // Handle negotiation needed
    peer.onnegotiationneeded = () => {
      if (isInitiator) {
        peer
          .createOffer()
          .then((offer) => peer.setLocalDescription(offer))
          .then(() => {
            if (
              socketRef.current &&
              peer.localDescription &&
              otherUser.current
            ) {
              socketRef.current.emit("sending-signal", {
                userToSignal: otherUser.current,
                signal: peer.localDescription,
              });
            }
          })
          .catch((e) => console.error("Error creating offer:", e));
      }
    };

    // If not initiator, handle the incoming offer
    if (!isInitiator && incomingOffer) {
      peer
        .setRemoteDescription(new RTCSessionDescription(incomingOffer))
        .then(() => peer.createAnswer())
        .then((answer) => peer.setLocalDescription(answer))
        .then(() => {
          if (socketRef.current && peer.localDescription && otherUser.current) {
            socketRef.current.emit("returning-signal", {
              callerID: otherUser.current,
              signal: peer.localDescription,
            });
          }
        })
        .catch((e) => console.error("Error handling incoming offer:", e));
    }
  };

  // Toggle audio
  const toggleAudio = () => {
    if (localStream) {
      const audioTracks = localStream.getAudioTracks();
      if (audioTracks.length > 0) {
        const enabled = !audioTracks[0].enabled;
        audioTracks[0].enabled = enabled;
        setAudioEnabled(enabled);

        // Notify other user about media state change
        if (socketRef.current && connected) {
          socketRef.current.emit("media-state-change", {
            audioEnabled: enabled,
            videoEnabled: videoEnabled,
          });
        }
      }
    }
  };

  // Toggle video
  const toggleVideo = () => {
    if (localStream) {
      const videoTracks = localStream.getVideoTracks();
      if (videoTracks.length > 0) {
        const enabled = !videoTracks[0].enabled;
        videoTracks[0].enabled = enabled;
        setVideoEnabled(enabled);

        // Notify other user about media state change
        if (socketRef.current && connected) {
          socketRef.current.emit("media-state-change", {
            audioEnabled: audioEnabled,
            videoEnabled: enabled,
          });
        }
      }
    }
  };

  // Switch camera
  const switchCamera = async () => {
    if (!localStream || availableCameras.length <= 1) return;

    try {
      // Find the next camera in the list
      const currentIndex = availableCameras.findIndex(
        (camera) => camera.deviceId === currentCamera
      );
      const nextIndex = (currentIndex + 1) % availableCameras.length;
      const nextCamera = availableCameras[nextIndex];

      // Get new stream with the next camera
      const newStream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: nextCamera.deviceId },
          frameRate: { ideal: 15, max: 15 },
        },
        audio: true,
      });

      // Replace video track in local stream
      const newVideoTrack = newStream.getVideoTracks()[0];
      const oldVideoTrack = localStream.getVideoTracks()[0];

      if (oldVideoTrack) {
        localStream.removeTrack(oldVideoTrack);
        oldVideoTrack.stop();
      }

      localStream.addTrack(newVideoTrack);

      // Copy current audio track settings
      if (localStream.getAudioTracks().length > 0) {
        const audioEnabled = localStream.getAudioTracks()[0].enabled;
        newStream.getAudioTracks().forEach((track) => {
          track.enabled = audioEnabled;
        });
      }

      // If connected, replace track in the RTCPeerConnection
      if (peerRef.current && connected) {
        const senders = peerRef.current.getSenders();
        const sender = senders.find((s) => s.track?.kind === "video");
        if (sender) {
          sender.replaceTrack(newVideoTrack);
        }
      }

      setCurrentCamera(nextCamera.deviceId);
      setLocalStream(newStream);
      userStream.current = newStream;
    } catch (err) {
      console.error("Error switching camera:", err);
    }
  };

  // Hang up the call
  const hangUp = () => {
    router.push("/");
  };

  // Helper function to determine if we're showing local or remote video fullscreen
  const isLocalVideoMain = () => {
    if (!socketRef.current) return false;
    return mainVideo === socketRef.current.id;
  };

  // Helper function to determine if the remote video is main
  const isRemoteVideoMain = () => {
    if (!otherUser.current) return false;
    return mainVideo === otherUser.current;
  };

  // Replace toggleFullscreen function with showParticipantFullscreen
  const showParticipantFullscreen = (participantId: string | null) => {
    // Create a new timestamp for this event
    const timestamp = Date.now();

    // Update our local state
    setMainVideo(participantId);
    setFullscreenTimestamp(timestamp);

    // Notify all participants
    if (socketRef.current && connected) {
      socketRef.current.emit("fullscreen-change", {
        participantId: participantId,
        timestamp: timestamp,
        initiator: socketRef.current.id,
      });
    }
  };

  // Function to show local video in fullscreen for all participants
  const showMyVideoFullscreen = () => {
    if (socketRef.current) {
      showParticipantFullscreen(socketRef.current.id as string);
    }
  };

  // Function to show remote video in fullscreen for all participants
  const showRemoteVideoFullscreen = () => {
    if (otherUser.current) {
      showParticipantFullscreen(otherUser.current as string);
    }
  };

  // Function to exit fullscreen for all participants
  const exitFullscreen = () => {
    showParticipantFullscreen(null);
  };

  // Add a new useEffect for handling tracked point drawing
  useEffect(() => {
    const drawTrackingPoint = () => {
      if (!canvasRef.current || !videoContainerRef.current || !trackingPoint)
        return;

      const canvas = canvasRef.current;
      const container = videoContainerRef.current;
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      // Match canvas size to container
      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw tracking point as a red dot
      ctx.beginPath();
      ctx.arc(
        trackingPoint.x * canvas.width,
        trackingPoint.y * canvas.height,
        8,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = "red";
      ctx.fill();

      // Add outer white circle for visibility
      ctx.beginPath();
      ctx.arc(
        trackingPoint.x * canvas.width,
        trackingPoint.y * canvas.height,
        10,
        0,
        2 * Math.PI
      );
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.stroke();
    };

    drawTrackingPoint();

    // Redraw on window resize
    window.addEventListener("resize", drawTrackingPoint);
    return () => {
      window.removeEventListener("resize", drawTrackingPoint);
    };
  }, [trackingPoint, isTracking]);

  // Function to handle canvas click for point selection
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    console.log("Canvas clicked!", {
      canSelectPoint,
      isLocalMain: isLocalVideoMain(),
      hasCanvas: !!canvasRef.current,
    });

    if (!canSelectPoint) {
      console.log(
        "Point selection is disabled. Click the tracking button first to enable selection."
      );
      return;
    }

    if (!isLocalVideoMain()) {
      console.log("Cannot select point - local video must be fullscreen");
      return;
    }

    if (!canvasRef.current) {
      console.log("Canvas reference not available");
      return;
    }

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    // Calculate normalized coordinates (0-1) for cross-device compatibility
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    const newPoint = { x, y };
    console.log("Setting tracking point to:", newPoint);
    setTrackingPoint(newPoint);
    setIsTracking(true);

    // Send point to server for tracking
    if (socketRef.current && connected) {
      console.log("Sending tracking point to server");
      const trackingData = {
        point: newPoint,
        participantId: socketRef.current.id,
        roomId: roomID,
      };
      console.log("Tracking data:", trackingData);
      socketRef.current.emit("start-tracking", trackingData);
    } else {
      console.log("Cannot send tracking point - not connected to server", {
        socketExists: !!socketRef.current,
        connected,
      });
    }
  };

  // Function to stop tracking
  const stopTracking = () => {
    setTrackingPoint(null);
    setIsTracking(false);

    if (socketRef.current && connected) {
      socketRef.current.emit("stop-tracking", {
        roomId: roomID,
      });
    }
  };

  // Add a new useEffect to manage frame capture and transmission
  useEffect(() => {
    if (
      !isTracking ||
      !isLocalVideoMain() ||
      !socketRef.current ||
      !connected
    ) {
      // Stop frame capturing if not tracking or not the main video
      if (isCapturingFrames) {
        stopFrameCapture();
      }
      return;
    }

    // Start frame capture if not already capturing
    if (!isCapturingFrames) {
      startFrameCapture();
    }

    return () => {
      stopFrameCapture();
    };
  }, [isTracking, connected, mainVideo]);

  // Function to start capturing and sending frames
  const startFrameCapture = () => {
    if (isCapturingFrames || !videoContainerRef.current || !localStream) return;

    // Create canvas for frame capture if it doesn't exist
    if (!captureCanvasRef.current) {
      const canvas = document.createElement("canvas");
      captureCanvasRef.current = canvas;
    }

    setIsCapturingFrames(true);

    // Function to capture and send frame
    const captureAndSendFrame = () => {
      try {
        if (!localStream || !captureCanvasRef.current || !socketRef.current)
          return;

        const videoTrack = localStream.getVideoTracks()[0];
        if (!videoTrack) return;

        // Get video dimensions
        const trackSettings = videoTrack.getSettings();
        const width = trackSettings.width || 640;
        const height = trackSettings.height || 480;

        // Set canvas dimensions
        const canvas = captureCanvasRef.current;
        canvas.width = width;
        canvas.height = height;

        // Create video element to draw from stream
        const video = document.createElement("video");
        video.srcObject = localStream;
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;

        // Wait for video to be ready
        video.onloadedmetadata = () => {
          const ctx = canvas.getContext("2d");
          if (!ctx) return;

          // Draw video frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Convert canvas to base64 image
          const frameData = canvas.toDataURL("image/jpeg", 0.7);

          // Send frame to server
          if (socketRef.current) {
            socketRef.current.emit("video-frame", {
              frame: frameData,
              roomId: roomID,
            });
          }

          // Clean up
          video.srcObject = null;
        };
      } catch (error) {
        console.error("Error capturing frame:", error);
      }
    };

    // Capture and send frame immediately
    captureAndSendFrame();

    // Set interval to capture and send frames (15fps = ~66.7ms)
    frameCapturerRef.current = window.setInterval(captureAndSendFrame, 67);
  };

  // Function to stop frame capture
  const stopFrameCapture = () => {
    if (frameCapturerRef.current) {
      clearInterval(frameCapturerRef.current);
      frameCapturerRef.current = null;
    }
    setIsCapturingFrames(false);
  };

  // Update the togglePointSelection function to start/stop frame capture
  const togglePointSelection = () => {
    console.log("Toggle point selection");

    if (!isLocalVideoMain()) {
      console.log(
        "Cannot toggle point selection - local video must be fullscreen"
      );
      alert(
        "Your video must be fullscreen to select tracking points. Click on your video first."
      );
      return;
    }

    const newState = !canSelectPoint;
    console.log(`Setting canSelectPoint to ${newState}`);
    setCanSelectPoint(newState);

    if (newState) {
      console.log(
        "Point selection enabled - click anywhere on the video to set a tracking point"
      );
    } else {
      console.log("Point selection disabled");
    }

    if (isTracking) {
      stopTracking();
    }
  };

  // Add a global click handler to debug clicks
  useEffect(() => {
    const handleGlobalClick = (e: MouseEvent) => {
      console.log("Global click detected!", {
        target: e.target,
        x: e.clientX,
        y: e.clientY,
        element: e.target instanceof Element ? e.target.tagName : "unknown",
      });
    };

    window.addEventListener("click", handleGlobalClick);

    return () => {
      window.removeEventListener("click", handleGlobalClick);
    };
  }, []);

  return (
    <div className="fixed inset-0 flex flex-col items-center w-full h-screen">
      {/* Video container - always full screen */}
      <div
        className="absolute inset-0 w-full h-full z-0"
        onClick={(e) => {
          console.log("Video container clicked");
          // Only handle clicks when in point selection mode and local video is main
          if (canSelectPoint && isLocalVideoMain() && canvasRef.current) {
            console.log(
              "Container click in selection mode - forwarding to canvas"
            );
            const canvas = canvasRef.current;
            const rect = canvas.getBoundingClientRect();

            // Calculate normalized coordinates (0-1) for cross-device compatibility
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            const newPoint = { x, y };
            console.log(
              "Setting tracking point from container click:",
              newPoint
            );
            setTrackingPoint(newPoint);
            setIsTracking(true);

            // Send point to server for tracking
            if (socketRef.current && connected) {
              console.log("Sending tracking point to server");
              const trackingData = {
                point: newPoint,
                participantId: socketRef.current.id,
                roomId: roomID,
              };
              console.log("Tracking data:", trackingData);
              socketRef.current.emit("start-tracking", trackingData);
            }

            // Disable point selection mode after setting a point
            setCanSelectPoint(false);
            return;
          }

          // Default behavior for regular clicks (exit fullscreen)
          if (mainVideo) {
            exitFullscreen();
          }
        }}
        ref={videoContainerRef}
      >
        {/* Determine which video should be shown fullscreen */}
        {connected &&
          remoteStream &&
          (isRemoteVideoMain() ? (
            // Remote video in fullscreen when remote participant is main
            <VideoPlayer
              stream={remoteStream}
              label="Remote"
              isClickable={true}
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                exitFullscreen();
              }}
            />
          ) : isLocalVideoMain() ? (
            // Local video in fullscreen when local participant is main
            <VideoPlayer
              stream={localStream}
              muted={true}
              label="You"
              isClickable={true}
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                exitFullscreen();
              }}
            />
          ) : (
            // Default view when no specific main video
            <VideoPlayer
              stream={remoteStream}
              label="Remote"
              isClickable={true}
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                showRemoteVideoFullscreen();
              }}
            />
          ))}

        {/* When not connected, show local video fullscreen */}
        {(!connected || !remoteStream) && (
          <VideoPlayer
            stream={localStream}
            muted={true}
            label="You"
            isClickable={connected}
            onClick={
              connected
                ? (e: React.MouseEvent) => {
                    e.stopPropagation();
                    showMyVideoFullscreen();
                  }
                : undefined
            }
          />
        )}

        {/* Canvas overlay for tracking points */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full z-20 pointer-events-auto"
          onClick={(e) => {
            e.stopPropagation(); // Prevent clicks from reaching elements below
            console.log("Direct canvas click detected!");
            handleCanvasClick(e);
          }}
          style={{
            cursor: canSelectPoint ? "crosshair" : "default",
          }}
        />

        {/* Transparent overlay for point selection - only visible when selecting points */}
        {canSelectPoint && isLocalVideoMain() && (
          <div
            className="absolute inset-0 w-full h-full z-50 bg-transparent cursor-crosshair"
            onClick={(e) => {
              e.stopPropagation();
              console.log("Point selection overlay clicked");

              const rect = e.currentTarget.getBoundingClientRect();
              // Calculate normalized coordinates (0-1) for cross-device compatibility
              const x = (e.clientX - rect.left) / rect.width;
              const y = (e.clientY - rect.top) / rect.height;

              const newPoint = { x, y };
              console.log("Setting tracking point from overlay:", newPoint);
              setTrackingPoint(newPoint);
              setIsTracking(true);

              // Send point to server for tracking
              if (socketRef.current && connected) {
                console.log("Sending tracking point to server");
                const trackingData = {
                  point: newPoint,
                  participantId: socketRef.current.id,
                  roomId: roomID,
                };
                console.log("Tracking data:", trackingData);
                socketRef.current.emit("start-tracking", trackingData);
              }

              // Disable point selection mode after setting a point
              setCanSelectPoint(false);
            }}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-yellow-500 bg-opacity-70 text-white px-4 py-2 rounded-lg text-lg font-bold animate-pulse">
                Click anywhere to set tracking point
              </div>
            </div>
          </div>
        )}
      </div>

      {/* UI overlay - positioned above the videos */}
      <div className="relative z-10 w-full px-4 py-6 flex flex-col h-full">
        <h1 className="text-2xl font-bold mb-4 text-center text-white drop-shadow-lg">
          Video Conference
          {mainVideo && socketRef.current && (
            <span className="ml-2 text-sm bg-blue-500 px-2 py-1 rounded-full">
              {mainVideo === socketRef.current.id ? "Your" : "Remote"} camera
              fullscreen
            </span>
          )}
        </h1>

        {/* Tracking mode indicator */}
        {canSelectPoint && (
          <div className="absolute top-16 left-1/2 transform -translate-x-1/2 bg-yellow-500 text-white px-3 py-1 rounded-full text-sm">
            Click anywhere on the video to select a tracking point
          </div>
        )}

        {/* Tracking controls */}
        {isLocalVideoMain() && (
          <div className="absolute top-24 left-4 z-30 flex flex-col items-center">
            <button
              onClick={togglePointSelection}
              className={`p-3 rounded-full flex items-center justify-center ${
                canSelectPoint
                  ? "bg-yellow-500 text-white animate-pulse"
                  : "bg-blue-500 text-white hover:bg-blue-600"
              }`}
              title={
                canSelectPoint ? "Cancel selection" : "Select tracking point"
              }
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.042 21.672 13.684 16.6m0 0-2.51 2.225.569-9.47 5.227 7.917-3.286-.672Zm-7.518-.267A8.25 8.25 0 1 1 20.25 10.5M8.288 14.212A5.25 5.25 0 1 1 17.25 10.5"
                />
              </svg>
              <span className="ml-2 hidden sm:inline">
                {canSelectPoint ? "Cancel" : "Track Point"}
              </span>
            </button>

            {canSelectPoint && (
              <div className="mt-2 bg-yellow-500 text-white px-2 py-1 rounded text-xs text-center">
                Click on video to set tracking point
              </div>
            )}

            {isTracking && (
              <button
                onClick={stopTracking}
                className="mt-2 p-3 rounded-full bg-red-500 text-white hover:bg-red-600 flex items-center"
                title="Stop tracking"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-6 h-6"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18 18 6M6 6l12 12"
                  />
                </svg>
                <span className="ml-2 hidden sm:inline">Stop Tracking</span>
              </button>
            )}
          </div>
        )}

        {/* Display appropriate PiP based on fullscreen state */}
        {connected && (
          <>
            {/* Show local video PiP when remote is fullscreen or no fullscreen */}
            {(isRemoteVideoMain() || !mainVideo) && (
              <div
                className="absolute bottom-4 right-4 max-w-1/8 z-20 shadow-xl rounded-lg overflow-hidden"
                ref={localVideoFrameRef}
              >
                <VideoPlayer
                  stream={localStream}
                  muted={true}
                  label="You"
                  isClickable={true}
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation();
                    showMyVideoFullscreen();
                  }}
                />
              </div>
            )}

            {/* Show remote video PiP when local is fullscreen */}
            {isLocalVideoMain() && remoteStream && (
              <div className="absolute bottom-4 right-4 max-w-1/8 z-20 shadow-xl rounded-lg overflow-hidden">
                <VideoPlayer
                  stream={remoteStream}
                  label="Remote"
                  isClickable={true}
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation();
                    showRemoteVideoFullscreen();
                  }}
                />
              </div>
            )}
          </>
        )}

        {/* Waiting message */}
        {waiting && (
          <div className="mb-6 p-4 bg-white bg-opacity-80 backdrop-blur-sm rounded-lg shadow">
            <p className="mb-2">Waiting for someone to join...</p>
            <p className="mb-2">Share this link to invite someone:</p>
            <div className="flex items-center">
              <input
                type="text"
                value={roomLink}
                readOnly
                className="flex-1 p-2 border rounded-l-lg bg-gray-50"
              />
              <button
                onClick={() => {
                  navigator.clipboard.writeText(roomLink);
                  alert("Link copied to clipboard!");
                }}
                className="bg-blue-600 text-white px-4 py-2 rounded-r-lg"
              >
                Copy
              </button>
            </div>
          </div>
        )}

        {/* Spacer to push controls to bottom */}
        <div className="flex-grow"></div>

        {/* Control bar at bottom */}
        <div className="mt-auto mb-4">
          <ControlBar
            audioEnabled={audioEnabled}
            videoEnabled={videoEnabled}
            toggleAudio={toggleAudio}
            toggleVideo={toggleVideo}
            switchCamera={switchCamera}
            hangUp={hangUp}
            canSwitchCamera={availableCameras.length > 1}
            connected={connected}
          />

          <div className="mt-4 text-center">
            <Link
              href="/"
              className="text-white hover:underline drop-shadow-lg"
            >
              Return to home
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
