"use client";

import { useEffect, useRef } from "react";

interface VideoPlayerProps {
  stream: MediaStream | null;
  muted?: boolean;
  label?: string;
  onClick?: (e: React.MouseEvent) => void;
  isClickable?: boolean;
}

export default function VideoPlayer({
  stream,
  muted = false,
  label,
  onClick,
  isClickable = false,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <div
      className={`relative bg-gray-800 rounded-lg overflow-hidden aspect-video shadow-lg ${
        isClickable
          ? "cursor-pointer hover:ring-2 hover:ring-blue-400 transition-all"
          : ""
      }`}
      onClick={onClick}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted={muted}
        className="w-full h-full object-cover"
      />
      {label && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
          {label}
        </div>
      )}
      {isClickable && (
        <div className="absolute top-2 right-2 bg-black bg-opacity-50 text-white p-1 rounded-full">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5-5-5m5 5v-4m0 4h-4"
            />
          </svg>
        </div>
      )}
      {!stream && (
        <div className="absolute inset-0 flex items-center justify-center text-white">
          <p>No video available</p>
        </div>
      )}
    </div>
  );
}
