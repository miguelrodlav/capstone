"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function Home() {
  const [roomID, setRoomID] = useState("");
  const router = useRouter();

  const joinRoom = (e: React.FormEvent) => {
    e.preventDefault();
    if (roomID.trim()) {
      router.push(`/conference?room=${roomID}`);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-white p-8 rounded-lg shadow-md">
        <h1 className="text-3xl font-bold mb-6 text-center">
          WebRTC Video Conference
        </h1>

        <div className="space-y-6">
          <div>
            <Link
              href="/conference"
              className="w-full py-3 px-4 flex justify-center items-center bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Create New Room
            </Link>
          </div>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center">
              <span className="px-2 bg-white text-gray-500">or</span>
            </div>
          </div>

          <form onSubmit={joinRoom} className="space-y-4">
            <div>
              <label
                htmlFor="roomID"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Join Existing Room
              </label>
              <input
                type="text"
                id="roomID"
                value={roomID}
                onChange={(e) => setRoomID(e.target.value)}
                placeholder="Enter Room ID"
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <button
              type="submit"
              disabled={!roomID.trim()}
              className="w-full py-3 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-green-300"
            >
              Join Room
            </button>
          </form>
        </div>

        <div className="mt-8 text-center text-sm text-gray-500">
          <p>One-to-one video conference with camera and audio controls</p>
        </div>
      </div>
    </main>
  );
}
