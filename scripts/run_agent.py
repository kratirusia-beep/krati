"""
Multimodal Attendance & Engagement Agent - Primary Runner.
Supports launching either the Web Intelligence Dashboard or CLI headless mode.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.web.app import create_app
from src.pipeline.classroom_agent import ClassroomAgent


def main():
    parser = argparse.ArgumentParser(description="Multimodal Attendance & Engagement Agent")
    parser.add_argument("--mode", choices=["web", "cli"], default="web", help="Execution mode: web dashboard or cli monitor")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live hardware capture (default 0)")
    parser.add_argument("--sim", action="store_true", default=True, help="Use classroom simulator (default True)")
    parser.add_argument("--no-sim", dest="sim", action="store_false", help="Use live hardware camera instead of simulator")

    args = parser.parse_args()

    print("=" * 70)
    print("[AGENT] MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT")
    print("        AI-Powered Privacy-Preserving Classroom Intelligence System")
    print("        100% Offline Edge Execution")
    print("=" * 70)

    if args.mode == "web":
        print(f"\n[WEB] Starting Web Control Center on http://{args.host}:{args.port}")
        print("      Press Ctrl+C to terminate.")
        app = create_app()
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    else:
        print(f"\n[CAMERA] Starting CLI Classroom Agent Monitor (Simulator={args.sim})...")
        agent = ClassroomAgent(video_source=args.camera, enable_microphone=True, use_simulator=args.sim)
        agent.start()

        import time
        try:
            while True:
                time.sleep(2.0)
                payload = agent.get_dashboard_payload()
                if payload.get("status") == "initializing":
                    continue
                print(f"[{payload['timestamp']}] Engagement: {payload['overall_engagement']}% | "
                      f"Active: {payload['active_students']} | "
                      f"Audio: {payload['audio']['environment']} | "
                      f"Present: {payload['attendance']['present_count']}/{payload['attendance']['enrolled_total']} | "
                      f"FPS: {payload['fps']}")
                for sug in payload.get("suggestions", [])[:1]:
                    print(f"   [SUGGESTION] [{sug['urgency']}] {sug['title']}: {sug['action_prompt']}")
        except KeyboardInterrupt:
            print("\nShutting down Classroom Agent...")
            agent.stop()


if __name__ == "__main__":
    main()
