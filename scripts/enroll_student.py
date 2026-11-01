"""
CLI Student Enrollment Utility.
Epic 2: Student Enrollment & Identity Registry (Task 3 & 4)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.enrollment.student_registry import StudentRegistry


def main():
    parser = argparse.ArgumentParser(description="Enroll student into offline identity registry")
    parser.add_argument("--id", required=True, help="Student ID (e.g. STU-107)")
    parser.add_argument("--name", required=True, help="Full Name")
    parser.add_argument("--roll", default="", help="Roll Number")
    parser.add_argument("--dept", default="Computer Science", help="Department")
    parser.add_argument("--image", default="", help="Path to reference face image file (optional)")
    parser.add_argument("--camera", action="store_true", help="Capture reference photo from webcam")

    args = parser.parse_args()
    registry = StudentRegistry()

    face_crop = None
    if args.image:
        face_crop = cv2.imread(args.image)
        if face_crop is None:
            print(f"[Error] Could not read image from {args.image}")
            sys.exit(1)
    elif args.camera:
        print("[Info] Opening webcam to capture reference photo. Press SPACE to capture, ESC to cancel.")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Capture Reference Face - Press SPACE", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                face_crop = frame
                break
            elif key == 27:  # ESC
                break
        cap.release()
        cv2.destroyAllWindows()

    if face_crop is None:
        print("[Info] No photo provided. Generating synthetic biometric reference.")
        face_crop = np.random.randint(60, 210, (64, 64, 3), dtype=np.uint8)
        cv2.circle(face_crop, (32, 32), 20, (200, 200, 200), -1)

    profile = registry.enroll(args.id, args.name, args.roll, args.dept, face_crop)
    print(f"\n[SUCCESS] Student enrolled successfully:")
    print(f"   ID: {profile.student_id}")
    print(f"   Name: {profile.name}")
    print(f"   Roll No: {profile.roll_no}")
    print(f"   Department: {profile.department}")
    print(f"   Biometric SHA-256 Hash: {profile.embedding_hash[:16]}...")
    print(f"   Enrolled At: {profile.enrolled_at}")


if __name__ == "__main__":
    main()
