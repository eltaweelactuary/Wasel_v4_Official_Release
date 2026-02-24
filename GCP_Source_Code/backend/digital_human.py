"""
Wasel v4 Pro: Digital Human Renderer
Migrated and enhanced from v3's DigitalHumanRenderer.
Generates stylized skeletal avatar videos from landmark DNA.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DigitalHumanRenderer:
    """
    Synthesizes a high-fidelity Digital Human Avatar from skeletal landmarks.
    Supports both YOLO (17 keypoints) and MediaPipe (33+ keypoints) formats.
    """

    # ─── COCO Skeleton Connections (for YOLO 17-keypoint format) ───
    COCO_SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # Head
        (5, 6),                                   # Shoulders
        (5, 7), (7, 9),                           # Left arm
        (6, 8), (8, 10),                          # Right arm
        (5, 11), (6, 12),                         # Torso
        (11, 12),                                  # Hips
        (11, 13), (13, 15),                       # Left leg
        (12, 14), (14, 16),                       # Right leg
    ]

    # ─── Color Palette (Elite Studio v4 Theme) ───
    COLORS = {
        "head": (56, 189, 248),      # Sky blue
        "torso": (139, 92, 246),     # Purple
        "left_arm": (34, 197, 94),   # Green
        "right_arm": (251, 146, 60), # Orange
        "left_leg": (59, 130, 246),  # Blue
        "right_leg": (244, 63, 94),  # Rose
        "joint": (255, 255, 255),    # White
    }

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    def _get_limb_color(self, i1: int, i2: int):
        """Assign color based on body part."""
        if i1 <= 4 or i2 <= 4:
            return self.COLORS["head"]
        if i1 in (5, 7, 9):
            return self.COLORS["left_arm"]
        if i1 in (6, 8, 10):
            return self.COLORS["right_arm"]
        if {i1, i2} & {11, 13, 15}:
            return self.COLORS["left_leg"]
        if {i1, i2} & {12, 14, 16}:
            return self.COLORS["right_leg"]
        return self.COLORS["torso"]

    def render_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Render a single frame from keypoints.
        
        Args:
            keypoints: Flattened array of (x, y, conf) per joint.
                       YOLO: 51 values (17 joints × 3)
                       MediaPipe: 225+ values
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(self.height):
            ratio = y / self.height
            canvas[y, :] = [int(15 + 10 * ratio), int(15 + 5 * ratio), int(30 + 15 * ratio)]

        n = len(keypoints)

        if n >= 51:  # YOLO format (17 joints × 3)
            joints = keypoints.reshape(-1, 3)[:17]  # (17, 3)
            
            # Scale to canvas
            points = []
            for x, y, conf in joints:
                px = int(x * self.width) if x <= 1.0 else int(x)
                py = int(y * self.height) if y <= 1.0 else int(y)
                points.append((px, py, conf))
            
            # Draw skeleton
            for i1, i2 in self.COCO_SKELETON:
                if points[i1][2] > 0.3 and points[i2][2] > 0.3:
                    color = self._get_limb_color(i1, i2)
                    cv2.line(canvas, points[i1][:2], points[i2][:2], color, 3, cv2.LINE_AA)
            
            # Draw joints
            for px, py, conf in points:
                if conf > 0.3:
                    cv2.circle(canvas, (px, py), 6, self.COLORS["joint"], -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 8, (100, 100, 100), 1, cv2.LINE_AA)
        
        elif n >= 225:  # MediaPipe format (v3 compat)
            # Draw hands (first 126 features = 42 joints × 3)
            for hand_start, color in [(0, self.COLORS["left_arm"]), (63, self.COLORS["right_arm"])]:
                for i in range(21):
                    idx = hand_start + i * 3
                    if idx + 2 < n:
                        x = int((keypoints[idx] + 0.5) * self.width)
                        y = int((keypoints[idx + 1] + 0.5) * self.height)
                        if 0 < x < self.width and 0 < y < self.height:
                            cv2.circle(canvas, (x, y), 4, color, -1)
            
            # Draw pose (next 99 features = 33 joints × 3)
            pose_start = 126
            for i in range(33):
                idx = pose_start + i * 3
                if idx + 2 < n:
                    x = int((keypoints[idx] + 0.5) * self.width)
                    y = int((keypoints[idx + 1] + 0.5) * self.height)
                    if 0 < x < self.width and 0 < y < self.height:
                        cv2.circle(canvas, (x, y), 5, self.COLORS["torso"], -1)
        
        return canvas

    def render_video(self, sequence: np.ndarray, output_path: str, fps: int = 30) -> str:
        """
        Render a full video from a landmark sequence.
        
        Args:
            sequence: (N_frames, N_features) array
            output_path: Path to save the output .mp4
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        for frame_data in sequence:
            canvas = self.render_frame(frame_data)
            out.write(canvas)
        
        out.release()
        logger.info(f"✅ Avatar video saved: {output_path} ({len(sequence)} frames)")
        return output_path

    def stitch_and_render(self, dna_list: List[np.ndarray], output_path: str,
                          transition_frames: int = 10, fps: int = 30) -> str:
        """
        Stitch multiple word DNAs into a sentence video with smooth transitions.
        """
        if not dna_list:
            return output_path
        
        full_sequence = []
        for i, dna in enumerate(dna_list):
            full_sequence.extend(dna.tolist())
            
            # Add transition (interpolation) between words
            if i < len(dna_list) - 1:
                last_frame = dna[-1]
                next_frame = dna_list[i + 1][0]
                for t in range(transition_frames):
                    alpha = t / transition_frames
                    interp = last_frame * (1 - alpha) + next_frame * alpha
                    full_sequence.append(interp.tolist())
        
        return self.render_video(np.array(full_sequence), output_path, fps)
