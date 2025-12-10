#!/usr/bin/env python3
"""
Animation Replay System - Replay recorded air hockey matches.

Usage:
    python replay_match.py <animation_file>
    
The animation file is saved automatically during simulation runs.
"""

import argparse
import pickle
from pathlib import Path
from pydrake.all import StartMeshcat
import time


def replay_animation(animation_file: str):
    """
    Replay a recorded meshcat animation.
    
    Args:
        animation_file: Path to saved animation file (.pkl)
    """
    animation_path = Path(animation_file)
    
    if not animation_path.exists():
        print(f"Error: Animation file not found: {animation_file}")
        return
    
    print(f"\nLoading animation from: {animation_file}")
    
    # Load animation
    with open(animation_path, 'rb') as f:
        animation_data = pickle.load(f)
    
    # Start meshcat
    print("Starting Meshcat visualizer...")
    meshcat = StartMeshcat()
    
    print(f"\nMeshcat URL: {meshcat.web_url()}")
    print("Open this URL in your browser to view the replay\n")
    
    time.sleep(2)  # Give time to open browser
    
    # Set the animation
    meshcat.SetAnimation(animation_data)
    
    print("="*70)
    print("REPLAY CONTROLS")
    print("="*70)
    print("The animation is now loaded in Meshcat.")
    print("Use the Meshcat timeline controls to:")
    print("  - Play/Pause")
    print("  - Scrub through the timeline")
    print("  - Adjust playback speed")
    print("="*70)
    
    print("\nPress Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting replay...")


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded air hockey match animations"
    )
    parser.add_argument(
        "animation_file",
        type=str,
        help="Path to animation file (.pkl)"
    )
    
    args = parser.parse_args()
    replay_animation(args.animation_file)


if __name__ == "__main__":
    main()
