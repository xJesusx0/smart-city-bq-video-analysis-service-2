#!/usr/bin/env python3
import sys
import os

# Add the current directory to python path for convenience
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_analysis.main import main

if __name__ == "__main__":
    main()
