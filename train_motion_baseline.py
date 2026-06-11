"""Compatibility entrypoint for the motion baseline package.

The implementation lives in the motion_baseline/ folder so the code can be
kept in smaller, easier-to-read modules.
"""

from motion_baseline.cli import main


if __name__ == "__main__":
    main()
