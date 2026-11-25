"""Helpers for reading recording scales from filenames and metadata files."""

import glob
import os
import re
from optparse import OptionParser

import yaml

__all__ = ["parse_filename", "guess_fps_and_pixelsize"]


def parse_filename(filename):
    """Extract FPS and pixel size from names such as ``movie_12_fps_1.34_um``."""
    match = re.search(
        r"(\d+(?:\.\d+)?)[^\d]*fps.*?(\d+(?:\.\d+)?)[^\d]*um",
        os.fspath(filename),
        re.IGNORECASE,
    )
    if match is None:
        return None, None
    return float(match.group(1)), float(match.group(2))


def guess_fps_and_pixelsize(path=None):
    """Look for FPS and pixel size in command-line options, paths, or YAML files."""
    fps = pixel_size = None

    parser = OptionParser()
    parser.add_option("--fps", dest="fps", help="frame rate (Hz)", default=None)
    parser.add_option("--framerate", dest="fps", help="frame rate (Hz)", default=None)
    parser.add_option("--pixelsize", dest="pixel_size", help="pixel size (um)", default=None)
    parser.add_option("--pixel_size", dest="pixel_size", help="pixel size (um)", default=None)
    options, args = parser.parse_args()

    if args:
        fps, pixel_size = parse_filename(args[0])

    if path is not None:
        fps_from_path, pixel_size_from_path = parse_filename(path)
        fps = fps or fps_from_path
        pixel_size = pixel_size or pixel_size_from_path

        if fps is None or pixel_size is None:
            search_path = os.fspath(path)
            if not os.path.isdir(search_path):
                search_path = os.path.dirname(search_path)
            description_files = glob.glob(os.path.join(search_path, "*.yaml"))
            description_files += glob.glob(
                os.path.join(os.path.dirname(search_path), "*.yaml")
            )

            for description_file in description_files:
                with open(description_file, encoding="utf-8") as file:
                    description = yaml.safe_load(file)
                if not isinstance(description, dict):
                    continue
                fps = fps or description.get("framerate") or description.get("fps")
                if "scaled" in description:
                    pixel_size = 1
                else:
                    pixel_size = (
                        pixel_size
                        or description.get("pixel_size")
                        or description.get("pixelsize")
                        or description.get("pixel_width")
                    )

    if options.fps is not None:
        fps = fps or float(options.fps)
    if options.pixel_size is not None:
        pixel_size = pixel_size or float(options.pixel_size)

    return fps, pixel_size
