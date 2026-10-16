#!/usr/bin/env python3
"""Download source videos (and their Chinese subtitle track, if available) for local use only.

the project design log (ethical/practical safeguards): full videos are for local analysis
only and must not be redistributed. This script only ever writes to paths.raw_videos_dir,
which should stay out of version control and out of any shared paper appendix.

Usage:
    python scripts/01_fetch_source_videos.py --url https://youtu.be/XXXXXXXX --video-id vid01
    python scripts/01_fetch_source_videos.py --url-list data/sources.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, ensure_dir, get_logger

log = get_logger("fetch_source_videos")


def fetch_one(url, video_id, out_dir, cookies=None):
    ensure_dir(out_dir)
    dst_template = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bv*[height<=720]+ba/b[height<=720]",
        "--merge-output-format", "mp4",
        "--write-sub", "--write-auto-sub", "--sub-lang", "zh-Hant,zh-TW,zh",
        "--convert-subs", "srt",
        "-o", dst_template,
        url,
    ]
    if cookies:
        cmd[1:1] = ["--cookies", cookies]
    log.info("Fetching %s -> %s", url, video_id)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="single video URL")
    group.add_argument("--url-list", help="text file: one 'video_id,url' pair per line")
    parser.add_argument("--video-id", help="required when using --url")
    parser.add_argument("--cookies", default=None,
                         help="cookies.txt (Netscape format) exported from a browser logged "
                              "into YouTube -- needed if YouTube responds with a bot-check "
                              "('Sign in to confirm you're not a bot') from this server's IP")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["paths"]["raw_videos_dir"]

    if args.url:
        if not args.video_id:
            parser.error("--video-id is required when using --url")
        fetch_one(args.url, args.video_id, out_dir, cookies=args.cookies)
    else:
        with open(args.url_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("video_id,"):
                    continue
                video_id, url = [p.strip() for p in line.split(",", 1)]
                fetch_one(url, video_id, out_dir, cookies=args.cookies)

    log.info(
        "Done. Reminder (project design log): do not redistribute full videos; "
        "store only timestamps + derived annotations in anything shared."
    )


if __name__ == "__main__":
    main()
