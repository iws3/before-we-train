"""Download the public-domain text used by 1.2.4 and 1.3.5.

    python data/get_data.py

Saves data/raw/tinyshakespeare.txt (about 1.1 MB, Shakespeare, public domain).
No network? Put ANY plain-text file anywhere and pass it to the scripts with
--text path/to/file.txt. A text in your own language works too.
"""
import os
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "tinyshakespeare.txt")


def main() -> int:
    if os.path.exists(DEST):
        print(f"already downloaded: {DEST}")
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"download failed: {exc}\nUse --text with your own file instead.")
        return 1
    print(f"saved {os.path.getsize(DEST):,} bytes to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
