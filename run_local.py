"""Start the API locally.

    python run_local.py                # http://127.0.0.1:8000
    python run_local.py --reload        # auto-reload on code changes
    python run_local.py --tunnel        # also open a Cloudflare quick tunnel

The tunnel is OPTIONAL and off by default. In Colab it was mandatory because the
runtime is not reachable on localhost; on your own laptop it is only needed when
you want to hand someone a public URL for a demo. It requires the `cloudflared`
binary on PATH (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
"""

import argparse
import re
import subprocess
import time

import uvicorn

from app.config import API_HOST, API_PORT


def start_tunnel(port):
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError:
        print("cloudflared not found on PATH - skipping tunnel. "
              "Install it or drop the --tunnel flag.")
        return None

    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            print(f"\nPublic tunnel URL: {match.group(0)}\n")
            return proc
    print("Tunnel started but no URL appeared within 30s; check cloudflared output.")
    return proc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--tunnel", action="store_true",
                        help="expose the API via a Cloudflare quick tunnel")
    args = parser.parse_args()

    if args.tunnel:
        start_tunnel(args.port)

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
