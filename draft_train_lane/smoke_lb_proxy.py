#!/usr/bin/env python3
"""Smoke LB: plain TCP proxy FROM_PORT -> 127.0.0.1:TO_PORT (env).

Stands in for the sglang PD router in SMOKE-mode launcher dry-runs (the real
router needs a prefill/decode pair). Exercises the same launcher wiring:
srun-launched process, health-gated, torn down by pid.
"""
import os
import socket
import threading


def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    finally:
        try:
            a.close()
            b.close()
        except OSError:
            pass


def main():
    frm = int(os.environ["FROM_PORT"])
    to = int(os.environ["TO_PORT"])
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", frm))
    s.listen(64)
    print(f"smoke LB proxy {frm} -> 127.0.0.1:{to}", flush=True)
    while True:
        c, _ = s.accept()
        u = socket.socket()
        u.connect(("127.0.0.1", to))
        threading.Thread(target=pipe, args=(c, u), daemon=True).start()
        threading.Thread(target=pipe, args=(u, c), daemon=True).start()


if __name__ == "__main__":
    main()
