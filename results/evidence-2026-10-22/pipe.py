"""The pod's end of bridge.py: llama-server starts this as an MCP server, and it passes stdin and
stdout to the Mac's port that `ssh -R` forwards here.

    python3 pipe.py PORT
"""

import socket
import sys
import threading

connection = socket.create_connection(("127.0.0.1", int(sys.argv[1])))


def up():
    while data := sys.stdin.buffer.read1(65536):
        connection.sendall(data)
    connection.shutdown(socket.SHUT_WR)


threading.Thread(target=up, daemon=True).start()
while data := connection.recv(65536):
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
