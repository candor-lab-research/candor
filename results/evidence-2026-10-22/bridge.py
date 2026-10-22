"""Serves one of the Mac's stdio MCP servers on a local port, a fresh process per connection, so a
llama-server on a pod can reach it through `ssh -R`. The tools run here, exactly as they do locally.

    python3 bridge.py PORT DIR COMMAND...
"""

import asyncio
import subprocess
import sys

PORT = int(sys.argv[1])
DIR = sys.argv[2]
COMMAND = sys.argv[3:]


async def handle(reader, writer):
    server = await asyncio.create_subprocess_exec(
        *COMMAND, cwd=DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    print(f"{PORT}: connected, pid {server.pid}", flush=True)

    async def up():
        try:
            while data := await reader.read(65536):
                server.stdin.write(data)
                await server.stdin.drain()
        except (ConnectionError, BrokenPipeError):
            pass
        server.stdin.close()

    async def down():
        while data := await server.stdout.read(65536):
            writer.write(data)
            await writer.drain()
        writer.close()

    try:
        await asyncio.gather(up(), down())
    except ConnectionError:
        server.terminate()
    print(f"{PORT}: closed, exit {await server.wait()}", flush=True)


async def main():
    async with await asyncio.start_server(handle, "127.0.0.1", PORT) as listening:
        print(f"{PORT}: listening", flush=True)
        await listening.serve_forever()


asyncio.run(main())
