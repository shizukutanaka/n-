from __future__ import annotations

import html
import os
import shlex
import subprocess
import sys


def service_unit(
    port: int = 18000, os_name: str | None = None
) -> tuple[str, str, str]:
    """Return the filename, unit text, and explicit install command."""
    current = os_name
    executable = sys.executable
    windows = os.name == "nt" if current is None else current == "nt"
    macos = (
        sys.platform.startswith("darwin")
        if current is None
        else current.startswith("darwin")
    )
    if windows:
        task_command = subprocess.list2cmdline(
            (executable, "-m", "nmesh.gateway.server", "--port", str(port))
        )
        command = (
            f'schtasks /create /tn nmesh-gateway /sc onlogon '
            f'/tr "{task_command}"'
        )
        return "nmesh-gateway.cmd", command + "\n", command
    if macos:
        label = "com.nmesh.gateway"
        text = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array><string>{html.escape(executable)}</string>
    <string>-m</string><string>nmesh.gateway.server</string>
    <string>--port</string><string>{port}</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"""
        return "com.nmesh.gateway.plist", text, f"launchctl load ~/Library/LaunchAgents/{label}.plist"
    text = f"""[Unit]
Description=nmesh gateway
After=network.target

[Service]
ExecStart={shlex.join((executable, "-m", "nmesh.gateway.server", "--port", str(port)))}
Restart=on-failure

[Install]
WantedBy=default.target
"""
    return "nmesh-gateway.service", text, "systemctl --user enable --now ~/.config/systemd/user/nmesh-gateway.service"
