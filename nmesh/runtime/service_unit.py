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
    current = os.name if os_name is None else os_name
    executable = sys.executable
    if current == "nt":
        task_command = subprocess.list2cmdline(
            (executable, "-m", "nmesh.gateway.server", "--port", str(port))
        )
        command = (
            f'schtasks /create /tn nmesh-gateway /sc onlogon '
            f'/tr "{task_command}"'
        )
        return "nmesh-gateway.cmd", command + "\n", command
    if current == "darwin":
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


def linux_service_unit(port: int = 18000) -> tuple[str, str, str]:
    return service_unit(port, "posix")


def macos_service_unit(port: int = 18000) -> tuple[str, str, str]:
    return service_unit(port, "darwin")


def windows_service_unit(port: int = 18000) -> tuple[str, str, str]:
    return service_unit(port, "nt")


linux_unit = linux_service_unit
macos_unit = macos_service_unit
windows_unit = windows_service_unit
