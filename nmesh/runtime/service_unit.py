from __future__ import annotations

import html
import os
import shlex
import subprocess
import sys
from pathlib import Path

from nmesh.paths import nmesh_home


def launcher_script(
    port: int = 18000, os_name: str | None = None
) -> tuple[str, str]:
    """Return a launcher filename and script that carries gateway.env."""
    windows = os.name == "nt" if os_name is None else os_name == "nt"
    if windows:
        filename = "nmesh-gateway-launcher.cmd"
        executable = subprocess.list2cmdline((sys.executable,))
        text = f"""@echo off
setlocal
if exist "%~dp0gateway.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0gateway.env") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)
set "NMESH_HOME=%~dp0"
{executable} -m nmesh.gateway.server --port {port}
"""
        return filename, text.replace("\n", "\r\n")
    filename = "nmesh-gateway-launcher.sh"
    executable = shlex.quote(sys.executable)
    text = f"""#!/bin/sh
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$script_dir/gateway.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|'#'*) continue ;;
      *=*) key=${{line%%=*}}; value=${{line#*=}}; export "$key=$value" ;;
    esac
  done < "$script_dir/gateway.env"
fi
export NMESH_HOME="$script_dir"
exec {executable} -m nmesh.gateway.server --port {port}
"""
    return filename, text


def unit_install_path(
    filename: str, os_name: str | None = None
) -> Path | None:
    """Where the generated unit must be saved for the install command."""
    windows = os.name == "nt" if os_name is None else os_name == "nt"
    if windows:
        return None
    macos = (
        sys.platform.startswith("darwin")
        if os_name is None
        else os_name.startswith("darwin")
    )
    if macos:
        return Path.home() / "Library" / "LaunchAgents" / filename
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config_home) / "systemd" / "user" / filename


def service_unit(
    port: int = 18000, os_name: str | None = None
) -> tuple[str, str, str]:
    """Return the filename, unit text, and explicit install command."""
    current = os_name
    windows = os.name == "nt" if current is None else current == "nt"
    macos = (
        sys.platform.startswith("darwin")
        if current is None
        else current.startswith("darwin")
    )
    launcher, _ = launcher_script(port, current)
    launcher_path = str(nmesh_home() / launcher)
    if windows:
        task_command = subprocess.list2cmdline((launcher_path,))
        command = (
            f'schtasks /create /tn nmesh-gateway /sc onlogon '
            f'/tr {task_command}'
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
  <array><string>{html.escape(launcher_path)}</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
"""
        install = unit_install_path(f"{label}.plist", current)
        return "com.nmesh.gateway.plist", text, f"launchctl load {install}"
    text = f"""[Unit]
Description=nmesh gateway
After=network.target

[Service]
ExecStart={shlex.join((launcher_path,))}
Restart=on-failure

[Install]
WantedBy=default.target
"""
    install = unit_install_path("nmesh-gateway.service", current)
    return "nmesh-gateway.service", text, f"systemctl --user enable --now {install}"


def watch_unit(
    interval_hours: int = 24, os_name: str | None = None
) -> tuple[str, str, str]:
    """Return a periodic watch unit and explicit install command."""
    if interval_hours < 1:
        raise ValueError("interval_hours must be at least 1")
    current = os_name
    windows = os.name == "nt" if current is None else current == "nt"
    macos = (
        sys.platform.startswith("darwin")
        if current is None
        else current.startswith("darwin")
    )
    executable = str(sys.executable)
    if windows:
        command_line = subprocess.list2cmdline(
            (executable, "-m", "nmesh.cli", "watch")
        )
        command = (
            f"schtasks /create /tn nmesh-watch /sc daily "
            f"/mo {max(interval_hours // 24, 1)} /tr {command_line}"
        )
        return "nmesh-watch.xml", command + "\n", command
    if macos:
        label = "com.nmesh.watch"
        text = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array><string>{html.escape(executable)}</string><string>-m</string>
  <string>nmesh.cli</string><string>watch</string></array>
  <key>StartInterval</key><integer>{interval_hours * 3600}</integer>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"""
        return f"{label}.plist", text, f"launchctl load ~/Library/LaunchAgents/{label}.plist"
    service_path = unit_install_path("nmesh-watch.service", current)
    timer_path = unit_install_path("nmesh-watch.timer", current)
    text = f"""# Save as {service_path}
[Unit]
Description=nmesh watch

[Service]
ExecStart={shlex.join((executable, "-m", "nmesh.cli", "watch"))}

[Install]
WantedBy=default.target

# Save as {timer_path}
[Unit]
Description=Run nmesh watch periodically

[Timer]
OnBootSec=5min
OnUnitActiveSec={interval_hours}h
Persistent=true

[Install]
WantedBy=timers.target
"""
    install = unit_install_path("nmesh-watch.timer", current)
    command = f"systemctl --user enable --now {install}"
    return "nmesh-watch.service + nmesh-watch.timer", text, command
