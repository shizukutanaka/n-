"""Local web UI for the gateway: one static page plus the JSON view it reads.

The page is served by the gateway itself (which binds 127.0.0.1 only) and is
fully self-contained — no external assets, no build step — so the single-file
install story does not change. It renders only what the gateway already knows:
the plan's services, their runtime state, and live telemetry. The two actions
it offers are the gateway's existing non-destructive admin operations
(reload the plan, unload a service); there is deliberately no way to edit
configuration or delete anything from here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# The shell carries no data, so — like /health — it is served without the API
# key; every byte of status data comes from /ui/api/overview, which the auth
# middleware guards exactly like /status and /metrics when NMESH_API_KEY set.
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nmesh — local status</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 72rem;
         padding: 0 1rem; color: #1a1a1a; }
  header { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
  h1 { font-size: 1.4rem; margin: 0; }
  .badge { padding: 0.15rem 0.6rem; border-radius: 1rem; font-size: 0.85rem;
           background: #eee; }
  .badge.up { background: #d7f0d7; }
  .badge.down { background: #f6d7d7; }
  .muted { color: #666; font-size: 0.85rem; }
  .error { color: #a00; }
  #warnings { color: #8a6d00; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { border-bottom: 1px solid #ddd; padding: 0.4rem 0.5rem;
           text-align: left; font-size: 0.92rem; }
  th { border-bottom: 2px solid #999; }
  button { cursor: pointer; }
  footer { margin-top: 2rem; }
</style>
</head>
<body>
<header>
  <h1>nmesh</h1>
  <span id="gw-state" class="badge">loading…</span>
  <button id="refresh" type="button">Refresh</button>
  <button id="reload" type="button">Reload plan</button>
  <span id="updated" class="muted"></span>
</header>

<section id="keybox" hidden>
  <p>This gateway requires an API key (<code>NMESH_API_KEY</code>). Enter it so
  this page can read status and use the actions; it is kept only in this tab's
  session storage and sent as a bearer token to this gateway.</p>
  <input id="key" type="password" autocomplete="off" size="40">
  <button id="key-apply" type="button">Use key</button>
</section>

<p id="error" class="error" hidden></p>
<ul id="warnings"></ul>
<div id="services"></div>

<footer class="muted">
  Local only: this page is served by the nmesh gateway on 127.0.0.1 and reads
  <code>/ui/api/overview</code>. Unloading a service stops its engine to free
  memory; it starts again automatically on the next request. Configuration
  cannot be changed from here.
</footer>

<script>
"use strict";
const KEY_SLOT = "nmesh-ui-key";
const state = { timer: null };

function el(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function apiKey() {
  return sessionStorage.getItem(KEY_SLOT) || "";
}

async function api(path, options) {
  const init = Object.assign({}, options || {});
  const headers = new Headers(init.headers || {});
  const key = apiKey();
  if (key) headers.set("Authorization", "Bearer " + key);
  init.headers = headers;
  return fetch(path, init);
}

function showError(message) {
  const box = document.getElementById("error");
  box.hidden = !message;
  box.textContent = message || "";
}

function fmt(value, digits) {
  if (value === null || value === undefined) return "–";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

function serviceState(service) {
  if (!service.running) return "stopped";
  if (service.sleeping) return "sleeping";
  if (service.idle) return "idle";
  return "running";
}

function render(data) {
  const badge = document.getElementById("gw-state");
  badge.textContent = data.gateway_running ? "gateway running" : "gateway idle";
  badge.className = "badge " + (data.gateway_running ? "up" : "down");
  document.getElementById("updated").textContent =
    "updated " + new Date().toLocaleTimeString();

  const warnings = document.getElementById("warnings");
  warnings.replaceChildren();
  for (const warning of data.warnings || []) {
    warnings.appendChild(el("li", warning));
  }

  const host = document.getElementById("services");
  host.replaceChildren();
  if (!data.services.length) {
    host.appendChild(el("p", "No services in the current plan."));
    return;
  }
  const table = el("table");
  const head = el("tr");
  for (const label of ["Service", "Model", "Backend", "Port", "State", "PID",
                       "Restarts", "In flight", "Samples", "Decode tok/s",
                       "TTFT s", ""]) {
    head.appendChild(el("th", label));
  }
  table.appendChild(head);
  for (const service of data.services) {
    const row = el("tr");
    const telemetry = service.telemetry || {};
    const cells = [
      service.name,
      service.model || "–",
      service.backend || "–",
      service.port === null || service.port === undefined
        ? "–" : String(service.port),
      serviceState(service),
      service.pid === null || service.pid === undefined
        ? "–" : String(service.pid),
      String(service.restarts || 0),
      String(service.in_flight || 0),
      fmt(telemetry.samples, 0),
      fmt(telemetry.decode_tps_median, 1),
      fmt(telemetry.ttft_s_median, 2),
    ];
    for (const text of cells) row.appendChild(el("td", text));
    const action = el("td");
    if (service.running && service.planned) {
      const button = el("button", "Unload");
      button.type = "button";
      button.addEventListener("click", () => unload(service.name));
      action.appendChild(button);
    }
    row.appendChild(action);
    table.appendChild(row);
  }
  host.appendChild(table);
}

async function load() {
  let response;
  try {
    response = await api("/ui/api/overview");
  } catch (error) {
    showError("Cannot reach the gateway: " + error);
    return;
  }
  if (response.status === 401) {
    document.getElementById("keybox").hidden = false;
    showError("The gateway rejected the request (401). Enter the API key above.");
    return;
  }
  if (!response.ok) {
    showError("Gateway answered " + response.status + ".");
    return;
  }
  document.getElementById("keybox").hidden = true;
  showError("");
  render(await response.json());
}

async function act(path, label) {
  const response = await api(path, { method: "POST" });
  if (!response.ok) {
    showError(label + " failed: gateway answered " + response.status + ".");
    return;
  }
  await load();
}

async function unload(name) {
  if (!window.confirm("Unload " + name + "? Its engine stops and restarts " +
                      "automatically on the next request.")) return;
  await act("/admin/unload/" + encodeURIComponent(name), "Unload");
}

document.getElementById("refresh").addEventListener("click", load);
document.getElementById("reload").addEventListener("click",
  () => act("/admin/reload", "Reload"));
document.getElementById("key-apply").addEventListener("click", () => {
  const value = document.getElementById("key").value.trim();
  if (value) sessionStorage.setItem(KEY_SLOT, value);
  load();
});

load();
state.timer = setInterval(load, 5000);
</script>
</body>
</html>
"""


def build_overview(
    *,
    planned: Sequence[Mapping[str, object]],
    runtime: Mapping[str, object],
    running: Sequence[Mapping[str, object]],
    telemetry: Mapping[str, Mapping[str, float]],
    keep_alive: float,
) -> dict[str, object]:
    """Merge the plan, /status and /admin/running views into the one payload
    the UI renders, so the page needs a single request per refresh and the
    merge rules stay unit-testable without HTTP."""
    planned_by_name = {str(item["name"]): item for item in planned}
    runtime_entries = runtime.get("services")
    runtime_by_name: dict[str, Mapping[str, object]] = {}
    if isinstance(runtime_entries, list):
        for entry in runtime_entries:
            if isinstance(entry, Mapping) and isinstance(entry.get("service"), str):
                runtime_by_name[str(entry["service"])] = entry
    running_by_name = {
        str(item["service"]): item
        for item in running
        if isinstance(item.get("service"), str)
    }
    names = [str(item["name"]) for item in planned]
    names.extend(name for name in runtime_by_name if name not in planned_by_name)

    services: list[dict[str, object]] = []
    for name in names:
        plan_entry = planned_by_name.get(name, {})
        runtime_entry = runtime_by_name.get(name, {})
        running_entry = running_by_name.get(name, {})
        services.append({
            "name": name,
            "model": plan_entry.get("model") or runtime_entry.get("model_ref"),
            "backend": plan_entry.get("backend") or runtime_entry.get("backend"),
            "port": plan_entry.get("port") or runtime_entry.get("port"),
            "roles": plan_entry.get("roles") or [],
            "planned": name in planned_by_name,
            "running": bool(
                running_entry.get("running", runtime_entry.get("running", False))
            ),
            "idle": bool(running_entry.get("idle", runtime_entry.get("idle", False))),
            "idle_seconds": running_entry.get("idle_seconds"),
            "in_flight": running_entry.get("in_flight", 0),
            "pid": runtime_entry.get("pid"),
            "restarts": runtime_entry.get("restarts", 0),
            "shared": bool(runtime_entry.get("shared", False)),
            "sleeping": bool(runtime_entry.get("sleeping", False)),
            "telemetry": telemetry.get(name),
        })
    warnings = runtime.get("warnings")
    return {
        "gateway_running": bool(runtime.get("running", False)),
        "warnings": list(warnings) if isinstance(warnings, list) else [],
        "keep_alive": keep_alive,
        "services": services,
    }
