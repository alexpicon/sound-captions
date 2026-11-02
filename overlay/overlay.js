"use strict";
(() => {
  const root = document.getElementById("captions");
  const speechRoot = document.getElementById("speech-captions");
  const layer = new URLSearchParams(location.search).get("layer");
  const showSounds = layer !== "speech", showSpeech = layer !== "sound";
  const labels = new Set(["footsteps", "gunshot", "explosion"]);
  const sides = new Set(["left", "right", "center", "unknown"]);
  const rows = new Map();
  let socket, session = null, backoff = 250, reconnectTimer, stopped = false, speechRow = null;
  function clearSpeech() { speechRoot.replaceChildren(); speechRow = null; }
  function clear() { rows.clear(); root.replaceChildren(); clearSpeech(); }
  function tick() {
    const now = Date.now() / 1000;
    for (const [key, row] of rows) {
      if (now >= row.expires) { row.node.remove(); rows.delete(key); }
      else if (!matchMedia("(prefers-reduced-motion: reduce)").matches) row.node.classList.toggle("fading", now >= row.expires - .25);
    }
    if (speechRow) {
      if (now >= speechRow.expires) clearSpeech();
      else if (!matchMedia("(prefers-reduced-motion: reduce)").matches)
        speechRow.node.classList.toggle("fading", now >= speechRow.expires - .15);
    }
  }
  function caption(event) {
    if (!showSounds) return;
    if (!event || !(labels.has(event.label) || (event.mode === 'csv_replay' && event.label === 'reload')) || !sides.has(event.direction)
        || !Number.isSafeInteger(event.id) || event.id < 1
        || typeof event.session !== "string" || event.session.length > 64
        || !Number.isFinite(event.expires_at) || !Number.isFinite(event.emitted_at)
        || event.expires_at > event.emitted_at + 3.05 || event.expires_at <= Date.now()/1000) return;
    if (session !== event.session) { clear(); session = event.session; }
    const key = `${session}:${event.id}`;
    if (rows.has(key)) return;
    tick();
    while (rows.size >= 2) { const first = rows.keys().next().value; rows.get(first).node.remove(); rows.delete(first); }
    const node = document.createElement("div"); node.className = "caption"; node.dataset.captionId = String(event.id);
    const prefix = document.createElement("span"); prefix.className = "ink"; prefix.textContent = "[";
    const word = document.createElement("span"); word.className = "sound";
    const text = document.createElement("span"); text.className = "ink"; text.textContent = event.label; word.append(text);
    const suffix = document.createElement("span"); suffix.className = "ink";
    suffix.textContent = ({left:" ← left]",right:" → right]"})[event.direction] || "]";
    node.append(prefix, word, suffix); root.append(node);
    rows.set(key, {node, expires:Math.min(event.expires_at, Date.now()/1000 + 3)});
    tick();
  }
  function speech(event) {
    if (!showSpeech) return;
    if (!event || event.mode !== "csv_replay" || typeof event.session !== "string"
        || event.session.length > 64 || typeof event.text !== "string" || event.text.length > 1000) return;
    if (session !== event.session) { clear(); session = event.session; }
    if (!event.text) { clearSpeech(); return; }
    if (!Number.isFinite(event.expires_at) || !Number.isFinite(event.emitted_at)
        || event.expires_at <= Date.now()/1000) return;
    clearSpeech();
    const node = document.createElement("div"); node.className = "caption speech-caption";
    const box = document.createElement("span"); box.className = "sound";
    const text = document.createElement("span"); text.className = "ink"; text.textContent = event.text;
    box.append(text); node.append(box); speechRoot.append(node);
    speechRow = {node, expires:event.expires_at}; tick();
  }
  function receive(event) {
    let data; try { data = JSON.parse(event.data); } catch { return; }
    if (data.type === "snapshot") {
      clear(); session = data.session;
      if (Array.isArray(data.captions)) for (const value of data.captions.slice(-2)) caption(value);
      speech(data.speech);
    } else if (data.type === "caption") caption(data);
    else if (data.type === "speech") speech(data);
    else if (data.type === "reset") clear();
    else if (data.type === "status" && ["paused","stopped","reconnecting","waiting-for-playback","loading"].includes(data.phase)) clear();
  }
  async function connect() {
    if (stopped) return;
    try {
      const response = await fetch("/config.json", {cache:"no-store"});
      if (!response.ok) throw new Error("Overlay configuration unavailable");
      const config = await response.json();
      const url = new URL(config.websocket);
      if (url.protocol !== "ws:" || !["127.0.0.1", "localhost"].includes(url.hostname)) throw new Error("Local WebSocket required");
      socket = new WebSocket(url.href);
      socket.onmessage = receive;
      socket.onopen = () => { backoff = 250; };
      socket.onerror = () => socket.close();
      socket.onclose = () => { clear(); if (!stopped) { reconnectTimer=setTimeout(connect,backoff); backoff=Math.min(backoff*2,5000); } };
    } catch { clear(); if (!stopped) { reconnectTimer=setTimeout(connect,backoff); backoff=Math.min(backoff*2,5000); } }
  }
  setInterval(tick, 50);
  addEventListener("pagehide", () => { stopped=true; clearTimeout(reconnectTimer); socket?.close(); clear(); });
  connect();
})();
