"use strict";
const $ = id => document.getElementById(id);
const player = $("player");
let current = null, rows = [], revision = "", duration = 0, dirty = false;
let editing = null, history = [], loopEnd = null, loading = false, saving = false;

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
async function api(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed.");
  return value;
}
function status() {
  $("save-state").textContent = saving ? "Saving…" : dirty ? "Unsaved changes" : "Saved / unchanged";
  $("save").disabled = !current || !dirty || loading || saving;
  $("undo").disabled = !history.length || loading || saving;
  $("clip").disabled = loading || saving;
  for (const id of ["add", "replay", "mark-start", "mark-end", "cancel-edit"]) $(id).disabled = !current || loading || saving;
}
function changed() {
  dirty = true;
  status();
  render();
}
function remember() { history.push(structuredClone(rows)); }
function resetEditor() {
  editing = null;
  $("edit-mode").textContent = "New annotation";
  $("add").innerHTML = "Add annotation <kbd>A</kbd>";
  $("cancel-edit").hidden = true;
}
function contextDirection() {
  const context = ["talking", "music"].includes($("label").value);
  $("direction").disabled = context;
  if (context) $("direction").value = "unknown";
  $("direction-help").textContent = context
    ? "Background context only. Mark the full interval; direction does not apply."
    : "Label centered sounds too. The later caption filter will hide them by default.";
}
function render() {
  const body = $("rows");
  body.replaceChildren();
  $("count").textContent = rows.length;
  $("empty").hidden = rows.length > 0;
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.classList.toggle("editing", editing === index);
    const arrow = {left: "← Left", right: "Right →", center: "Center", unknown: "Unknown"};
    for (const value of [Number(row.start_s).toFixed(3), Number(row.end_s).toFixed(3), row.label,
      ["talking", "music"].includes(row.label) ? "—" : arrow[row.direction]]) {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    }
    const actions = document.createElement("td");
    for (const [label, action] of [["Play", () => replay(row)], ["Edit", () => edit(index)], ["Delete", () => remove(index)]]) {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = label;
      button.setAttribute("aria-label", `${label} ${row.label} at ${Number(row.start_s).toFixed(3)} seconds`);
      if (label === "Delete") button.className = "danger";
      button.disabled = loading || saving;
      button.addEventListener("click", action); actions.append(button);
    }
    tr.append(actions); body.append(tr);
  });
}
function edit(index) {
  editing = index;
  const row = rows[index];
  $("start").value = Number(row.start_s).toFixed(3);
  $("end").value = Number(row.end_s).toFixed(3);
  $("label").value = row.label; $("direction").value = row.direction;
  contextDirection();
  $("edit-mode").textContent = "Editing annotation";
  $("add").textContent = "Update annotation";
  $("cancel-edit").hidden = false;
  render(); message("Editing this row. Update it to keep your changes.");
}
function remove(index) {
  remember(); rows.splice(index, 1); resetEditor(); changed();
  message("Annotation deleted. Undo restores it until you leave this clip.");
}
function selection() {
  const start = Number($("start").value), end = Number($("end").value);
  if ($("start").value === "" || $("end").value === "" || !Number.isFinite(start) || !Number.isFinite(end)
    || start < 0 || start >= end || end > duration + .001) {
    throw new Error(`Choose a start before the end, within 0–${duration.toFixed(3)} seconds.`);
  }
  return {start_s: start, end_s: Math.min(end, duration)};
}
function add(event) {
  event?.preventDefault();
  if (!current || loading || saving) return;
  try {
    const row = {...selection(), label: $("label").value, direction: $("direction").value};
    if (row.end_s - row.start_s < .001) throw new Error("Mark an interval of at least 0.001 seconds.");
    remember();
    if (editing === null) rows.push(row); else rows[editing] = row;
    rows.sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s || a.label.localeCompare(b.label));
    resetEditor(); changed(); message("Annotation added to the list. Save CSV to write it to disk.");
  } catch (error) { message(error.message, true); }
}
async function replay(row) {
  if (!current || loading) return;
  try {
    row ||= selection();
    loopEnd = Number(row.end_s);
    player.currentTime = Number(row.start_s);
    await player.play();
  } catch (error) { loopEnd = null; message(error.message, true); }
}
async function save() {
  if (!dirty || !current || saving || loading) return;
  saving = true; status(); render();
  try {
    const value = await api("/api/annotations", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id: current, rows, revision})});
    rows = value.rows; revision = value.revision; dirty = false;
    resetEditor(); message(`Saved ${rows.length} annotations to ${value.csv_path}.`);
  } catch (error) { message(`${error.message} Your unsaved rows are still here.`, true); }
  finally { saving = false; status(); render(); }
}
async function loadClip(id) {
  if (loading || saving) return;
  if (dirty && !confirm("This clip has unsaved annotations. Discard those changes and switch clips?")) {
    $("clip").value = current; return;
  }
  loading = true; player.pause(); loopEnd = null; status(); render();
  message("Opening clip…");
  try {
    const [metadata, annotations] = await Promise.all([
      api(`/api/metadata?id=${encodeURIComponent(id)}`), api(`/api/annotations?id=${encodeURIComponent(id)}`)
    ]);
    current = id; duration = metadata.duration; rows = annotations.rows; revision = annotations.revision;
    dirty = false; history = []; resetEditor();
    $("start").value = "0.000"; $("end").value = Math.min(1, duration).toFixed(3);
    $("start").max = $("end").max = duration.toFixed(3);
    $("clock").textContent = "0.000";
    player.src = `/media?id=${encodeURIComponent(id)}`; player.load();
    $("clip-info").textContent = `${duration.toFixed(2)} seconds · ${metadata.channels} channels · ${(metadata.sample_rate / 1000).toFixed(1)} kHz${metadata.channels !== 2 ? " · Direction labeling needs stereo audio" : ""}`;
    $("save-path").textContent = annotations.csv_path;
    message("Clip ready. Nothing is saved until you press Save CSV.");
  } catch (error) {
    $("clip").value = current || ""; message(error.message, true);
  } finally { loading = false; status(); render(); }
}
function mark(id) {
  if (!current || loading || saving) return;
  $(id).value = Math.min(player.currentTime, duration).toFixed(3);
}
function seek(delta) { loopEnd = null; player.currentTime = Math.max(0, Math.min(duration, player.currentTime + delta)); }
$("annotation-form").addEventListener("submit", add);
$("clip").addEventListener("change", e => loadClip(e.target.value));
$("mark-start").addEventListener("click", () => mark("start"));
$("mark-end").addEventListener("click", () => mark("end"));
$("label").addEventListener("change", contextDirection);
$("cancel-edit").addEventListener("click", () => { resetEditor(); render(); });
$("replay").addEventListener("click", () => replay());
$("save").addEventListener("click", save);
$("undo").addEventListener("click", () => {
  if (!history.length || saving || loading) return;
  rows = history.pop(); resetEditor(); changed(); message("Last annotation change undone.");
});
$("back").addEventListener("click", () => seek(-2));
$("forward").addEventListener("click", () => seek(2));
$("play").addEventListener("click", () => {
  if (!current || loading) return;
  loopEnd = null;
  if (player.paused) player.play().catch(error => message(error.message, true)); else player.pause();
});
$("speed").addEventListener("change", () => { player.playbackRate = Number($("speed").value); });
player.addEventListener("loadedmetadata", () => { player.playbackRate = Number($("speed").value); });
player.addEventListener("timeupdate", () => {
  $("clock").textContent = player.currentTime.toFixed(3);
  if (loopEnd !== null && player.currentTime >= loopEnd) { player.pause(); loopEnd = null; }
});
player.addEventListener("error", () => message("The browser could not play this file. Try Chrome or Safari and check that the media exists.", true));
window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("keydown", event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") { event.preventDefault(); save(); return; }
  if (event.metaKey || event.ctrlKey || event.altKey || /INPUT|SELECT|TEXTAREA|BUTTON|VIDEO/.test(event.target.tagName)) return;
  const action = {" ": () => $("play").click(), i: () => mark("start"), o: () => mark("end"),
    r: () => replay(), a: () => add(), ArrowLeft: () => seek(-2), ArrowRight: () => seek(2)}[event.key];
  if (action) { event.preventDefault(); action(); }
});
async function init() {
  status();
  try {
    const value = await api("/api/clips");
    $("clip").replaceChildren();
    for (const [group, title] of [["valorant", "Valorant · development"], ["xonotic_second_game", "Xonotic · reserved for second-game adaptation"]]) {
      const options = document.createElement("optgroup"); options.label = title;
      for (const clip of value.clips.filter(clip => clip.group === group)) {
        const option = document.createElement("option"); option.value = clip.id;
        option.textContent = clip.name + (clip.available ? "" : " (file missing)"); option.disabled = !clip.available;
        options.append(option);
      }
      $("clip").append(options);
    }
    const first = value.clips.find(clip => clip.available);
    if (!first) throw new Error("No development media found. See README for the local file layout.");
    $("clip").value = first.id;
    await loadClip(first.id);
  } catch (error) { message(error.message, true); }
}
init();
