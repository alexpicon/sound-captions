"use strict";
const $ = id => document.getElementById(id);
const player = $("player");
let current = null, rows = [], revision = "", duration = 0, dirty = false;
let editing = null, history = [], loopEnd = null, loading = false, saving = false;
let mode = "development", suggestions = [], decisions = {}, complete = false, reviewRevision = "", pendingEdit = null;
const pending = () => suggestions.filter(row => !decisions[row.id]);
const query = id => `id=${encodeURIComponent(id)}&mode=${mode}`;

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
  $("mode").disabled = loading || saving;
  $("finish").disabled = !current || loading || saving || pending().length > 0;
  for (const id of ["add", "replay", "mark-start", "mark-end", "cancel-edit"]) $(id).disabled = !current || loading || saving;
}
function changed() {
  dirty = true; complete = false;
  status();
  render();
}
function remember() { history.push(structuredClone({rows, decisions, complete})); }
function resetEditor() {
  editing = null;
  pendingEdit = null;
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
  $("count").textContent = rows.length + pending().length;
  $("empty").hidden = rows.length + pending().length > 0;
  $("review-summary").textContent = mode === "test"
    ? `${rows.length} human labels. ${complete ? "Entire test clip marked labeled." : "Listen through the full clip and label game sounds plus talking/music before finishing."}`
    : `${rows.length} confirmed · ${pending().length} unconfirmed. ${complete ? "Entire clip marked reviewed." : "Check all guesses and label missed sounds before finishing the entire clip."}`;
  const entries = [...rows.map((row, index) => ({row, index})), ...pending().map(row => ({row, proposal: true}))];
  entries.sort((a, b) => a.row.start_s - b.row.start_s || a.row.end_s - b.row.end_s);
  entries.forEach(({row, index, proposal}) => {
    const tr = document.createElement("tr");
    tr.classList.toggle("editing", proposal ? pendingEdit === row.id : editing === index);
    tr.classList.toggle("unconfirmed", Boolean(proposal));
    const arrow = {left: "← Left", right: "Right →", center: "Center", unknown: "Unknown"};
    const values = [Number(row.start_s).toFixed(3), Number(row.end_s).toFixed(3), row.label,
      ["talking", "music"].includes(row.label) ? "—" : arrow[row.direction],
      proposal ? `Unconfirmed · model score ${Number(row.score).toFixed(2)}` : "Confirmed"];
    for (const [column, value] of values.entries()) {
      const td = document.createElement("td"); td.textContent = value;
      td.dataset.label = ["Start", "End", "Sound / context", "Direction", "Review status"][column];
      tr.append(td);
    }
    const actions = document.createElement("td");
    const buttons = proposal
      ? [["Play", () => replay(row)], ["Edit", () => editProposal(row)], ["Confirm", () => resolveProposal(row, true)], ["Delete", () => resolveProposal(row, false)]]
      : [["Play", () => replay(row)], ["Edit", () => edit(index)], ["Delete", () => remove(index)]];
    for (const [label, action] of buttons) {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = label;
      button.setAttribute("aria-label", `${label} ${proposal ? "guess " : ""}${row.label} at ${Number(row.start_s).toFixed(3)} seconds`);
      if (label === "Delete") button.className = "danger";
      button.disabled = loading || saving;
      button.addEventListener("click", action); actions.append(button);
    }
    tr.append(actions); body.append(tr);
  });
}
function editProposal(row) {
  resetEditor(); pendingEdit = row.id;
  $("start").value = Number(row.start_s).toFixed(3); $("end").value = Number(row.end_s).toFixed(3);
  $("label").value = row.label; $("direction").value = row.direction; contextDirection();
  $("edit-mode").textContent = "Correcting an unconfirmed guess";
  $("add").textContent = "Confirm corrected annotation"; $("cancel-edit").hidden = false;
  render();
}
function resolveProposal(row, accept) {
  remember(); decisions[row.id] = accept ? "accepted" : "rejected";
  if (accept) rows.push(Object.fromEntries(["start_s", "end_s", "label", "direction"].map(field => [field, row[field]])));
  resetEditor(); changed(); message(accept ? "Guess confirmed. Save to keep it." : "Guess deleted. Save to remember your decision.");
}
function edit(index) {
  pendingEdit = null;
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
    if (pendingEdit !== null) { decisions[pendingEdit] = "accepted"; rows.push(row); }
    else if (editing === null) rows.push(row); else rows[editing] = row;
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
async function save(finish = false) {
  if ((!dirty && !finish) || !current || saving || loading) return;
  if (finish && pending().length) { message("Confirm or delete every guess first.", true); return; }
  saving = true; status(); render();
  try {
    const value = await api("/api/annotations", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id: current, rows, revision, mode, review_revision: reviewRevision,
        review: {decisions, complete: finish || complete}})});
    rows = value.rows; revision = value.revision; dirty = false;
    decisions = value.review.decisions; complete = value.review.complete; reviewRevision = value.review_revision;
    resetEditor(); message(finish ? "Clip fully reviewed. Saved confirmed labels." : `Saved ${rows.length} confirmed annotations to ${value.csv_path}.`);
    refreshAugmentation();
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
      api(`/api/metadata?${query(id)}`), api(`/api/annotations?${query(id)}`)
    ]);
    current = id; duration = metadata.duration; rows = annotations.rows; revision = annotations.revision;
    suggestions = annotations.suggestions; decisions = annotations.review.decisions;
    complete = annotations.review.complete; reviewRevision = annotations.review_revision;
    dirty = false; history = []; resetEditor();
    $("start").value = "0.000"; $("end").value = Math.min(1, duration).toFixed(3);
    $("start").max = $("end").max = duration.toFixed(3);
    $("clock").textContent = "0.000";
    player.src = `/media?${query(id)}`; player.load();
    $("clip-info").textContent = `${duration.toFixed(2)} seconds · ${metadata.channels} channels · ${(metadata.sample_rate / 1000).toFixed(1)} kHz${metadata.channels !== 2 ? " · Direction labeling needs stereo audio" : ""}`;
    $("save-path").textContent = annotations.csv_path;
    message("Clip ready. Nothing is saved until you press Save CSV.");
    refreshAugmentation();
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
$("save").addEventListener("click", () => save());
$("finish").addEventListener("click", () => save(true));
$("undo").addEventListener("click", () => {
  if (!history.length || saving || loading) return;
  const previous = history.pop(); rows = previous.rows; decisions = previous.decisions;
  resetEditor(); changed(); message("Last annotation change undone.");
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
async function refreshAugmentation() {
  if (mode === "test") { $("augmentation-state").textContent = "Answer key only. This clip is excluded from suggestions, voice removal, and training."; return; }
  if (!current || !complete) { $("augmentation-state").textContent = "Finishing the full clip queues a voice-reduced training copy. Save CSV keeps partial progress."; return; }
  const requestedId = current;
  try {
    const value = await api(`/api/augmentation?id=${encodeURIComponent(current)}`);
    if (requestedId !== current || mode !== "development") return;
    $("augmentation-state").textContent = ({not_prepared: "Voice-reduced copy is not prepared yet. Finish reviewing to queue it.",
      preparing: "Preparing an aligned stereo voice-reduced copy in the background…", ready: "Original + voice-reduced stereo copy ready for training, grouped in the same clip fold.",
      error: `Voice removal: ${value.error}`})[value.state];
  } catch (error) { $("augmentation-state").textContent = error.message; }
}
setInterval(() => { if (complete && mode === "development") refreshAugmentation(); }, 3000);
$("mode").addEventListener("change", async event => {
  if (dirty && !confirm("Discard unsaved annotations and switch labeling mode?")) { $("mode").value = mode; return; }
  dirty = false; player.pause(); current = null; rows = []; suggestions = []; decisions = {}; history = []; complete = false;
  mode = event.target.value; resetEditor(); render();
  await init();
});
async function init() {
  status();
  try {
    const value = await api(`/api/clips?mode=${mode}`);
    $("mode-badge").textContent = mode === "test" ? "Answer key · no model output" : "Development labeling";
    $("mode-help").textContent = mode === "test"
      ? "Label from the original audio only. No model guesses or scores appear here. These clips stay excluded from training. The final lab clip remains closed."
      : "Guesses are unconfirmed until you check them. Model scores are not accuracy percentages. YAMNet has no reload class; add missed sounds manually.";
    $("clip").replaceChildren();
    const groups = mode === "test" ? [["valorant", "Valorant · test answer key"], ["unseen_cs", "CS · unseen-game answer key"]]
      : [["valorant", "Valorant · development"], ["xonotic_second_game", "Xonotic · reserved for second-game adaptation"]];
    for (const [group, title] of groups) {
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
