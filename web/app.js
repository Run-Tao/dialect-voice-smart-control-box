"use strict";

const TARGET_SR = 16000;
const MAX_SECONDS = 15;

const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const recStatus = document.getElementById("recStatus");

let audioCtx = null, stream = null, processor = null, source = null;
let chunks = [], recording = false, inputSr = 48000;

function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE"); writeStr(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, "data"); view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function downsample(buffer, fromSr, toSr) {
  if (fromSr === toSr) return buffer;
  const ratio = fromSr / toSr;
  const len = Math.floor(buffer.length / ratio);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const idx = i * ratio, lo = Math.floor(idx), hi = Math.min(lo + 1, buffer.length - 1);
    out[i] = buffer[lo] + (buffer[hi] - buffer[lo]) * (idx - lo);
  }
  return out;
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  audioCtx = new AudioContext();
  inputSr = audioCtx.sampleRate;
  source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  chunks = [];
  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    const total = chunks.reduce((n, c) => n + c.length, 0);
    if (total / inputSr >= MAX_SECONDS) stopRecording();
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
  recording = true;
  micBtn.classList.add("recording");
  micLabel.textContent = "录音中…";
  recStatus.textContent = "正在聆听（最长 15 秒），再次点击结束";
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  processor.disconnect(); source.disconnect();
  stream.getTracks().forEach(t => t.stop());
  await audioCtx.close();
  micBtn.classList.remove("recording");
  micLabel.textContent = "识别中…";
  recStatus.textContent = "正在本地识别…";
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.length; }
  const samples = downsample(merged, inputSr, TARGET_SR);
  const form = new FormData();
  form.append("audio", encodeWav(samples, TARGET_SR), "command.wav");
  try {
    const resp = await fetch("/api/recognize", { method: "POST", body: form });
    renderResult(await resp.json());
  } catch (err) {
    recStatus.textContent = "请求失败: " + err;
  }
  micLabel.textContent = "点击说话";
}

micBtn.addEventListener("click", () => {
  if (recording) stopRecording();
  else startRecording().catch(err => { recStatus.textContent = "无法访问麦克风: " + err; });
});

document.getElementById("textBtn").addEventListener("click", async () => {
  const text = document.getElementById("textInput").value.trim();
  if (!text) return;
  const resp = await fetch("/api/text", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  renderResult(await resp.json());
});
document.getElementById("textInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("textBtn").click();
});

function renderResult(r) {
  document.getElementById("rawText").textContent = r.raw_text || "（空）";
  document.getElementById("normText").textContent = r.normalized_text || "—";
  document.getElementById("cmdJson").textContent = r.command ? JSON.stringify(r.command, null, 2) : "—";
  const fb = document.getElementById("feedback");
  fb.textContent = r.feedback || r.error || "—";
  fb.className = "val feedback " + (r.ok ? "ok" : "err");
  const t = r.timings_ms || {};
  document.getElementById("timings").textContent =
    Object.entries(t).map(([k, v]) => `${k}: ${v}ms`).join(" · ") || "—";
  recStatus.textContent = r.ok ? "执行成功" : (r.error || "未执行");
  refreshState();
}

const DEV_LABELS = { light: "灯", ac: "空调", tv: "电视", curtain: "窗帘", fan: "风扇", socket: "插座" };
const ROOM_LABELS = { living_room: "客厅", bedroom: "卧室", kitchen: "厨房", bathroom: "浴室", study: "书房" };

function devStatus(dev, s) {
  if (dev === "curtain") return s.open ? "已开" : "已关";
  if (!s.on) return "关";
  if (dev === "light") return `开 · ${s.brightness}%`;
  if (dev === "ac") return `开 · ${s.temp}°C`;
  if (dev === "tv") return `开 · CH${s.channel} · 音${s.volume}`;
  if (dev === "fan") return `开 · ${s.speed}档`;
  return "开";
}

async function refreshState() {
  try {
    const resp = await fetch("/api/state");
    const { state } = await resp.json();
    const grid = document.getElementById("stateGrid");
    grid.innerHTML = "";
    for (const [room, devs] of Object.entries(state || {})) {
      const card = document.createElement("div");
      card.className = "room-card";
      let html = `<h3>${ROOM_LABELS[room] || room}</h3>`;
      for (const [dev, s] of Object.entries(devs)) {
        const active = dev === "curtain" ? s.open : s.on;
        html += `<div class="dev-row"><span>${DEV_LABELS[dev] || dev}</span>` +
                `<span class="${active ? "on" : "off"}">${devStatus(dev, s)}</span></div>`;
      }
      card.innerHTML = html;
      grid.appendChild(card);
    }
  } catch (e) { /* 服务未就绪时静默 */ }
}

document.getElementById("resetBtn").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  refreshState();
});

refreshState();
setInterval(refreshState, 5000);