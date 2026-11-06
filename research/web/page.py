# -*- coding: utf-8 -*-
"""v3 Web 演示台的离线内联页面（CSS/JS 全部内联，不依赖任何 CDN）。"""
import html
import json

_CSS = r"""
:root{
  --bg:#eef2f7; --panel:#ffffff; --ink:#1f2937; --muted:#6b7280;
  --brand:#2563eb; --brand-2:#1d4ed8; --ok:#059669; --bad:#dc2626;
  --amber:#d97706; --line:#e5e7eb; --soft:#f8fafc; --shadow:0 10px 30px rgba(15,23,42,.08);
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",system-ui,-apple-system,sans-serif;
  background:linear-gradient(180deg,#eef2f7 0%,#f6f8fb 100%);
  color:var(--ink); min-height:100vh;
}
.app{max-width:1180px;margin:0 auto;padding:24px 20px 48px}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,#2563eb,#0ea5e9);
  color:#fff;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700;
  box-shadow:var(--shadow)}
h1{margin:0;font-size:22px;letter-spacing:.5px}
.brand p{margin:3px 0 0;color:var(--muted);font-size:13px}
.model-info{display:flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);
  border-radius:999px;padding:8px 14px;font-size:13px;box-shadow:var(--shadow);max-width:420px}
.model-info .chip{background:#e0ecff;color:var(--brand-2);border-radius:999px;padding:2px 10px;font-weight:600}
#model-name{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
  background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px 18px;
  box-shadow:var(--shadow);margin-bottom:18px}
.threshold-box{flex:1;min-width:300px}
.threshold-box label{font-size:14px;font-weight:600}
.threshold-box b{color:var(--brand);font-variant-numeric:tabular-nums}
#threshold{width:100%;margin-top:8px;accent-color:var(--brand);cursor:pointer}
.hint{color:var(--muted);font-size:12px;margin-top:4px}
.actions{display:flex;gap:10px}
.btn{border:none;border-radius:10px;padding:10px 16px;font-size:14px;cursor:pointer;
  font-family:inherit;transition:all .15s}
.btn.ghost{background:var(--soft);border:1px solid var(--line);color:var(--ink)}
.btn.ghost:hover{background:#eef2ff;border-color:#c7d2fe}
.btn:disabled{opacity:.5;cursor:not-allowed}
.banner{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;border-radius:12px;
  padding:10px 14px;font-size:13px;margin-bottom:14px}
.banner.info{background:#eff6ff;border-color:#bfdbfe;color:#1e40af}
.hidden{display:none}
.layout{display:grid;grid-template-columns:1.35fr 1fr;gap:18px;margin-bottom:18px}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px 20px;
  box-shadow:var(--shadow)}
.panel h2{margin:0 0 4px;font-size:16px}
.sub{color:var(--muted);font-size:12px;margin:4px 0 14px}
.cmd-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
@media(max-width:560px){.cmd-grid{grid-template-columns:1fr}}
.cmd-btn{display:flex;flex-direction:column;align-items:flex-start;gap:4px;border:1px solid var(--line);
  border-radius:14px;background:linear-gradient(180deg,#ffffff,#f8fafc);padding:12px 14px;cursor:pointer;
  font-family:inherit;text-align:left;transition:all .15s;position:relative;overflow:hidden}
.cmd-btn:hover{border-color:#c7d2fe;transform:translateY(-1px);box-shadow:0 6px 16px rgba(37,99,235,.12)}
.cmd-btn:active{transform:none}
.cmd-btn.busy{opacity:.55;pointer-events:none}
.cmd-btn .cmd-name{font-size:17px;font-weight:700;color:var(--ink)}
.cmd-btn .cmd-code{font-size:11px;color:var(--brand);letter-spacing:.4px;font-weight:600}
.cmd-btn .cmd-text{font-size:12px;color:var(--muted);line-height:1.4}
.cmd-btn.unknown{border-color:#fecaca;background:linear-gradient(180deg,#fff7f7,#fef2f2)}
.cmd-btn.unknown .cmd-code{color:var(--bad)}
.warnings{color:var(--amber);font-size:12px;margin:12px 0 0;line-height:1.6}
.devices-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.device{border:1px solid var(--line);border-radius:14px;padding:14px;text-align:center;
  background:var(--soft);transition:all .25s}
.device .icon{font-size:30px;line-height:1;margin-bottom:8px}
.device .name{font-size:14px;font-weight:700}
.device .status{font-size:13px;color:var(--muted);margin-top:4px}
.device.on{background:#ecfdf5;border-color:#a7f3d0;box-shadow:inset 0 0 0 1px #a7f3d0}
.device.on .status{color:var(--ok);font-weight:700}
.device.off .status{color:#94a3b8}
.state-line{font-size:13px;color:var(--muted);margin:6px 0 12px}
.state-line b{color:var(--ink)}
.result{border:1px dashed var(--line);border-radius:14px;padding:14px;background:var(--soft);min-height:150px}
.result-empty{color:var(--muted);font-size:13px;line-height:1.7;padding:8px 4px}
.result-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.result-command{font-size:20px;font-weight:800}
.result-raw{color:var(--muted);font-size:12px;margin-top:2px}
.badge{border-radius:999px;padding:4px 12px;font-size:12px;font-weight:700}
.badge.ok{background:#d1fae5;color:#065f46}
.badge.bad{background:#fee2e2;color:#991b1b}
.conf-row{display:flex;align-items:center;gap:10px;margin:14px 0 12px;font-size:12px;color:var(--muted)}
.bar{flex:1;height:10px;border-radius:999px;background:#e5e7eb;overflow:hidden}
.bar .fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#22c55e,#059669);transition:width .4s}
.bar .fill.bad{background:linear-gradient(90deg,#f87171,#dc2626)}
.conf-row b{font-variant-numeric:tabular-nums;color:var(--ink);min-width:52px;text-align:right}
.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;font-size:12px}
.result-grid div{display:flex;justify-content:space-between;gap:8px;border-top:1px dashed var(--line);padding-top:6px}
.result-grid span{color:var(--muted)}
.result-grid b{color:var(--ink);font-weight:600;text-align:right}
.log-panel{padding-bottom:10px}
.log-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
.table-wrap{overflow-x:auto;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:760px}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--soft);color:var(--muted);font-weight:600;position:sticky;top:0}
td.mono,.mono{font-family:Consolas,"Courier New",monospace;font-size:11px}
.tag-ok{color:var(--ok);font-weight:700}
.tag-bad{color:var(--bad);font-weight:700}
.empty-row td{color:var(--muted);text-align:center;padding:18px}
footer{color:#94a3b8;font-size:12px;text-align:center;margin-top:22px;line-height:1.8}
code{background:#eef2f7;border-radius:4px;padding:1px 5px;font-size:11px}
"""

_JS = r"""
(function(){
  const CATALOG = JSON.parse(document.getElementById('catalog-data').textContent || '{}');
  const state = { threshold: parseFloat(document.getElementById('threshold').value) };
  const $ = (id) => document.getElementById(id);
  const banner = $('banner'), thresholdInput = $('threshold'), thresholdValue = $('threshold-value');
  let busy = false;

  const CMD_COLORS = {};

  function showBanner(text, info){
    banner.textContent = text;
    banner.className = 'banner' + (info ? ' info' : '');
  }
  function hideBanner(){ banner.className = 'banner hidden'; }

  function fmtPct(x){ return (x * 100).toFixed(1) + '%'; }
  function fmtTime(ts){
    if(!ts) return '—';
    const m = String(ts).match(/T(\d{2}:\d{2}:\d{2})/);
    return m ? m[1] : ts;
  }
  function baseName(p){
    if(!p) return '—';
    const parts = String(p).split(/[\\/]/);
    return parts[parts.length - 1];
  }
  function esc(s){
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  async function api(url, options){
    const resp = await fetch(url, options);
    let data = null;
    try { data = await resp.json(); } catch(e) { data = null; }
    if(!resp.ok){
      const msg = (data && (data.error || data.message)) || ('HTTP ' + resp.status);
      throw new Error(msg);
    }
    return data;
  }

  function setBusy(on){
    busy = on;
    document.querySelectorAll('.cmd-btn').forEach((b) => b.classList.toggle('busy', on));
    $('reset-btn').disabled = on;
  }

  function renderState(data){
    const st = (data && data.state) || {};
    const lightOn = st.light === 'on';
    const curtainOpen = st.curtain === 'open';
    const lightEl = $('device-light'), curtainEl = $('device-curtain');
    lightEl.className = 'device ' + (lightOn ? 'on' : 'off');
    curtainEl.className = 'device ' + (curtainOpen ? 'on' : 'off');
    $('light-status').textContent = lightOn ? '開' : '關';
    $('curtain-status').textContent = curtainOpen ? '開' : '關';
    $('state-text').textContent = data.state_text || ('燈：' + (lightOn?'開':'關') + '；窗簾：' + (curtainOpen?'開':'關'));
  }

  function renderResult(r){
    const accepted = !!r.accepted;
    const conf = Number(r.confidence || 0);
    const cls = accepted ? 'ok' : 'bad';
    const label = r.command_label || r.predicted_intent || '未知指令';
    const rawCmd = r.predicted_intent || (r.predicted || '');
    const audioMeta = baseName(r.audio_path);
    const reasonText = r.rejection_reason === 'unknown_command'
      ? '未知指令（不在六類命令內）'
      : (r.rejection_reason === 'low_confidence' ? '置信度低於閾值' : '—');
    const html = '' +
      '<div class="result-head">' +
        '<div><div class="result-command">' + esc(label) + '</div>' +
        '<div class="result-raw">' + esc(rawCmd) + ' · ' + esc(audioMeta) + '</div></div>' +
        '<span class="badge ' + cls + '">' + (accepted ? '已接受' : '已拒識') + '</span>' +
      '</div>' +
      '<div class="conf-row"><span>置信度</span>' +
        '<div class="bar"><div class="fill ' + (accepted ? '' : 'bad') + '" style="width:' + (conf*100).toFixed(1) + '%"></div></div>' +
        '<b>' + fmtPct(conf) + '</b></div>' +
      '<div class="result-grid">' +
        '<div><span>本次閾值</span><b>' + Number(r.threshold).toFixed(2) + '</b></div>' +
        '<div><span>設備反饋</span><b>' + esc(r.response || r.device_response) + '</b></div>' +
        '<div><span>延遲</span><b>' + (Number(r.latency_sec || r.latency || 0) * 1000).toFixed(0) + ' ms</b></div>' +
        '<div><span>拒識原因</span><b>' + esc(reasonText) + '</b></div>' +
      '</div>';
    $('result-card').innerHTML = html;
  }

  function renderLog(payload){
    const rows = (payload && payload.records) || [];
    const tbody = $('log-body');
    $('log-meta').textContent = '共 ' + rows.length + ' 條（最近 ' + (payload.limit || 10) + ' 條，自動刷新）';
    if(!rows.length){
      tbody.innerHTML = '<tr class="empty-row"><td colspan="8">暫無操作記錄，點擊左側命令按鈕開始演示。</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map((r) => {
      const ok = !!r.accepted;
      return '<tr>' +
        '<td class="mono">' + esc(fmtTime(r.timestamp)) + '</td>' +
        '<td class="mono">' + esc(r.seq) + '</td>' +
        '<td class="mono" title="' + esc(r.audio_path) + '">' + esc(baseName(r.audio_path)) + '</td>' +
        '<td>' + esc(r.command_label || r.predicted_intent || '—') + '</td>' +
        '<td class="mono">' + fmtPct(Number(r.confidence || 0)) + '</td>' +
        '<td class="' + (ok ? 'tag-ok' : 'tag-bad') + '">' + (ok ? '接受' : '拒識') + '</td>' +
        '<td>' + esc(r.response || r.device_response) + '</td>' +
        '<td class="mono">' + (Number(r.latency_sec || 0) * 1000).toFixed(0) + ' ms</td>' +
      '</tr>';
    }).join('');
  }

  async function refreshState(){
    try { renderState(await api('/api/state')); } catch(e) { /* 静默 */ }
  }
  async function refreshLog(){
    try { renderLog(await api('/api/log?limit=10')); } catch(e) { /* 静默 */ }
  }

  async function predict(audioPath){
    if(busy) return;
    hideBanner();
    setBusy(true);
    $('result-card').innerHTML = '<div class="result-empty">識別中…（DTW 模板模型在 CPU 上可能需要數秒）</div>';
    try {
      const data = await api('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_path: audioPath, threshold: state.threshold })
      });
      renderResult(data);
      renderState({ state: data.state, state_text: data.state_text });
      await refreshLog();
    } catch(e) {
      showBanner('處理失敗：' + e.message);
    } finally {
      setBusy(false);
    }
  }

  document.querySelectorAll('.cmd-btn').forEach((btn) => {
    btn.addEventListener('click', () => predict(btn.dataset.audio));
  });

  thresholdInput.addEventListener('input', () => {
    state.threshold = parseFloat(thresholdInput.value);
    thresholdValue.textContent = state.threshold.toFixed(2);
  });

  $('reset-btn').addEventListener('click', async () => {
    if(busy) return;
    hideBanner();
    setBusy(true);
    try {
      const data = await api('/api/reset', { method: 'POST' });
      renderState(data);
      $('result-card').innerHTML = '<div class="result-empty">設備狀態已重置為出廠狀態。日誌保留，可繼續演示。</div>';
      await refreshLog();
      showBanner('設備狀態已重置。', true);
      setTimeout(hideBanner, 2500);
    } catch(e) {
      showBanner('重置失敗：' + e.message);
    } finally {
      setBusy(false);
    }
  });

  refreshState();
  refreshLog();
  setInterval(() => {
    if(document.hidden) return;
    refreshState();
    refreshLog();
  }, 4000);
})();
"""

_PAGE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <div class="logo">聲</div>
      <div>
        <h1>方言智控盒 · 第三階段演示台</h1>
        <p>語音指令 → 識別／拒識 → 模擬設備 → 操作日誌（全離線）</p>
      </div>
    </div>
    <div class="model-info" title="__MODEL_NAME__"><span class="chip">模型</span><span id="model-name">__MODEL_NAME__</span></div>
  </header>

  <section class="toolbar">
    <div class="threshold-box">
      <label for="threshold">拒識閾值 <b id="threshold-value">__DEFAULT_THRESHOLD__</b></label>
      <input type="range" id="threshold" min="0" max="1" step="0.01" value="__DEFAULT_THRESHOLD__">
      <div class="hint">置信度低於閾值即拒識，設備不動作。拖動滑塊可對比同一音頻「接受／拒識」的差異。</div>
    </div>
    <div class="actions">
      <button id="reset-btn" class="btn ghost" type="button">重置設備狀態</button>
    </div>
  </section>

  <div id="banner" class="banner hidden"></div>

  <main class="layout">
    <section class="panel">
      <h2>六類命令 · 未見表達測試音頻</h2>
      <p class="sub">音頻來自 <code>track_a_expression_split.csv</code> 的 test 表達組（未參與訓練）；未知指令按鈕用於驗證拒識。</p>
      <div class="cmd-grid">
__COMMAND_BUTTONS__
__UNKNOWN_BUTTON__
      </div>
__WARNINGS__
    </section>

    <section class="panel">
      <h2>模擬設備狀態</h2>
      <p class="sub">燈與窗簾的狀態機；ALL_OFF 同時關閉兩者，STATUS_ALL 只查詢不改變。</p>
      <div class="devices-grid">
        <div class="device off" id="device-light">
          <div class="icon">💡</div>
          <div class="name">燈光</div>
          <div class="status" id="light-status">關</div>
        </div>
        <div class="device off" id="device-curtain">
          <div class="icon">🪟</div>
          <div class="name">窗簾</div>
          <div class="status" id="curtain-status">關</div>
        </div>
      </div>
      <div class="state-line">目前狀態：<b id="state-text">燈：關；窗簾：關</b></div>
      <div class="result" id="result-card">
        <div class="result-empty">點擊左側命令按鈕，觀察「識別命令、置信度、是否拒識、設備反饋、延遲」的完整鏈路。</div>
      </div>
    </section>
  </main>

  <section class="panel log-panel">
    <div class="log-head">
      <h2>操作日誌</h2>
      <span class="sub" id="log-meta">最近 10 條 · 自動刷新</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>時間</th><th>#</th><th>音頻</th><th>識別命令</th><th>置信度</th>
          <th>結果</th><th>設備反饋</th><th>延遲</th>
        </tr></thead>
        <tbody id="log-body">
          <tr class="empty-row"><td colspan="8">暫無操作記錄，點擊左側命令按鈕開始演示。</td></tr>
        </tbody>
      </table>
    </div>
    <p class="sub" style="margin-top:10px">每條操作都會追加寫入 <code>__LOG_PATH__</code>（JSONL，UTF-8，可審計）。</p>
  </section>

  <footer>
    離線內聯頁面 · 無 CDN 依賴 · 模擬設備（未連接真實 ESP32）· 輸入為 TTS 測試音頻，非實時麥克風<br>
    啟動方式：<code>python scripts/run_web_demo.py</code> · 第三階段原型（v3）
  </footer>
</div>
<script type="application/json" id="catalog-data">__CATALOG_JSON__</script>
<script>__JS__</script>
</body>
</html>
"""


def _command_button(item, index):
    audio = html.escape(item.get('audio_path', ''), quote=True)
    label = html.escape(item.get('label') or item.get('command') or '未知指令')
    code = html.escape(item.get('command') or 'UNKNOWN')
    text = html.escape(item.get('text') or '')
    text_html = f'<span class="cmd-text">「{text}」</span>' if text else \
        '<span class="cmd-text">未知指令測試音頻</span>'
    return (
        f'<button class="cmd-btn" type="button" data-index="{index}" '
        f'data-audio="{audio}" data-label="{label}">'
        f'<span class="cmd-name">{label}</span>'
        f'<span class="cmd-code">{code}</span>'
        f'{text_html}</button>'
    )


def render_page(catalog, default_threshold=0.9, model_name='v2 DTW', log_path=''):
    """渲染演示台单页 HTML（全部内联，离线可用）。"""
    commands = catalog.get('commands') or []
    buttons = '\n'.join(_command_button(item, i) for i, item in enumerate(commands))
    unknown = catalog.get('unknown')
    if unknown:
        audio = html.escape(unknown.get('audio_path', ''), quote=True)
        text = html.escape(unknown.get('text') or '')
        text_html = f'<span class="cmd-text">「{text}」</span>' if text else \
            '<span class="cmd-text">未知指令測試音頻</span>'
        unknown_button = (
            '<button class="cmd-btn unknown" type="button" data-index="unknown" '
            f'data-audio="{audio}" data-label="未知指令">'
            '<span class="cmd-name">未知指令（應拒識）</span>'
            '<span class="cmd-code">UNKNOWN</span>'
            f'{text_html}</button>'
        )
    else:
        unknown_button = (
            '<button class="cmd-btn unknown" type="button" disabled>'
            '<span class="cmd-name">未知指令（無可用音頻）</span>'
            '<span class="cmd-code">UNKNOWN</span>'
            '<span class="cmd-text">未找到未知指令測試音頻</span></button>'
        )
    warnings = ''
    if catalog.get('warnings'):
        warnings = '<p class="warnings">⚠ ' + '<br>⚠ '.join(
            html.escape(w) for w in catalog['warnings']) + '</p>'

    catalog_json = json.dumps(catalog, ensure_ascii=False).replace('</', '<\\/')
    page = _PAGE
    for placeholder, value in (
        ('__TITLE__', '方言智控盒 · 第三階段語音控制演示台'),
        ('__CSS__', _CSS),
        ('__JS__', _JS),
        ('__MODEL_NAME__', html.escape(str(model_name))),
        ('__DEFAULT_THRESHOLD__', f'{float(default_threshold):.2f}'),
        ('__COMMAND_BUTTONS__', buttons),
        ('__UNKNOWN_BUTTON__', unknown_button),
        ('__WARNINGS__', warnings),
        ('__LOG_PATH__', html.escape(str(log_path))),
        ('__CATALOG_JSON__', catalog_json),
    ):
        page = page.replace(placeholder, value)
    return page
