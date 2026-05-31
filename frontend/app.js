/* ─── STATE ARCHITECTURE ─── */
const state = {
  ws: null,
  wsUrl: 'ws://' + window.location.host + '/ws/stream',
  streamInterval: null,
  streamFps: 5,
  isStreaming: false,
  webcamStream: null,
  brailleReference: null,
  selectedUploadFile: null
};

// Automatic SSL/WS detection
if (window.location.protocol === 'https:') {
  state.wsUrl = 'wss://' + window.location.host + '/ws/stream';
} else if (window.location.hostname === '') {
  state.wsUrl = 'ws://localhost:8000/ws/stream';
}

/* ─── HAPTIC FEEDBACK (accessibility) ─── */
function hapticTap(pattern) {
  if (navigator.vibrate) {
    navigator.vibrate(pattern);
  }
}

function initHapticButtons() {
  document.addEventListener(
    'click',
    (e) => {
      const el = e.target.closest('button, .nav-btn, .premium-toggle, .drop-zone, .interactive-card');
      if (el) {
        hapticTap(12);
      }
    },
    { passive: true }
  );
}

/* ─── ON SYSTEM BOOT ─── */
document.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  renderHeroInteractiveCards();
  loadReferenceData();
  loadHistoryData();
  initUploadZone();
  initHapticButtons();
});

/* ─── TAB PANEL SELECTORS ─── */
function switchTab(tabId) {
  document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
  
  if (tabId === 'camera') {
    document.getElementById('navCamera').classList.add('active');
    document.getElementById('tabCamera').classList.add('active');
  } else if (tabId === 'upload') {
    document.getElementById('navUpload').classList.add('active');
    document.getElementById('tabUpload').classList.add('active');
    loadHistoryData(); // Refresh history panel automatically when tab is opened
  } else if (tabId === 'reference') {
    document.getElementById('navReference').classList.add('active');
    document.getElementById('tabReference').classList.add('active');
  }
}

/* ─── TOAST HUD NOTIFICATIONS ─── */
function showToast(message, duration = 3000) {
  const toast = document.getElementById('toast');
  toast.innerText = message;
  toast.style.display = 'block';
  
  setTimeout(() => {
    toast.style.display = 'none';
  }, duration);
}

/* ─── GENERATE INTERACTIVE HERO CARDS ─── */
function renderHeroInteractiveCards() {
  const container = document.getElementById('heroLetters');
  container.innerHTML = '';
  
  // Spell out "SPARSH" (Hindi Touch name) in beautiful interactive Braille hover blocks
  const sparshChars = [
    { char: 'S', pattern: '011100' },
    { char: 'P', pattern: '111100' },
    { char: 'A', pattern: '100000' },
    { char: 'R', pattern: '111010' },
    { char: 'S', pattern: '011100' },
    { char: 'H', pattern: '110010' }
  ];
  
  sparshChars.forEach(item => {
    const card = document.createElement('div');
    card.className = 'interactive-card';
    card.setAttribute('data-char', item.char);
    card.title = `Letter: ${item.char} (Pattern: ${item.pattern})`;
    
    // Create 6-bit grid
    for (let i = 0; i < 6; i++) {
      const dot = document.createElement('div');
      dot.className = item.pattern[i] === '1' ? 'dot filled' : 'dot';
      card.appendChild(dot);
    }
    
    // Quick web voice read on hover
    card.addEventListener('mouseenter', () => {
      speakWord(item.char);
    });
    
    container.appendChild(card);
  });
}

/* ─── WEBSOCKET CONTROLLER ─── */
function connectWebSocket() {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  
  dot.className = 'status-indicator connecting';
  label.innerText = 'Connecting...';
  
  state.ws = new WebSocket(state.wsUrl);
  
  state.ws.onopen = () => {
    dot.className = 'status-indicator connected';
    label.innerText = 'Online Core';
    showToast('HUD streaming socket connected.');
  };
  
  state.ws.onclose = () => {
    dot.className = 'status-indicator';
    label.innerText = 'Offline Core';
    if (state.isStreaming) {
      stopCamera();
    }
    setTimeout(connectWebSocket, 5000); // Autoconnect
  };
  
  state.ws.onerror = (err) => {
    console.error('Socket error:', err);
    dot.className = 'status-indicator';
    label.innerText = 'Offline Error';
  };
  
  state.ws.onmessage = (event) => {
    try {
      const response = JSON.parse(event.data);
      if (response.type === 'result') {
        renderStreamOutput(response);
      } else if (response.type === 'error') {
        showToast('Socket Error: ' + response.message);
      }
    } catch (e) {
      console.error('Parsing frame outcome failed:', e);
    }
  };
}

/* ─── LIVE WEBCAM SCANNER ─── */
function cameraErrorMessage(err) {
  const name = err && err.name ? err.name : '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera blocked. Click the lock/camera icon in the address bar → Allow camera → reload.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No camera found on this device.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Camera is in use by another app (Zoom, Teams, etc.). Close it and try again.';
  }
  if (name === 'OverconstrainedError') {
    return 'Camera constraints not supported. Retrying with default settings…';
  }
  if (name === 'SecurityError') {
    return 'Camera needs a secure page. Open http://localhost:8000/app (not a file:// link).';
  }
  return 'Could not start camera: ' + (err && err.message ? err.message : 'unknown error');
}

async function tryGetUserMedia(constraints) {
  return navigator.mediaDevices.getUserMedia({ ...constraints, audio: false });
}

async function startCamera() {
  const video = document.getElementById('videoFeed');
  const placeholder = document.getElementById('videoPlaceholder');
  const btnStart = document.getElementById('btnStartCamera');
  const btnStop = document.getElementById('btnStopCamera');

  if (!window.isSecureContext) {
    const host = window.location.hostname;
    const msg =
      host === 'localhost' || host === '127.0.0.1'
        ? 'Camera requires a secure page. Use http://localhost:8000/app'
        : 'On mobile, camera needs HTTPS. Open via localhost tunnel (ngrok) or deploy with SSL—not raw IP http.';
    showToast(msg, 6000);
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Camera API not supported in this browser.');
    return;
  }

  const attempts = [
    { video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } },
    { video: { width: { ideal: 1280 }, height: { ideal: 720 } } },
    { video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } },
    { video: true },
  ];

  let lastErr = null;
  for (const constraints of attempts) {
    try {
      state.webcamStream = await tryGetUserMedia(constraints);
      video.srcObject = state.webcamStream;
      await video.play();

      placeholder.classList.add('hidden');
      video.classList.remove('hidden');
      btnStart.classList.add('hidden');
      btnStop.classList.remove('hidden');

      document.getElementById('fpsBadge').style.display = 'block';

      state.isStreaming = true;
      startFrameLoop();
      hapticTap([15, 25, 15]);
      showToast('Tactile HUD Scanner active.');
      return;
    } catch (err) {
      lastErr = err;
      console.warn('Camera attempt failed:', constraints, err);
    }
  }

  console.error('Webcam failed:', lastErr);
  showToast(cameraErrorMessage(lastErr));
}

function stopCamera() {
  const video = document.getElementById('videoFeed');
  const placeholder = document.getElementById('videoPlaceholder');
  const btnStart = document.getElementById('btnStartCamera');
  const btnStop = document.getElementById('btnStopCamera');
  
  state.isStreaming = false;
  
  if (state.streamInterval) {
    clearInterval(state.streamInterval);
    state.streamInterval = null;
  }
  
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
  
  video.classList.add('hidden');
  placeholder.classList.remove('hidden');
  btnStop.classList.add('hidden');
  btnStart.classList.remove('hidden');
  
  document.getElementById('fpsBadge').style.display = 'none';
  document.getElementById('annotatedCard').style.display = 'none';
}

function onFpsChange(val) {
  document.getElementById('fpsSliderVal').innerText = val + ' Hz';
  state.streamFps = parseInt(val);
  
  if (state.isStreaming) {
    startFrameLoop();
  }
}

function startFrameLoop() {
  if (state.streamInterval) {
    clearInterval(state.streamInterval);
  }
  
  const video = document.getElementById('videoFeed');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  let frames = 0;
  let lastFpsTime = performance.now();
  
  state.streamInterval = setInterval(() => {
    if (!state.isStreaming || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w === 0 || h === 0) return;
    
    // Maintain highly optimized 640px horizontal package width
    const targetW = 640;
    const targetH = (h / w) * targetW;
    
    canvas.width = targetW;
    canvas.height = targetH;
    ctx.drawImage(video, 0, 0, targetW, targetH);
    
    const base64Img = canvas.toDataURL('image/jpeg', 0.7);
    
    const msg = {
      type: 'frame',
      image: base64Img,
      include_annotated: document.getElementById('toggleAnnotated').checked,
      include_audio: document.getElementById('toggleAudio').checked,
      correct_perspective: document.getElementById('togglePerspective').checked
    };
    
    state.ws.send(JSON.stringify(msg));
    
    // FPS calculator
    frames++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      document.getElementById('fpsValue').innerText = frames;
      frames = 0;
      lastFpsTime = now;
    }
  }, 1000 / state.streamFps);
}

/* ─── SCANNER RESULT RENDERER ─── */
function renderCellPredictions(containerId, cells) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!cells || cells.length === 0) {
    el.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  el.classList.remove('hidden');
  el.innerHTML = '<p class="cell-predictions-title">Per-cell ML predictions</p>';
  const row = document.createElement('div');
  row.className = 'cell-predictions-row';
  cells.forEach((c) => {
    const chip = document.createElement('span');
    chip.className = 'cell-chip';
    if (c.confidence_pct >= 80) chip.classList.add('conf-high');
    else if (c.confidence_pct >= 55) chip.classList.add('conf-mid');
    else chip.classList.add('conf-low');
    const label = c.char === ' ' ? 'space' : c.char;
    chip.innerText = `${label} ${c.confidence_pct}%`;
    chip.title = `Pattern: ${c.pattern}`;
    row.appendChild(chip);
  });
  el.appendChild(row);
}

function renderDecodedText(textBoard, text, cellCount) {
  if (text && text.trim()) {
    textBoard.innerHTML = `<div class="hud-text">${escapeHtml(text)}</div>`;
    return;
  }
  if (cellCount > 0) {
    textBoard.innerHTML =
      '<span class="output-placeholder">Cells were detected but could not be decoded. Check the per-cell predictions below and try turning off orientation warp.</span>';
    return;
  }
  textBoard.innerHTML = '<span class="output-placeholder">No Braille cells detected. Use a clear, well-lit photo of embossed dots.</span>';
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderStreamOutput(data) {
  if (!data.success) return;
  
  // Diagnostics
  document.getElementById('statCells').innerText = data.cell_count;
  document.getElementById('statDots').innerText = data.dot_count;
  document.getElementById('statConf').innerText = Math.round(data.confidence * 100) + '%';
  document.getElementById('statMs').innerText = data.processing_time_ms;
  
  const transcription = document.getElementById('outputText');
  renderDecodedText(transcription, data.text, data.cell_count);
  renderCellPredictions('streamCellPredictions', data.cells);
  
  if (data.text && data.text.trim()) {
    renderLineGrid(data.lines, data.patterns_by_line);
  }
  
  // Display overlaid frame
  const annCard = document.getElementById('annotatedCard');
  const annImg = document.getElementById('annotatedImg');
  if (data.annotated_image_b64) {
    annCard.style.display = 'block';
    annImg.src = 'data:image/jpeg;base64,' + data.annotated_image_b64;
  } else {
    annCard.style.display = 'none';
  }
  
  // Play synthesized audio if streaming voice checked
  if (data.audio_b64) {
    playBase64Audio(data.audio_b64);
  }
}

function renderLineGrid(lines, patternsByLine) {
  const container = document.getElementById('linesContainer');
  container.innerHTML = '';
  
  if (!lines || lines.length === 0) {
    container.innerHTML = '<p class="output-placeholder">Matrix details will populate after dot recognition completes.</p>';
    return;
  }
  
  lines.forEach((lineText, idx) => {
    if (!lineText.trim()) return;
    
    const row = document.createElement('div');
    row.className = 'line-row';
    
    const num = document.createElement('span');
    num.className = 'line-num';
    num.innerText = `Line ${idx + 1}`;
    
    const text = document.createElement('span');
    text.className = 'line-text';
    text.innerText = lineText;
    
    const patterns = document.createElement('span');
    patterns.className = 'line-patterns';
    
    const patternStr = (patternsByLine && patternsByLine[idx]) 
      ? patternsByLine[idx].map(p => p.substring(0, 6)).join(' ')
      : '';
    patterns.innerText = patternStr;
    patterns.title = patternStr;
    
    row.appendChild(num);
    row.appendChild(text);
    row.appendChild(patterns);
    container.appendChild(row);
  });
}

/* ─── DRAG & DROP FILE ZONE ─── */
function initUploadZone() {
  const zone = document.getElementById('dropzone');
  
  ['dragenter', 'dragover'].forEach(name => {
    zone.addEventListener(name, (e) => {
      e.preventDefault();
      zone.classList.add('active');
    });
  });
  
  ['dragleave', 'drop'].forEach(name => {
    zone.addEventListener(name, (e) => {
      e.preventDefault();
      zone.classList.remove('active');
    });
  });
}

function onDragOver(e) {
  e.preventDefault();
}

function onDragLeave(e) {
  e.preventDefault();
}

function onDrop(e) {
  e.preventDefault();
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    processUploadedFile(files[0]);
  }
}

function onFileSelected(input) {
  if (input.files.length > 0) {
    processUploadedFile(input.files[0]);
  }
}

function clearUpload() {
  hapticTap(18);
  state.selectedUploadFile = null;

  document.getElementById('fileInput').value = '';
  document.getElementById('dropzone').classList.remove('hidden');
  document.getElementById('uploadPreview').classList.add('hidden');
  document.getElementById('btnAnalyze').setAttribute('disabled', 'true');

  document.getElementById('uploadOutputText').innerHTML =
    '<span class="output-placeholder">Upload an image from your dataset to display translated results.</span>';
  document.getElementById('uploadCellPredictions').classList.add('hidden');
  document.getElementById('uploadCellPredictions').innerHTML = '';
  document.getElementById('uploadStats').style.display = 'none';
  document.getElementById('uploadAnnotatedCard').style.display = 'none';

  showToast('Upload cleared. You can add a new image.');
}

function processUploadedFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Invalid file structure. Only JPEG or PNG is compatible.');
    return;
  }

  hapticTap(10);
  state.selectedUploadFile = file;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('dropzone').classList.add('hidden');
    
    const preview = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('uploadPreviewImg');
    const meta = document.getElementById('uploadMeta');
    
    preview.classList.remove('hidden');
    previewImg.src = e.target.result;
    meta.innerText = `${file.name} — ${(file.size / 1024).toFixed(1)} KB`;
    
    document.getElementById('btnAnalyze').removeAttribute('disabled');
  };
  reader.readAsDataURL(file);
}

/* ─── EXECUTE UPLOAD MATRIX ANALYSIS ─── */
async function analyzeUpload() {
  if (!state.selectedUploadFile) return;
  
  const loader = document.getElementById('uploadLoading');
  loader.classList.remove('hidden');
  
  const formData = new FormData();
  formData.append('file', state.selectedUploadFile);
  
  const includeAnnotated = document.getElementById('uploadAnnotated').checked;
  const includeAudio = document.getElementById('uploadAudio').checked;
  const correctPerspective = document.getElementById('uploadPerspective').checked;
  
  let endpoint = '/api/upload';
  if (window.location.hostname === '') {
    endpoint = 'http://localhost:8000/api/upload';
  }
  
  const url = `${endpoint}?include_annotated=${includeAnnotated}&include_audio=${includeAudio}&correct_perspective=${correctPerspective}`;
  
  try {
    const res = await fetch(url, {
      method: 'POST',
      body: formData
    });
    
    const data = await res.json();
    loader.classList.add('hidden');
    
    if (res.ok && data.success) {
      hapticTap([10, 30, 10]);
      renderUploadOutput(data);
      loadHistoryData();
    } else {
      hapticTap([40, 60, 40]);
      showToast('CV Engine failed: ' + (data.error || 'Server error'));
    }
  } catch (err) {
    loader.classList.add('hidden');
    console.error('Fetch error:', err);
    showToast('Connection failed. Verify backend service status.');
  }
}

function renderUploadOutput(data) {
  document.getElementById('uploadStats').style.display = 'grid';
  document.getElementById('uStatCells').innerText = data.cell_count;
  document.getElementById('uStatConf').innerText = Math.round(data.confidence * 100) + '%';
  document.getElementById('uStatMs').innerText = data.processing_time_ms;
  
  const textBoard = document.getElementById('uploadOutputText');
  renderDecodedText(textBoard, data.text, data.cell_count);
  renderCellPredictions('uploadCellPredictions', data.cells);
  
  const overlayCard = document.getElementById('uploadAnnotatedCard');
  const overlayImg = document.getElementById('uploadAnnotatedImg');
  if (data.annotated_image_b64) {
    overlayCard.style.display = 'block';
    overlayImg.src = 'data:image/jpeg;base64,' + data.annotated_image_b64;
  } else {
    overlayCard.style.display = 'none';
  }
  
  if (data.audio_b64) {
    playBase64Audio(data.audio_b64);
  }
}

/* ─── DATABASE HISTORY LOGS RETRIEVER ─── */
async function loadHistoryData() {
  let endpoint = '/api/history';
  if (window.location.hostname === '') {
    endpoint = 'http://localhost:8000/api/history';
  }
  
  try {
    const res = await fetch(endpoint);
    if (!res.ok) return;
    
    const records = await res.json();
    const container = document.getElementById('historyList');
    container.innerHTML = '';
    
    if (records.length === 0) {
      container.innerHTML = '<p class="output-placeholder" style="text-align:center; padding:2rem;">No historical records stored. Upload some files to populate the database.</p>';
      return;
    }
    
    records.forEach(item => {
      const row = document.createElement('div');
      row.className = 'history-row';
      
      // Thumbnail
      const thumb = document.createElement('img');
      thumb.className = 'hist-thumb';
      // Load the original image from the backend `/uploads/` path
      let thumbUrl = item.annotated_image ? item.annotated_image : item.original_image;
      if (window.location.hostname === '') {
        thumbUrl = 'http://localhost:8000' + thumbUrl;
      }
      thumb.src = thumbUrl;
      thumb.alt = 'Tactile scan audit';
      thumb.onclick = () => {
        window.open(thumbUrl, '_blank');
      };
      
      // Text and date
      const txtGroup = document.createElement('div');
      txtGroup.className = 'hist-text-group';
      
      const txt = document.createElement('span');
      txt.className = 'hist-text';
      txt.innerText = item.text;
      
      const dt = document.createElement('span');
      dt.className = 'hist-date';
      dt.innerText = item.timestamp;
      
      txtGroup.appendChild(txt);
      txtGroup.appendChild(dt);
      
      // Meta parameters
      const metaGroup = document.createElement('div');
      metaGroup.className = 'hist-meta';
      
      const cells = document.createElement('span');
      cells.className = 'hist-meta-item';
      cells.innerHTML = `Cells: <span class="hist-meta-val">${item.cell_count}</span>`;
      
      const dots = document.createElement('span');
      dots.className = 'hist-meta-item';
      dots.innerHTML = `Dots: <span class="hist-meta-val">${item.dot_count}</span>`;
      
      metaGroup.appendChild(cells);
      metaGroup.appendChild(dots);
      
      // Diagnostics badges
      const diagGroup = document.createElement('div');
      diagGroup.className = 'hist-diagnostics';
      
      const confBadge = document.createElement('span');
      confBadge.className = 'diag-badge badge-precision';
      confBadge.innerText = `Fit: ${Math.round(item.confidence * 100)}%`;
      
      const delayBadge = document.createElement('span');
      delayBadge.className = 'diag-badge badge-speed';
      delayBadge.innerText = `${item.processing_time}ms`;
      
      diagGroup.appendChild(confBadge);
      diagGroup.appendChild(delayBadge);
      
      // Action speak button
      const actBtn = document.createElement('button');
      actBtn.className = 'hist-action-btn';
      actBtn.innerHTML = '🔊';
      actBtn.title = 'Speak text';
      actBtn.onclick = () => {
        speakWord(item.text);
      };
      
      row.appendChild(thumb);
      row.appendChild(txtGroup);
      row.appendChild(metaGroup);
      row.appendChild(diagGroup);
      row.appendChild(actBtn);
      
      container.appendChild(row);
    });
  } catch (err) {
    console.error('Failed to load DB history:', err);
  }
}

async function clearDbHistory() {
  if (!confirm('Are you sure you want to delete all saved scans and clear the database logs?')) return;
  
  let endpoint = '/api/history';
  if (window.location.hostname === '') {
    endpoint = 'http://localhost:8000/api/history';
  }
  
  try {
    const res = await fetch(endpoint, { method: 'DELETE' });
    const data = await res.json();
    if (res.ok && data.success) {
      showToast('Database successfully wiped.');
      loadHistoryData();
      
      // Reset dropzone visually
      document.getElementById('uploadPreview').classList.add('hidden');
      document.getElementById('dropzone').classList.remove('hidden');
      document.getElementById('btnAnalyze').setAttribute('disabled', 'true');
      state.selectedUploadFile = null;
    } else {
      showToast('Could not clear database logs.');
    }
  } catch (err) {
    console.error('Clear history error:', err);
    showToast('Connection failed.');
  }
}

/* ─── DYNAMIC TACTILE CODEX GENERATOR ─── */
async function loadReferenceData() {
  let endpoint = '/api/braille/reference';
  if (window.location.hostname === '') {
    endpoint = 'http://localhost:8000/api/braille/reference';
  }
  
  try {
    const res = await fetch(endpoint);
    if (!res.ok) return;
    
    const data = await res.json();
    state.brailleReference = data.patterns;
    
    const grid = document.getElementById('refGrid');
    grid.innerHTML = '';
    
    // Sort reference matrix alphabetically for professional layout
    const sorted = Object.entries(data.patterns)
      .filter(([_, char]) => char.length === 1) // omit special indicators
      .sort((a, b) => a[1].localeCompare(b[1]));
      
    sorted.forEach(([pattern, char]) => {
      const card = document.createElement('div');
      card.className = 'codex-card';
      
      const gridDiv = document.createElement('div');
      gridDiv.className = 'codex-cell-grid';
      
      for (let i = 0; i < 6; i++) {
        const dot = document.createElement('span');
        dot.className = pattern[i] === '1' ? 'filled' : 'empty';
        gridDiv.appendChild(dot);
      }
      
      const label = document.createElement('div');
      label.className = 'codex-lbl';
      
      const charSpan = document.createElement('span');
      charSpan.className = 'codex-char';
      charSpan.innerText = char.toUpperCase();
      
      const patSpan = document.createElement('span');
      patSpan.className = 'codex-pattern';
      patSpan.innerText = pattern;
      
      label.appendChild(charSpan);
      label.appendChild(patSpan);
      
      card.appendChild(gridDiv);
      card.appendChild(label);
      grid.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load Codex:', err);
  }
}

/* ─── ACCESSIBILITY HIGH-FIDELITY SPEECH ─── */
function playBase64Audio(b64String) {
  const player = document.getElementById('audioPlayer');
  player.src = 'data:audio/mp3;base64,' + b64String;
  player.play().catch(e => console.log('Audio stream interrupted:', e));
}

function speakOutput() {
  const text = document.getElementById('outputText').innerText;
  if (!text || text.includes('No active feed')) return;
  speakWord(text);
}

function speakUploadOutput() {
  const text = document.getElementById('uploadOutputText').innerText;
  if (!text || text.includes('Upload an image')) return;
  speakWord(text);
}

// Speaks words with full pronunciation instead of spelling letters individual-by-individual
function speakWord(text) {
  if (!text || !text.trim()) return;
  
  // Clean text from single letter spacing or tags
  let cleaned = text.trim();
  
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(cleaned);
    
    // Choose optimal English speaking voice
    const voices = window.speechSynthesis.getVoices();
    const naturalVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Natural'));
    if (naturalVoice) {
      utterance.voice = naturalVoice;
    }
    
    utterance.rate = 0.92; // Slightly slower, highly legible rate
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  } else {
    showToast('Browser Speech Synthesis is not supported in this client.');
  }
}
