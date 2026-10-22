// AI-Serve Dashboard Console JavaScript Controller

const BASE_URL = window.location.origin;

// Global States
let activeTab = 'overview';
let ttsMode = 'design';
let currentTheme = 'light';
let lastApiData = { models: null, status: null, gpu: null, voices: null, comfyTask: null };
let comfyPollInterval = null;
let comfyBatchPollInterval = null;
// task_id -> data URL, so already-rendered thumbnails aren't re-fetched from
// /api/comfyui/task/{id} on every ~1.5s status poll -- only newly-completed items
// trigger a fetch.
const comfyBatchImageCache = {};
// task_ids with a fetch currently in flight -- separate from the cache above since
// "no entry yet" and "already being fetched" need to be distinguishable to avoid
// firing a second fetch on the next poll tick before the first one resolves.
const comfyBatchImageFetching = new Set();
let smoketestPollInterval = null;
let gatewayChatHistory = [];

// Multimodal input base64 states
let currentMultimodalBase64 = null;
let currentMultimodalFormat = null; // "image" or "audio"
let currentMultimodalFilename = "";

// Voice cloning reference audio base64 state
let ttsRefAudioBase64 = null;
let ttsRefAudioFilename = "";
let localVoices = [];

// UI Elements binding
const UI = {
    // Navigation & Theme
    tabLinks: document.querySelectorAll('.tabs .tab-link'),
    tabContents: document.querySelectorAll('.tab-content'),
    themeToggle: document.getElementById('theme-toggle'),
    logsViewport: document.getElementById('logs-viewport'),
    statusBackend: document.getElementById('status-backend'),
    statusComfyui: document.getElementById('status-comfyui'),

    // Gateway Chat
    gatewayModelSelect: document.getElementById('gateway-model-select'),
    gatewayThinking: document.getElementById('gateway-thinking'),
    gatewayTemp: document.getElementById('gateway-temp'),
    gatewayMaxTokens: document.getElementById('gateway-max-tokens'),
    gatewayRetries: document.getElementById('gateway-retries'),
    gatewayKeepalive: document.getElementById('gateway-keepalive'),
    gatewaySchema: document.getElementById('gateway-schema'),
    gatewaySystemPrompt: document.getElementById('gateway-system-prompt'),
    multimodalUploadZone: document.getElementById('multimodal-upload-zone'),
    multimodalInput: document.getElementById('multimodal-input'),
    multimodalPreviewContainer: document.getElementById('multimodal-preview-container'),
    multimodalImgPreview: document.getElementById('multimodal-img-preview'),
    multimodalAudioPreview: document.getElementById('multimodal-audio-preview'),
    multimodalFilename: document.getElementById('multimodal-filename'),
    removeMultimodal: document.getElementById('remove-multimodal'),
    multimodalPlaceholder: document.getElementById('multimodal-placeholder'),
    chatMessages: document.getElementById('chat-messages'),
    chatUserInput: document.getElementById('chat-user-input'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    clearChatBtn: document.getElementById('clear-chat-btn'),
    openSchemaHelperBtn: document.getElementById('open-schema-helper-btn'),
    tokenInputCount: document.getElementById('token-input-count'),
    tokenHistoryCount: document.getElementById('token-history-count'),
    tokenMaxLimit: document.getElementById('token-max-limit'),
    tokenProgressFill: document.getElementById('token-progress-fill'),
    tokenWarningActions: document.getElementById('token-warning-actions'),
    tokenWarningMsg: document.getElementById('token-warning-msg'),
    strategyPruneBtn: document.getElementById('strategy-prune-btn'),
    strategyClearBtn: document.getElementById('strategy-clear-btn'),
    strategyAutoshrinkBtn: document.getElementById('strategy-autoshrink-btn'),
    strategyAutosummarizeBtn: document.getElementById('strategy-autosummarize-btn'),
    schemaHelperModal: document.getElementById('schema-helper-modal'),
    closeSchemaHelperBtn: document.getElementById('close-schema-helper-btn'),
    providerTabLinks: document.querySelectorAll('.provider-tab-link'),
    providerDescContent: document.getElementById('provider-desc-content'),
    builderFieldsList: document.getElementById('builder-fields-list'),
    builderAddFieldBtn: document.getElementById('builder-add-field-btn'),
    schemaPreviewCode: document.getElementById('schema-preview-code'),
    payloadPreviewCode: document.getElementById('payload-preview-code'),
    copySchemaOnlyBtn: document.getElementById('copy-schema-only-btn'),
    copyPayloadBtn: document.getElementById('copy-payload-btn'),
    applySchemaBtn: document.getElementById('apply-schema-btn'),
    presetButtons: document.querySelectorAll('.btn-preset'),

    inspectHeaderBtn: document.getElementById('inspect-header-btn'),
    ggufHeaderModal: document.getElementById('gguf-header-modal'),
    ggufHeaderTitle: document.getElementById('gguf-header-title'),
    ggufHeaderContent: document.getElementById('gguf-header-content'),
    closeGgufHeaderBtn: document.getElementById('close-gguf-header-btn'),
    copyGgufHeaderBtn: document.getElementById('copy-gguf-header-btn'),

    // Training (remote fine-tuning monitor)
    trainingRefreshBtn: document.getElementById('training-refresh-btn'),
    trainingJobsList: document.getElementById('training-jobs-list'),
    trainingLogModal: document.getElementById('training-log-modal'),
    trainingLogTitle: document.getElementById('training-log-title'),
    trainingLogContent: document.getElementById('training-log-content'),
    closeTrainingLogBtn: document.getElementById('close-training-log-btn'),
    trainingChartModal: document.getElementById('training-chart-modal'),
    trainingChartTitle: document.getElementById('training-chart-title'),
    closeTrainingChartBtn: document.getElementById('close-training-chart-btn'),
    trainingChartCombined: document.getElementById('training-chart-combined'),
    trainingChartCaption: document.getElementById('training-chart-caption'),
    chartToggleGradNorm: document.getElementById('chart-toggle-grad-norm'),
    chartToggleRewards: document.getElementById('chart-toggle-rewards'),
    chartToggleRewardsWrap: document.getElementById('chart-toggle-rewards-wrap'),
    chartSmoothing: document.getElementById('chart-smoothing'),
    chartSmoothingValue: document.getElementById('chart-smoothing-value'),

    // Servers & GPUs
    modelSelect: document.getElementById('model-select'),
    enableVision: document.getElementById('enable-vision'),
    visionGroup: document.getElementById('vision-group'),
    mmprojContainer: document.getElementById('mmproj-container'),
    mmprojSelect: document.getElementById('mmproj-select'),
    enableMtp: document.getElementById('enable-mtp'),
    mtpGroup: document.getElementById('mtp-group'),
    mtpContainer: document.getElementById('mtp-container'),
    mtpSelect: document.getElementById('mtp-select'),
    serverPort: document.getElementById('server-port'),
    gpuSelect: document.getElementById('gpu-select'),
    ctxSize: document.getElementById('ctx-size'),
    threads: document.getElementById('threads'),
    gpuLayers: document.getElementById('gpu-layers'),
    startBtn: document.getElementById('start-btn'),
    comfyuiStatusIndicator: document.getElementById('comfyui-status-indicator'),
    comfyuiGpuSelect: document.getElementById('comfyui-gpu-select'),
    startComfyuiBtn: document.getElementById('start-comfyui-btn'),
    stopComfyuiBtn: document.getElementById('stop-comfyui-btn'),
    openComfyuiLink: document.getElementById('open-comfyui-link'),
    gpuContainer: document.getElementById('gpu-container'),
    instancesList: document.getElementById('instances-list'),

    // Image Gen (ComfyUI)
    comfyModel: document.getElementById('comfy-model'),
    comfyAspectRatio: document.getElementById('comfy-aspect-ratio'),
    comfyResolution: document.getElementById('comfy-resolution'),
    comfyLora: document.getElementById('comfy-lora'),
    comfyLoraStrengthGroup: document.getElementById('comfy-lora-strength-group'),
    comfyLoraStrength: document.getElementById('comfy-lora-strength'),
    comfySteps: document.getElementById('comfy-steps'),
    comfySeed: document.getElementById('comfy-seed'),
    comfyDenoise: document.getElementById('comfy-denoise'),
    comfyDenoiseGroup: document.getElementById('comfy-denoise-group'),
    comfyGenerateBtn: document.getElementById('comfy-generate-btn'),
    comfyPrompt: document.getElementById('comfy-prompt'),
    comfyPromptLabel: document.getElementById('comfy-prompt-label'),
    comfyNegativePrompt: document.getElementById('comfy-negative-prompt'),
    comfyNegativePromptGroup: document.getElementById('comfy-negative-prompt-group'),
    comfyModeGenerateBtn: document.getElementById('comfy-mode-generate-btn'),
    comfyModeEditBtn: document.getElementById('comfy-mode-edit-btn'),
    comfyGenerateOnlyFields: document.getElementById('comfy-generate-only-fields'),
    comfyEditFields: document.getElementById('comfy-edit-fields'),
    comfyEditModel: document.getElementById('comfy-edit-model'),
    comfyEditUploadZone: document.getElementById('comfy-edit-upload-zone'),
    comfyEditInput: document.getElementById('comfy-edit-input'),
    comfyEditPreviewContainer: document.getElementById('comfy-edit-preview-container'),
    comfyEditImgPreview: document.getElementById('comfy-edit-img-preview'),
    comfyEditRemoveBtn: document.getElementById('comfy-edit-remove-btn'),
    comfyEditPlaceholder: document.getElementById('comfy-edit-placeholder'),
    comfyModeStyleBtn: document.getElementById('comfy-mode-style-btn'),
    comfyStyleFields: document.getElementById('comfy-style-fields'),
    comfyStyleModel: document.getElementById('comfy-style-model'),
    comfyStyleContentUploadZone: document.getElementById('comfy-style-content-upload-zone'),
    comfyStyleContentInput: document.getElementById('comfy-style-content-input'),
    comfyStyleContentPreviewContainer: document.getElementById('comfy-style-content-preview-container'),
    comfyStyleContentImgPreview: document.getElementById('comfy-style-content-img-preview'),
    comfyStyleContentRemoveBtn: document.getElementById('comfy-style-content-remove-btn'),
    comfyStyleContentPlaceholder: document.getElementById('comfy-style-content-placeholder'),
    comfyStyleRefUploadZone: document.getElementById('comfy-style-ref-upload-zone'),
    comfyStyleRefInput: document.getElementById('comfy-style-ref-input'),
    comfyStyleRefPreviewContainer: document.getElementById('comfy-style-ref-preview-container'),
    comfyStyleRefImgPreview: document.getElementById('comfy-style-ref-img-preview'),
    comfyStyleRefRemoveBtn: document.getElementById('comfy-style-ref-remove-btn'),
    comfyStyleRefPlaceholder: document.getElementById('comfy-style-ref-placeholder'),
    comfyProgressPanel: document.getElementById('comfy-progress-panel'),
    comfyProgressPercentage: document.getElementById('comfy-progress-percentage'),
    comfyProgressFill: document.getElementById('comfy-progress-fill'),
    comfyProgressNode: document.getElementById('comfy-progress-node'),
    comfyModeBatchBtn: document.getElementById('comfy-mode-batch-btn'),
    comfyBatchFields: document.getElementById('comfy-batch-fields'),
    comfySingleConfigGrid: document.getElementById('comfy-single-config-grid'),
    comfySingleFilenameGroup: document.getElementById('comfy-single-filename-group'),
    comfyFilename: document.getElementById('comfy-filename'),
    comfySingleResults: document.getElementById('comfy-single-results'),
    comfyBatchResults: document.getElementById('comfy-batch-results'),
    comfyBatchFileInput: document.getElementById('comfy-batch-file-input'),
    comfyBatchJson: document.getElementById('comfy-batch-json'),
    comfyBatchStartBtn: document.getElementById('comfy-batch-start-btn'),
    comfyBatchStatusLabel: document.getElementById('comfy-batch-status-label'),
    comfyBatchProgressFill: document.getElementById('comfy-batch-progress-fill'),
    comfyBatchProgressText: document.getElementById('comfy-batch-progress-text'),
    comfyBatchPauseBtn: document.getElementById('comfy-batch-pause-btn'),
    comfyBatchResumeBtn: document.getElementById('comfy-batch-resume-btn'),
    comfyBatchCancelBtn: document.getElementById('comfy-batch-cancel-btn'),
    comfyBatchGallery: document.getElementById('comfy-batch-gallery'),
    comfyLoading: document.getElementById('comfy-loading'),
    imagePlaceholderBox: document.getElementById('image-placeholder-box'),
    comfyResultImg: document.getElementById('comfy-result-img'),

    // Speech (TTS / STT)
    speechServiceTtsBtn: document.getElementById('speech-service-tts-btn'),
    speechServiceSttBtn: document.getElementById('speech-service-stt-btn'),
    speechTtsContainer: document.getElementById('speech-tts-container'),
    speechSttContainer: document.getElementById('speech-stt-container'),
    ttsModeDesignBtn: document.getElementById('tts-mode-design-btn'),
    ttsModeCloneBtn: document.getElementById('tts-mode-clone-btn'),
    ttsLanguage: document.getElementById('tts-language'),
    ttsPreset: document.getElementById('tts-preset'),
    ttsDesignFields: document.getElementById('tts-design-fields'),
    ttsDesignInstructionContainer: document.getElementById('tts-design-instruction-container'),
    ttsDesignTagsContainer: document.getElementById('tts-design-tags-container'),
    ttsInstruct: document.getElementById('tts-instruct'),
    ttsCloneFields: document.getElementById('tts-clone-fields'),
    ttsLocalVoiceSelect: document.getElementById('tts-local-voice-select'),
    ttsAudioUploadArea: document.getElementById('tts-audio-upload-area'),
    ttsAudioInput: document.getElementById('tts-audio-input'),
    ttsAudioPreviewContainer: document.getElementById('tts-audio-preview-container'),
    ttsAudioPreview: document.getElementById('tts-audio-preview'),
    ttsRemoveAudio: document.getElementById('tts-remove-audio'),
    ttsAudioUploadPlaceholder: document.getElementById('tts-audio-upload-placeholder'),
    ttsRefText: document.getElementById('tts-ref-text'),
    ttsText: document.getElementById('tts-text'),
    ttsGenerateBtn: document.getElementById('tts-generate-btn'),
    ttsLoading: document.getElementById('tts-loading'),
    ttsResultContainer: document.getElementById('tts-result-container'),
    ttsResultAudio: document.getElementById('tts-result-audio'),
    sttLocalVoiceSelect: document.getElementById('stt-local-voice-select'),
    sttAudioUploadArea: document.getElementById('stt-audio-upload-area'),
    sttAudioInput: document.getElementById('stt-audio-input'),
    sttAudioPreviewContainer: document.getElementById('stt-audio-preview-container'),
    sttAudioPreview: document.getElementById('stt-audio-preview'),
    sttRemoveAudio: document.getElementById('stt-remove-audio'),
    sttAudioUploadPlaceholder: document.getElementById('stt-audio-upload-placeholder'),
    sttTranscriptResult: document.getElementById('stt-transcript-result'),
    sttCopyBtn: document.getElementById('stt-copy-btn'),
    sttTranscribeBtn: document.getElementById('stt-transcribe-btn'),
    sttLoading: document.getElementById('stt-loading'),

    // Does It Work? (smoke test)
    smoketestGpuSelect: document.getElementById('smoketest-gpu-select'),
    runSmoketestBtn: document.getElementById('run-smoketest-btn'),
    smoketestProgressPanel: document.getElementById('smoketest-progress-panel'),
    smoketestProgressBar: document.getElementById('smoketest-progress-bar'),
    smoketestModelTitle: document.getElementById('smoketest-model-title'),
    smoketestPercentage: document.getElementById('smoketest-percentage'),
    smoketestStatusText: document.getElementById('smoketest-status-text'),
    smoketestResultsList: document.getElementById('smoketest-results-list'),

    // Dev Docs
    apiExplorerList: document.getElementById('api-explorer-list'),
    copyFullApiDocsBtn: document.getElementById('copy-full-api-docs-btn')
};

// Core API Fetch Helper
async function api(path, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    if (body) {
        options.body = JSON.stringify(body);
    }
    const resp = await fetch(`${BASE_URL}${path}`, options);
    if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${resp.status}`);
    }
    return resp.json();
}

// System Logs console logging
function addLog(msg, type = 'system') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${msg}`;
    UI.logsViewport.appendChild(entry);
    UI.logsViewport.scrollTop = UI.logsViewport.scrollHeight;
}

// Clipboard helper: navigator.clipboard is only available in "secure contexts"
// (https:// or http://localhost) — opening this dashboard over a plain
// http:// LAN/Tailscale IP (e.g. from another device) makes it undefined, so
// every copy button would otherwise silently no-op. Falls back to the legacy
// execCommand('copy') trick, which still works in that case.
async function copyTextToClipboard(text) {
    if (window.isSecureContext && navigator.clipboard) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            console.error('navigator.clipboard.writeText failed', err);
        }
    }
    try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(textarea);
        if (ok) return true;
    } catch (err) {
        console.error('execCommand copy fallback failed', err);
    }
    return false;
}

function logCopyResult(ok, label) {
    addLog(ok ? `${label} copied to clipboard.` : `Copy failed — clipboard access is blocked on this connection (try opening the dashboard via https:// or localhost).`, ok ? 'system' : 'error');
}

// ----------------------------------------------------
// Tab & Theme Initialization
// ----------------------------------------------------
UI.tabLinks.forEach(link => {
    link.addEventListener('click', () => {
        UI.tabLinks.forEach(l => l.classList.remove('active'));
        UI.tabContents.forEach(c => c.classList.remove('active'));
        
        link.classList.add('active');
        const tab = link.getAttribute('data-tab');
        activeTab = tab;
        document.getElementById(`${tab}-tab`).classList.add('active');
        
        if (tab === 'compute') {
            pollSmoketestStatus();
        }
        if (typeof setServicesPolling === 'function') {
            setServicesPolling(tab === 'services');
        }
        if (typeof setDevicesPolling === 'function') {
            setDevicesPolling(tab === 'compute');
        }
        if (typeof setComputePolling === 'function') {
            setComputePolling(tab === 'compute');
        }
        if (typeof setOverviewPolling === 'function') {
            setOverviewPolling(tab === 'overview');
        }
    });
});

// Buttons elsewhere on a page that open another page ("Zur App-Übersicht" ...).
function gotoTab(tab) {
    const link = document.querySelector(`.tab-link[data-tab="${tab}"]`);
    if (link) link.click();
}
document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-goto]');
    if (b) gotoTab(b.dataset.goto);
});

UI.themeToggle.addEventListener('click', () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    addLog(`Switched to ${currentTheme} theme.`, 'system');
});

// Set initial theme
document.documentElement.setAttribute('data-theme', currentTheme);

// ----------------------------------------------------
// Servers & GPU Logic
// ----------------------------------------------------
async function loadModels() {
    try {
        const data = await api('/api/models');
        if (data.models) data.models.sort((a, b) => a.name.localeCompare(b.name, undefined, {sensitivity: 'base'}));
        if (data.vision_modules) data.vision_modules.sort((a, b) => a.name.localeCompare(b.name, undefined, {sensitivity: 'base'}));
        lastApiData.models = data;
        
        const modelBadges = m => `${(m.mtp_draft_path || m.mtp_bundled) ? ' (MTP)' : ''}${(m.mmproj_path || m.vision_bundled) ? ' (Vision)' : ''}`;

        // Populate model launch select
        UI.modelSelect.innerHTML = data.models.length
            ? data.models.map(m => `<option value="${m.path}">${m.name}${modelBadges(m)}</option>`).join('')
            : '<option value="" disabled>No models found</option>';

        // Populate vision modules
        if (data.vision_modules && data.vision_modules.length > 0) {
            UI.visionGroup.style.display = 'block';
            UI.mmprojSelect.innerHTML = data.vision_modules.map(m => `<option value="${m.path}">${m.name}</option>`).join('');
            autoMatchVisionModule();
        } else {
            UI.visionGroup.style.display = 'none';
        }

        // Populate MTP draft heads
        if (data.draft_modules) data.draft_modules.sort((a, b) => a.name.localeCompare(b.name, undefined, {sensitivity: 'base'}));
        if (data.draft_modules && data.draft_modules.length > 0) {
            UI.mtpGroup.style.display = 'block';
            UI.mtpSelect.innerHTML = data.draft_modules.map(m => `<option value="${m.path}">${m.name}</option>`).join('');
            autoMatchDraftModule();
        } else {
            UI.mtpGroup.style.display = 'none';
        }

        renderAPIExplorer();
    } catch (err) {
        UI.modelSelect.innerHTML = '<option value="" disabled>Failed to load models</option>';
        addLog(`Failed to load models: ${err.message}`, 'error');
    }
}

// Manage vision dropdown trigger
UI.enableVision.addEventListener('change', (e) => {
    UI.mmprojContainer.style.display = e.target.checked ? 'block' : 'none';
});

// Manage MTP draft-head dropdown trigger
UI.enableMtp.addEventListener('change', (e) => {
    UI.mtpContainer.style.display = e.target.checked ? 'block' : 'none';
});

// Pre-select MTP for the currently chosen model: either a matching sidecar draft head
// (e.g. mtp-gemma-4-12b-it.gguf pairs with gemma-4-12b-it-UD-Q4_K_XL.gguf) or, if the
// model bundles its own nextn tensors (e.g. Qwythos-9B-v2-MTP-Q4_K_M.gguf), no separate
// file is needed at all — --spec-type draft-mtp alone is enough.
function autoMatchDraftModule() {
    const data = lastApiData.models;
    if (!data) return;
    const modelPath = UI.modelSelect.value;
    const selectedModel = (data.models || []).find(m => m.path === modelPath);

    const match = (data.draft_modules || []).find(d => {
        const stem = d.name.toLowerCase().replace(/\.gguf$/, '').replace(/^mtp-/, '');
        const modelName = (UI.modelSelect.selectedOptions[0]?.textContent || '').toLowerCase();
        return stem && modelName.startsWith(stem);
    });

    if (match) {
        UI.enableMtp.checked = true;
        UI.mtpContainer.style.display = 'block';
        UI.mtpSelect.value = match.path;
    } else if (selectedModel?.mtp_bundled) {
        UI.enableMtp.checked = true;
        UI.mtpContainer.style.display = 'none';
        UI.mtpSelect.value = '';
    } else {
        UI.enableMtp.checked = false;
        UI.mtpContainer.style.display = 'none';
        UI.mtpSelect.value = '';
    }
}
UI.modelSelect.addEventListener('change', autoMatchDraftModule);

// Pre-select vision for the currently chosen model: either an auto-matched mmproj
// sidecar (e.g. gemma-4-12b-it-mmproj-F32.gguf pairs with gemma-4-12b-it-UD-Q4_K_XL.gguf)
// or, for architectures that bundle their own vision encoder, no mmproj file is needed
// at all — llama-server picks it up from the model's own tensors.
function autoMatchVisionModule() {
    const data = lastApiData.models;
    if (!data) return;
    const modelPath = UI.modelSelect.value;
    const selectedModel = (data.models || []).find(m => m.path === modelPath);

    if (selectedModel?.mmproj_path) {
        UI.enableVision.checked = true;
        UI.mmprojContainer.style.display = 'block';
        UI.mmprojSelect.value = selectedModel.mmproj_path;
    } else if (selectedModel?.vision_bundled) {
        UI.enableVision.checked = true;
        UI.mmprojContainer.style.display = 'none';
        UI.mmprojSelect.value = '';
    } else {
        UI.enableVision.checked = false;
        UI.mmprojContainer.style.display = 'none';
        UI.mmprojSelect.value = '';
    }
}
UI.modelSelect.addEventListener('change', autoMatchVisionModule);

// Launch manual llama-server
UI.startBtn.addEventListener('click', async () => {
    const modelPath = UI.modelSelect.value;
    if (!modelPath) return addLog('Please select a model.', 'warn');
    
    UI.startBtn.disabled = true;
    addLog(`Initiating start request for port ${UI.serverPort.value}...`, 'system');
    
    try {
        const payload = {
            model_path: modelPath,
            port: parseInt(UI.serverPort.value),
            ctx_size: parseInt(UI.ctxSize.value),
            threads: parseInt(UI.threads.value),
            n_gpu_layers: parseInt(UI.gpuLayers.value),
            gpu_id: UI.gpuSelect.value !== "" ? parseInt(UI.gpuSelect.value) : 0
        };
        if (UI.enableVision.checked && UI.mmprojSelect.value) {
            payload.mmproj_path = UI.mmprojSelect.value;
        }
        if (UI.enableMtp.checked && UI.mtpSelect.value) {
            payload.draft_model_path = UI.mtpSelect.value;
        } else {
            // Backend auto-enables MTP by default (sidecar pairing or bundled nextn
            // tensors) whenever available, so unchecking must explicitly say "off" —
            // otherwise a bundled-capable model would ignore the unchecked box.
            payload.mtp = UI.enableMtp.checked;
        }

        await api('/api/start', 'POST', payload);
        addLog(`Launch request successful on port ${UI.serverPort.value}. Waiting for socket.`, 'info');
        // increment default port
        UI.serverPort.value = parseInt(UI.serverPort.value) + 1;
    } catch (err) {
        addLog(`Launch failed: ${err.message}`, 'error');
    } finally {
        UI.startBtn.disabled = false;
        pollSystemStatus();
    }
});

// Unload Model instance
async function stopInstance(port) {
    addLog(`Requesting stop for model on port ${port}...`, 'system');
    try {
        await api(`/api/stop?port=${port}`, 'POST');
        addLog(`Server on port ${port} stopped.`, 'info');
    } catch (err) {
        addLog(`Failed to stop server: ${err.message}`, 'error');
    } finally {
        pollSystemStatus();
    }
}

// None of the four TTS engines (OmniVoice, Qwen3-TTS, CosyVoice, Chatterbox) have a
// port the dashboard can key on -- OmniVoice/WhisperX are in-process, and Qwen3-TTS/
// CosyVoice/Chatterbox run under isolated venvs whose worker port is an implementation
// detail (see CLAUDE.md's TTS section). So none of them can go through stopInstance()'s
// /api/stop?port= path -- each has its own dedicated unload endpoint instead
// (/api/tts/unload, /api/tts/qwen3/unload, /api/tts/cosyvoice/unload,
// /api/tts/chatterbox/unload). Rather than hardcode a function+button per engine
// (which is how CosyVoice/Chatterbox ended up with no Stop button at all -- the old
// per-engine model-name string matching here never covered them), /api/status's
// synthetic instance entry for each engine now carries its own unload_endpoint, and
// this one generic function drives all four. ComfyUI's entry uses it too
// (/api/comfyui/unload: frees VRAM, keeps the process) even though it has a port.
async function unloadTtsEngine(endpoint) {
    addLog(`Requesting unload (${endpoint})...`, 'system');
    try {
        await api(endpoint, 'POST');
        addLog('Unloaded.', 'info');
    } catch (err) {
        addLog(`Failed to unload: ${err.message}`, 'error');
    } finally {
        pollSystemStatus();
    }
}

// Unload all model instances, optionally scoped to a single GPU
async function stopAllInstances(gpuId) {
    const scoped = gpuId !== undefined && gpuId !== null;
    addLog(scoped
        ? `Requesting shutdown for all model servers on GPU ${gpuId}...`
        : 'Requesting shutdown for all active model servers...', 'system');
    try {
        const url = scoped ? `/api/models/unload_all?gpu_id=${gpuId}` : '/api/models/unload_all';
        const res = await api(url, 'POST');
        addLog(res.message || 'Models stopped.', 'info');
    } catch (err) {
        addLog(`Failed to stop models: ${err.message}`, 'error');
    } finally {
        pollSystemStatus();
    }
}

window.stopInstance = stopInstance;
window.stopAllInstances = stopAllInstances;


// ComfyUI launch controls
UI.startComfyuiBtn.addEventListener('click', async () => {
    UI.startComfyuiBtn.disabled = true;
    const gpuVal = UI.comfyuiGpuSelect.value;
    const payload = { port: 8189 };
    // Omit entirely rather than defaulting to 0 -- unlike llama-server's gpu_id, an
    // omitted ComfyUI gpu_id doesn't mean "auto-pick the most-free GPU", it means
    // "leave every GPU visible" (PyTorch's own default of cuda:0 among them). See
    // ComfyUIStartRequest.gpu_id / CLAUDE.md's ComfyUI GPU selection section.
    if (gpuVal !== "") payload.gpu_id = parseInt(gpuVal);
    addLog(`Starting ComfyUI Server in background${gpuVal !== "" ? ` (GPU ${gpuVal})` : ""}...`, 'system');
    try {
        const resp = await api('/api/comfyui/start', 'POST', payload);
        if (resp.status === 'already running' && gpuVal !== "" && resp.gpu_id !== parseInt(gpuVal)) {
            addLog(`ComfyUI was already running on GPU ${resp.gpu_id !== null ? resp.gpu_id : 'Auto'} — stop it first to move it to GPU ${gpuVal}.`, 'warn');
        } else {
            addLog(`ComfyUI server started${resp.gpu_id !== null && resp.gpu_id !== undefined ? ` on GPU ${resp.gpu_id}` : ''}.`, 'info');
        }
    } catch (err) {
        addLog(`Failed to start ComfyUI: ${err.message}`, 'error');
    } finally {
        UI.startComfyuiBtn.disabled = false;
        pollSystemStatus();
    }
});

UI.stopComfyuiBtn.addEventListener('click', async () => {
    UI.stopComfyuiBtn.disabled = true;
    addLog('Stopping ComfyUI Server...', 'system');
    try {
        await api('/api/comfyui/stop', 'POST');
        addLog('ComfyUI server stopped.', 'info');
    } catch (err) {
        addLog(`Failed to stop ComfyUI: ${err.message}`, 'error');
    } finally {
        UI.stopComfyuiBtn.disabled = false;
        pollSystemStatus();
    }
});

// Render dynamic GPU Resources
function renderGPUContainer(gpus) {
    if (!gpus.length) {
        UI.gpuContainer.innerHTML = '<div class="empty-state">No GPUs detected.</div>';
        return;
    }
    UI.gpuContainer.innerHTML = gpus.map(g => {
        const percent = Math.round(g.usage_percent);
        return `
            <div class="gpu-item">
                <div class="gpu-name">
                    <span>GPU ${g.index}: ${g.name}</span>
                    <span>${percent}%</span>
                </div>
                <div class="vram-bar">
                    <div class="vram-fill" style="width: ${percent}%"></div>
                </div>
                <div class="vram-text">
                    <span>Used: ${Math.round(g.used)} MB</span>
                    <span>Total: ${Math.round(g.total)} MB</span>
                </div>
                <button class="btn btn-danger" style="margin-top: 0.4rem; padding: 0.2rem 0.5rem; font-size: 0.7rem; min-height: 0;" onclick="stopAllInstances(${g.index})">Unload GPU ${g.index}</button>
            </div>
        `;
    }).join('');

    // Sync gpu selector lists
    const optionsHtml = gpus.map(g => `<option value="${g.index}">GPU ${g.index}: ${g.name}</option>`).join('');
    
    // Save selections
    const prevGpuVal = UI.gpuSelect.value;
    const prevSmoketestGpuVal = UI.smoketestGpuSelect.value;
    const prevComfyuiGpuVal = UI.comfyuiGpuSelect.value;

    UI.gpuSelect.innerHTML = `<option value="">Auto (All)</option>${optionsHtml}`;
    UI.smoketestGpuSelect.innerHTML = optionsHtml;
    UI.comfyuiGpuSelect.innerHTML = `<option value="">Auto (All)</option>${optionsHtml}`;

    if (prevGpuVal) UI.gpuSelect.value = prevGpuVal;
    if (prevSmoketestGpuVal) UI.smoketestGpuSelect.value = prevSmoketestGpuVal;
    if (prevComfyuiGpuVal) UI.comfyuiGpuSelect.value = prevComfyuiGpuVal;
}

// Render dynamic Active Model Servers
function renderInstancesList(instances) {
    if (!instances.length) {
        UI.instancesList.innerHTML = '<div class="empty-state">No active servers. Managed model ports will show here.</div>';
    } else {
        UI.instancesList.innerHTML = instances.map(inst => {
            // inst.generating is ground truth from the instance's own /slots endpoint
            // (falls back to the gateway's active_requests counter if /slots was
            // unreachable) -- true even when traffic hits the instance's port
            // directly, bypassing this app's gateway entirely.
            const generating = inst.ready && inst.generating;
            const statusLabel = !inst.ready ? 'Starting' : (generating ? `Generating${inst.tps ? ` (${inst.tps} TPS)` : ''}` : 'Ready');
            const statusClass = !inst.ready ? 'omitted' : (generating ? 'badge-generating' : 'success');
            return `
            <div class="instance-card ${inst.ready ? 'ready' : ''}">
                <span class="badge ${statusClass} status-chip">
                    ${statusLabel}
                </span>
                <div class="model-name">${inst.model.split('/').pop()}</div>
                <div class="details">
                    ${inst.port !== null && inst.port !== undefined ? `<div>Port: <span class="badge-port">${inst.port}</span></div>` : ''}
                    <div>PID: ${inst.pid || '-'}</div>
                    <div>GPU layers: ${inst.n_gpu_layers} (GPU: ${inst.gpu_id !== null ? inst.gpu_id : 'Auto'})</div>
                    ${inst.ctx_size ? `<div>Context: ${inst.ctx_size.toLocaleString()} tokens</div>` : ''}
                    ${inst.reranking ? '<div>Mode: <span class="badge badge-primary" style="font-size:0.7rem; padding:0.1rem 0.3rem;">Reranking</span></div>' : (inst.embedding ? '<div>Mode: <span class="badge badge-primary" style="font-size:0.7rem; padding:0.1rem 0.3rem;">Embedding</span></div>' : '')}
                    ${inst.mtp_active ? '<div>Speculative: <span class="badge badge-primary" style="font-size:0.7rem; padding:0.1rem 0.3rem;">MTP</span></div>' : ''}
                </div>
                ${inst.port !== null && inst.port !== undefined && !inst.unload_endpoint ? `
                <button class="btn btn-danger btn-full" style="margin-top: 1rem; padding: 0.4rem 0.8rem; font-size: 0.8rem;" onclick="stopInstance(${inst.port})">
                    Stop & Unload
                </button>` : (inst.unload_endpoint ? (
                    // The unload endpoints for all four TTS engines deliberately refuse
                    // (409) while loading/generating, to avoid yanking an in-flight
                    // request -- see CLAUDE.md's TTS section. Without this, clicking
                    // Stop & Unload during that window silently 409s: the click "does
                    // nothing" from the user's point of view unless they happen to be
                    // watching the Logs panel for the resulting error line. Disabling
                    // the button (and explaining why via its title tooltip) instead of
                    // leaving it clickable makes that state legible up front.
                    !inst.ready || inst.generating ? `
                <button class="btn btn-danger btn-full" style="margin-top: 1rem; padding: 0.4rem 0.8rem; font-size: 0.8rem;" disabled title="Can't unload while ${!inst.ready ? 'still loading' : 'generating'} — wait for it to finish, or let the keep-alive timeout unload it automatically.">
                    Stop & Unload
                </button>` : `
                <button class="btn btn-danger btn-full" style="margin-top: 1rem; padding: 0.4rem 0.8rem; font-size: 0.8rem;" onclick="unloadTtsEngine('${inst.unload_endpoint}')">
                    Stop & Unload
                </button>`
                ) : '')}
            </div>
        `;
        }).join('');
    }

    // Render in the Gateway Chat sidebar
    const gatewayCard = document.getElementById('gateway-active-models-card');
    const gatewayList = document.getElementById('gateway-active-models-list');
    if (gatewayCard && gatewayList) {
        if (!instances.length) {
            gatewayCard.style.display = 'none';
            gatewayList.innerHTML = '';
        } else {
            gatewayCard.style.display = 'block';
            gatewayList.innerHTML = instances.map(inst => `
                <div style="background: rgba(148, 163, 184, 0.08); border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem; display: flex; flex-direction: column; gap: 0.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;">
                        <span style="font-size: 0.8rem; font-weight: 600; word-break: break-all; color: var(--text-main); line-height: 1.2;">${inst.model.split('/').pop()}</span>
                        <span class="badge ${!inst.ready ? 'omitted' : (inst.generating ? 'badge-generating' : 'success')}" style="font-size: 0.65rem; padding: 0.1rem 0.35rem; flex-shrink: 0;">
                            ${!inst.ready ? 'Starting' : (inst.generating ? 'Generating' : 'Ready')}
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; margin-top: 0.25rem;">
                        <span>${inst.port !== null && inst.port !== undefined ? `Port: ${inst.port} ` : ''}${inst.reranking ? '(Rerank)' : (inst.embedding ? '(Emb)' : '')}${inst.mtp_active ? ' (MTP)' : ''}</span>
                        ${inst.port !== null && inst.port !== undefined ? `
                        <button class="btn btn-danger" style="padding: 0.2rem 0.5rem; font-size: 0.7rem; min-height: 0;" onclick="stopInstance(${inst.port})">
                            Stop
                        </button>` : ''}
                    </div>
                </div>
            `).join('');
        }
    }
}

// ----------------------------------------------------
// Does It Work? (smoke test) Logic
// ----------------------------------------------------
UI.runSmoketestBtn.addEventListener('click', startSmoketest);

async function startSmoketest() {
    UI.runSmoketestBtn.disabled = true;
    UI.smoketestProgressPanel.style.display = 'block';
    UI.smoketestResultsList.innerHTML = '';
    addLog('Starting "does it work?" check...', 'system');

    try {
        await api('/api/smoke-test/start', 'POST', {
            gpu_id: parseInt(UI.smoketestGpuSelect.value) || 0
        });
        if (smoketestPollInterval) clearInterval(smoketestPollInterval);
        smoketestPollInterval = setInterval(pollSmoketestStatus, 2000);
    } catch (err) {
        addLog(`Smoke test start failed: ${err.message}`, 'error');
        UI.runSmoketestBtn.disabled = false;
        UI.smoketestProgressPanel.style.display = 'none';
    }
}

function renderSmoketestResults(results) {
    if (!results || !results.length) return;
    UI.smoketestResultsList.innerHTML = results.map(r => {
        const cls = r.status === 'passed' ? 'passed' : r.status === 'failed' ? 'failed' : 'error';
        const detail = r.status === 'error' ? (r.error || '') : (r.answer || '');
        return `
            <div class="card response-card" style="padding: 0.75rem 1rem; display: flex; align-items: center; gap: 1rem;">
                <span class="badge ${cls}" style="flex-shrink: 0;">${r.status.toUpperCase()}</span>
                <div style="flex: 1; min-width: 0;">
                    <div style="font-weight: 700;">${r.model}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); white-space: pre-wrap;">${detail}</div>
                </div>
            </div>`;
    }).join('');
}

async function pollSmoketestStatus() {
    try {
        const data = await api('/api/smoke-test/status');
        const prog = data.progress;

        UI.smoketestModelTitle.textContent = prog.model || 'Initializing...';
        UI.smoketestStatusText.textContent = `Status: ${prog.status} (Model ${prog.current} of ${prog.total})`;

        const percent = prog.total > 0 ? (prog.current / prog.total) * 100 : 0;
        UI.smoketestProgressBar.style.width = `${percent}%`;
        UI.smoketestPercentage.textContent = `${Math.round(percent)}%`;

        renderSmoketestResults(data.results);

        if (!data.running && prog.status === 'completed') {
            clearInterval(smoketestPollInterval);
            addLog('Smoke test completed.', 'success');
            UI.runSmoketestBtn.disabled = false;
        } else if (!data.running && (prog.status || '').startsWith('failed')) {
            clearInterval(smoketestPollInterval);
            addLog('Smoke test run failed.', 'error');
            UI.runSmoketestBtn.disabled = false;
        }
    } catch (err) {
        console.error('Smoke test polling error', err);
    }
}

// ----------------------------------------------------
// Developer Explorer Render
// ----------------------------------------------------
async function copyFullApiDocs() {
    try {
        const data = await api('/api/docs/raw');
        if (data && data.content) {
            const ok = await copyTextToClipboard(data.content);
            logCopyResult(ok, 'Full API documentation (docs/API.md)');
            const btn = UI.copyFullApiDocsBtn;
            if (ok && btn) {
                const origText = btn.textContent;
                btn.textContent = '✓ Copied!';
                setTimeout(() => { btn.textContent = origText; }, 2000);
            }
        }
    } catch (e) {
        addLog(`Failed to fetch API documentation: ${e.message}`, 'error');
    }
}

function copySectionApiDocs(category) {
    const categoryNames = {
        text: 'Text & Language Models (LLMs & Gateway)',
        image: 'Image Generation & Editing (ComfyUI)',
        audio: 'Speech & Audio (TTS & STT)',
        training: 'Remote Fine-Tuning'
    };
    const title = categoryNames[category] || category;
    let md = `# AI-Serve API Specification - ${title}\n\n`;
    
    const categoryEndpoints = window.apiEndpointsList ? window.apiEndpointsList.filter(ep => ep.category === category) : [];
    
    categoryEndpoints.forEach(ep => {
        md += `## ${ep.method} ${ep.path}\n`;
        md += `${ep.desc}\n\n`;
        if (ep.curl) {
            md += `\`\`\`bash\n${ep.curl}\n\`\`\`\n\n`;
        }
    });

    copyTextToClipboard(md).then(ok => logCopyResult(ok, `${title} API specs`));
}
window.copySectionApiDocs = copySectionApiDocs;

// Builds one field entry for the structured .api-field-list format -- an alternative to
// a flat <ul><li> wall of text for schemas dense enough to need a visually separated
// name/type/primary-description/detail-note per field. `note` is optional supplementary
// detail (caveats, edge cases) rendered dimmer and indented, so the one-line gist of a
// field is scannable without the surrounding nuance competing for the same visual weight.
function apiField(name, type, desc, note) {
    return `
        <div class="api-field">
            <div class="api-field-head"><code>${name}</code><span class="api-field-type">${type}</span></div>
            <div class="api-field-desc">${desc}</div>
            ${note ? `<div class="api-field-note">${note}</div>` : ''}
        </div>`;
}

function apiFieldList(fields) {
    return `<div class="api-field-list">${fields.join('')}</div>`;
}

// Renders a labeled set of example payloads/snippets for an endpoint -- e.g. structured
// output, thinking control, passthrough sampler params for /v1/chat/completions. Each
// example is {title, code} where code is pre-formatted text (JSON or curl-ish).
function apiExamples(examples) {
    return `<div class="api-examples">${examples.map(ex => `
        <div>
            <div class="api-example-label">${ex.title}</div>
            <div class="api-payload-box"><pre>${ex.code}</pre></div>
        </div>
    `).join('')}</div>`;
}

function renderAPIExplorer() {
    if (!UI.apiExplorerList) return;

    if (UI.copyFullApiDocsBtn && !UI.copyFullApiDocsBtn._bound) {
        UI.copyFullApiDocsBtn._bound = true;
        UI.copyFullApiDocsBtn.addEventListener('click', copyFullApiDocs);
    }

    const endpoints = [
        {
            category: 'text',
            method: 'GET',
            path: '/api/models',
            desc: 'List all available GGUF models and vision modules.',
            responseSchema: '<ul><li><code>models</code> (array): List of GGUF model files. Each item has <code>name</code> (filename), <code>path</code> (absolute path), <code>mtp_draft_path</code> (absolute path to a paired MTP speculative-decoding sidecar draft head, or <code>null</code>), <code>mtp_bundled</code> (boolean: true if this model\'s own GGUF already bundles its MTP/nextn tensors, e.g. a "-MTP-" quant, so no separate draft file is needed), <code>mmproj_path</code> (absolute path to an auto-matched vision projector sidecar, or <code>null</code>), and <code>vision_bundled</code> (boolean: true if this model\'s own GGUF already bundles a vision encoder, so no separate mmproj file is needed).</li><li><code>vision_modules</code> (array): List of vision projector files (mmproj). Each item has <code>name</code> and <code>path</code>.</li><li><code>draft_modules</code> (array): List of standalone MTP speculative-decoding sidecar draft head files. Each item has <code>name</code> and <code>path</code>.</li></ul>',
            liveResponse: lastApiData.models,
            curl: `curl ${BASE_URL}/api/models`
        },
        {
            category: 'text',
            method: 'GET',
            path: '/api/models/header',
            desc: 'Read a GGUF model\'s full raw metadata header — architecture, context length, quantization, rope/vision/MTP config, chat template, sampling defaults, etc.',
            responseSchema: '<ul><li><code>gguf_version</code> (int), <code>tensor_count</code> (int), <code>kv_count</code> (int).</li><li><code>metadata</code> (object): every KV pair from the header, keyed by its GGUF key (e.g. <code>general.architecture</code>, <code>llama.context_length</code>). Large arrays (e.g. the tokenizer vocab) are collapsed to <code>{"_type": "array", "length": N, "preview": [...]}</code>; very long strings are truncated with a trailing note.</li></ul>',
            payloadSchema: '<ul><li><code>path</code> (string, one of path/model_name required): absolute path to a .gguf file inside a configured model directory.</li><li><code>model_name</code> (string, one of path/model_name required): filename, stem, or fuzzy substring — resolved against every GGUF under the configured model directories (main models, mmproj sidecars, and MTP draft heads alike).</li></ul>',
            curl: `curl "${BASE_URL}/api/models/header?model_name=gemma-4-12b"`
        },
        {
            category: 'text',
            method: 'GET',
            path: '/api/version',
            desc: 'Backend performance/capability version — a plain incrementing integer, separate from the API/release version. Bumped whenever a change could plausibly affect inference speed or serving capability of the model backend (llama.cpp binary upgrades, MTP/speculative-decoding support, MLX support, GPU-selection logic, KV-cache behavior). Not bumped for UI-only, ComfyUI, or documentation changes. Also reports host OS/RAM/RAM-architecture, every detected GPU, and rolls up sub-component versions (currently llama.cpp; the same "components" object is where future ones like MLX would go) so this one call tells you everything about what a deployment is actually running.',
            responseSchema: '<ul><li><code>version</code> (int): current backend version.</li><li><code>changed</code> (string): description of what changed at the current version.</li><li><code>date</code> (string): date of the current version.</li><li><code>log</code> (array of {version, date, change}): full history, newest first.</li><li><code>os</code> (string): e.g. "Linux".</li><li><code>os_release</code> (string): e.g. "6.18.7-76061807-generic".</li><li><code>os_version</code> (string): full kernel/build version string.</li><li><code>ram_total_mb</code> (int): total system RAM in MB.</li><li><code>ram_architecture</code> (string): "unified" on Apple Silicon (RAM doubles as GPU memory) or "discrete" everywhere else (separate system RAM and VRAM pools, e.g. NVIDIA).</li><li><code>gpus</code> (array): same shape as <code>GET /api/gpu/status</code>\'s <code>gpus</code> (name, <code>driver_version</code>, total/used/free VRAM, usage_percent), plus a <code>runtime</code> object per GPU: <code>{vendor, kind, version}</code>, e.g. <code>{"vendor": "nvidia", "kind": "cuda", "version": "13.0"}</code>. Detected per-GPU by name (not once globally), so a mixed-vendor box — one NVIDIA + one AMD card, say — reports the correct stack for each card. <code>kind</code> is "cuda" (NVIDIA), "rocm" (AMD, via <code>/opt/rocm/.info/version</code> or <code>rocm-smi</code>), "level_zero" (Intel, best-effort via <code>clinfo</code>), "metal" (Apple), or null if the card\'s vendor couldn\'t be identified from its name.</li><li><code>components</code> (object): sub-component versions, e.g. <code>components.llamacpp</code> — <code>{binary, build_number, commit, raw, error}</code> parsed from <code>llama-server --version</code>.</li></ul>',
            liveResponse: lastApiData.version,
            curl: `curl ${BASE_URL}/api/version`
        },
        {
            category: 'text',
            method: 'GET',
            path: '/api/status',
            desc: 'Get status of active server instances.',
            responseSchema: '<ul><li><code>instances</code> (array): List of running llama-server instances, plus ComfyUI and OmniVoice (TTS) as synthetic entries whenever they\'re running/loaded. Each instance contains <code>port</code> (int or null — null for OmniVoice, which is an in-process lazy-loaded model with no subprocess/port of its own), <code>running</code>, <code>ready</code>, <code>model</code>, <code>pid</code>, <code>n_gpu_layers</code>, <code>gpu_id</code>, <code>ctx_size</code> (loaded context window in tokens), <code>embedding</code> (bool), <code>reranking</code> (bool: this instance was started with <code>--reranking</code>, implies <code>embedding</code>), <code>pooling</code> (string or null: the explicit <code>--pooling</code> override this instance was started with, if any — null means it\'s trusting the GGUF\'s own metadata), <code>draft_model</code>/<code>mtp_active</code> (MTP speculative decoding, for models/configs that support it), <code>active_requests</code> (int, in-flight chat/completions calls this app itself mediated through the gateway — 0 for traffic that hits a llama-server instance\'s own port directly), <code>generating</code> (bool, ground truth: for llama-server instances, polled live from the instance\'s own <code>/slots</code> endpoint so it\'s accurate even for traffic that bypasses this app\'s gateway; for ComfyUI, derived from its own pending/processing task count; for OmniVoice, from its own generation-in-progress state), <code>tps</code> (float or null, live decode speed derived by diffing token counts between polls while generating is true — llama-server instances only, always null for ComfyUI/OmniVoice), and a <code>health</code> check status object.</li></ul>',
            liveResponse: lastApiData.status,
            curl: `curl ${BASE_URL}/api/status`
        },
        {
            category: 'text',
            method: 'GET',
            path: '/api/gpu/status',
            desc: 'Get NVIDIA GPU hardware metrics.',
            responseSchema: '<ul><li><code>gpus</code> (array): List of GPU status blocks. Each contains <code>index</code> (GPU ID), <code>name</code>, <code>total</code> (VRAM in MB), <code>used</code> (VRAM in MB), <code>free</code> (VRAM in MB), and <code>usage_percent</code>.</li></ul>',
            liveResponse: lastApiData.gpu,
            curl: `curl ${BASE_URL}/api/gpu/status`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/v1/chat/completions',
            desc: 'OPENAI COMPATIBLE: Chat completions with on-the-fly model loading, multimodal vision input, audio transcription input, base64 audio output, and thinking reasoning content.',
            payload: {
                model: "Captain_Eris_Violet-0.420-abliterated.i1-Q5_K_M.gguf",
                messages: [{"role": "user", "content": "What is the capital of France?"}],
                temperature: 0.7,
                max_tokens: 2048,
                thinking: true
            },
            payloadSchema: apiFieldList([
                apiField('model', 'string, required', 'GGUF model name or absolute path.'),
                apiField('messages', 'array, required', 'Conversation context. Each message has <code>role</code> ("system", "user", "assistant") and <code>content</code> (string or array of content parts).'),
                apiField('temperature', 'float, optional', 'Sampling temperature.'),
                apiField('max_tokens', 'int, optional', 'Maximum generated tokens limit.'),
                apiField('thinking', 'boolean, optional', 'Enable/disable reasoning.',
                    'Mapped to llama-server\'s native <code>chat_template_kwargs.enable_thinking</code> — the reliable mechanism for templates that bake thinking in by default (e.g. gemma-4-12b-it, which otherwise reasons at length on every request regardless of a plain "don\'t think" instruction). A matching soft system-prompt instruction is also injected as a fallback for models without native <code>enable_thinking</code> support. See the "Disable thinking" example below.'),
                apiField('chat_template_kwargs', 'object, optional', 'Passed straight through to llama-server\'s Jinja chat-template rendering (OpenAI/vLLM-style convention).',
                    'e.g. <code>{"enable_thinking": false}</code> — must be a real JSON boolean, not a string (llama-server returns HTTP 400 for a string value). Takes precedence over <code>thinking</code> above if both set <code>enable_thinking</code>.'),
                apiField('response_format', 'object, optional', 'Structured output JSON Schema.',
                    'Supports the standard OpenAI wrapper (<code>{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}</code>) or a bare JSON Schema object, which the gateway wraps for you. See the "Structured output" example below.'),
                apiField('mtp', 'boolean, optional', 'Force-enable or force-disable MTP speculative decoding for this request.',
                    'Default (omitted/null): auto-enabled when the resolved model has a paired MTP draft head — see <code>mtp_draft_path</code> in <code>/api/models</code>.'),
                apiField('ctx_size', 'int, optional, default 32000', 'Context size (<code>-c</code>) to launch the model with, if it needs auto-loading.',
                    'If a matching model is already running with a smaller context, it\'s automatically restarted with this larger one. If it\'s already running with an equal-or-larger context, that instance is reused as-is (a smaller/equal request never evicts a bigger, possibly-warm instance).'),
                apiField('parallel', 'int, optional, default 1', 'llama-server parallel slots (<code>-np</code>) used if this request auto-loads the model.', 'Default from <code>LLAMA_DEFAULT_PARALLEL</code>. More slots cost single-stream speed (up to ~2x for a model that nearly fills the card) — set &gt;1 only if you send concurrent requests to this model; 0 = llama.cpp\'s auto choice (4). Asking for more slots than a running instance has restarts it, but only when idle.'),
                apiField('gpu_id', 'int, optional', 'Which GPU to route this request to.',
                    'Left unset, an already-running instance of this model is matched by model path alone — if the same model is loaded on multiple GPUs, the FIRST one registered is always picked (no load balancing), regardless of its current load; the other GPU\'s instance stays idle. Set this explicitly to target a specific GPU\'s instance. If no instance is running yet, leaving this unset auto-picks the GPU with the most free VRAM.'),
                apiField('<em>any other field</em>', 'optional', 'Forwarded verbatim to llama-server instead of being silently dropped.',
                    'Covers standard OpenAI sampling fields not listed above (<code>stop</code>, <code>seed</code>, <code>top_p</code>, <code>presence_penalty</code>, <code>frequency_penalty</code>, <code>logit_bias</code>, <code>logprobs</code>, <code>top_logprobs</code>) plus llama.cpp\'s own sampler extensions (<code>top_k</code>, <code>min_p</code>, <code>repeat_penalty</code>, <code>mirostat</code>/<code>mirostat_tau</code>/<code>mirostat_eta</code>, <code>dry_multiplier</code> and other <code>dry_*</code> params, <code>xtc_probability</code>/<code>xtc_threshold</code>, <code>grammar</code>, <code>samplers</code>, <code>cache_prompt</code>, <code>reasoning_format</code>, and more — see llama.cpp\'s <code>tools/server/server-schema.cpp</code> for the full, evolving list. See the "Stop sequence + sampler passthrough" example below.'),
            ]),
            examples: [
                {
                    title: 'Structured output (response_format)',
                    code: JSON.stringify({
                        model: "gemma-4-12b-it-UD-Q4_K_XL.gguf",
                        messages: [{ role: "user", content: "Extract the name and age from: Jon is 34 years old." }],
                        response_format: {
                            type: "json_schema",
                            json_schema: {
                                name: "person",
                                strict: true,
                                schema: {
                                    type: "object",
                                    properties: { name: { type: "string" }, age: { type: "integer" } },
                                    required: ["name", "age"]
                                }
                            }
                        }
                    }, null, 2)
                },
                {
                    title: 'Disable thinking (reliable, template-level)',
                    code: JSON.stringify({
                        model: "gemma-4-12b-it-UD-Q4_K_XL.gguf",
                        messages: [{ role: "user", content: "Say hello in one word." }],
                        chat_template_kwargs: { enable_thinking: false }
                    }, null, 2)
                },
                {
                    title: 'Stop sequence + sampler passthrough',
                    code: JSON.stringify({
                        model: "gemma-4-12b-it-UD-Q4_K_XL.gguf",
                        messages: [{ role: "user", content: "Count from 1 to 10." }],
                        stop: ["5"],
                        seed: 42,
                        top_p: 0.9,
                        repeat_penalty: 1.1
                    }, null, 2)
                }
            ],
            responseSchema: '<ul><li>Standard OpenAI chat completion response including <code>choices[].message.content</code> and optional <code>choices[].message.reasoning_content</code>.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "Captain_Eris_Violet-0.420-abliterated.i1-Q5_K_M.gguf",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "thinking": true
  }'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/chat/completions',
            desc: 'UNIFIED GATEWAY (LEGACY): Custom text/vision/audio inference endpoint with model auto-loading, thinking tags, and retries.',
            payload: {
                model_name: "gemma",
                messages: [{"role": "user", "content": "Generate a character."}],
                thinking: true,
                response_format: null,
                retries: 3
            },
            payloadSchema: '<ul><li><code>model_name</code> (string, required): Model search token.</li><li><code>messages</code> (array, required): Conversation history.</li><li><code>thinking</code> (boolean, optional): Enable thinking tags reasoning.</li><li><code>mtp</code> (boolean, optional): Force-enable or force-disable MTP speculative decoding for this request. Default (omitted/null): auto-enabled when the resolved model has a paired MTP draft head.</li><li><code>gpu_id</code> (int, optional): Which GPU to route this request to. Left unset, an already-running instance of this model is matched by model path alone — if the same model is loaded on multiple GPUs, the FIRST one registered is always picked (no load balancing), regardless of its current load; the other GPU\'s instance stays idle. Set this explicitly to target a specific GPU\'s instance. If no instance is running yet, leaving this unset auto-picks the GPU with the most free VRAM.</li><li><code>ctx_size</code> (int, optional, default 32000): Context size (<code>-c</code>) to launch the model with, if it needs auto-loading. If a matching model is already running with a smaller context, it\'s restarted with this larger one; equal-or-larger and it\'s reused as-is.</li><li><code>response_format</code> (object, optional): JSON schema constraint. Supports raw JSON Schema or OpenAI-style <code>json_schema</code> format, automatically mapped for llama.cpp, LM Studio, Ollama, and MLX.</li><li><code>retries</code> (int, optional): Automatic server load/execution retry count.</li><li><code>parallel</code> (int, optional): llama-server parallel slots (<code>-np</code>) used when this request auto-loads the model. Default 1 (<code>LLAMA_DEFAULT_PARALLEL</code>) — more slots cost single-stream speed (up to ~2x for a model that nearly fills the card). Set &gt;1 only if you send concurrent requests to this model; 0 = llama.cpp\'s auto choice (4). Asking for more slots than a running instance has restarts it, but only when idle.</li></ul>',
            responseSchema: '<ul><li>JSON object containing LLM completion and generation meta-metrics.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/chat/completions -H "Content-Type: application/json" -d '{"model_name": "gemma", "messages": [{"role": "user", "content": "Tell me a joke."}]}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/smoke-test/start',
            desc: 'DOES IT WORK?: Optional sanity check — loads each model sequentially on one GPU, asks "What is the capital of Bulgaria?", and checks the reply contains "Sofia". Not a performance benchmark. Defaults to every discovered model if model_paths is omitted.',
            payload: { gpu_id: 1 },
            payloadSchema: '<ul><li><code>model_paths</code> (array of strings, optional): specific model paths to check. Defaults to every model under the configured model directories.</li><li><code>gpu_id</code> (int, optional, default 0): GPU to pin the check to.</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "started".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/smoke-test/start -H "Content-Type: application/json" -d '{"gpu_id": 1}'`
        },
        {
            category: 'text',
            method: 'GET',
            path: '/api/smoke-test/status',
            desc: 'Get progress and results of the active or most recent smoke test.',
            responseSchema: '<ul><li><code>running</code> (bool).</li><li><code>progress</code> (object): {current, total, model, status}.</li><li><code>results</code> (array of {model, status, answer|error}).</li></ul>',
            curl: `curl ${BASE_URL}/api/smoke-test/status`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/start',
            desc: 'Manually start a llama-server instance on a specific port with explicit flags — the low-level counterpart to the gateway\'s auto-loading (<code>get_or_load_model</code>). Useful for pinning a specific GPU, context size, or MTP/vision configuration rather than letting the gateway pick defaults on first use.',
            payload: {
                model_path: "~/gguf_models/gemma-4-12b-it-Q5_K_M.gguf",
                ctx_size: 32000,
                n_gpu_layers: -1,
                gpu_id: 0
            },
            payloadSchema: '<ul><li><code>model_path</code> (string, required): absolute path to the .gguf model file.</li><li><code>port</code> (int, optional): port to run this instance on. If omitted, a free port is picked and reserved automatically (safe to call concurrently for different instances). If set to a port already occupied by a managed instance, that instance is terminated first and replaced — only pass this when you deliberately want to replace/restart the instance pinned to a specific port.</li><li><code>ctx_size</code> (int, default 32000): context size (<code>-c</code>).</li><li><code>threads</code> (int, default 8): CPU thread count (<code>-t</code>).</li><li><code>n_gpu_layers</code> (int, default -1): layers to offload to GPU (<code>-ngl</code>); -1 offloads all.</li><li><code>gpu_id</code> (int, optional): GPU index to pin via <code>CUDA_VISIBLE_DEVICES</code>. If omitted, the GPU with the most free VRAM is picked automatically and still pinned (never left unset, since that lets llama.cpp tensor-split across every visible GPU).</li><li><code>mmproj_path</code> (string, optional): absolute path to a vision projector .gguf to enable multimodal input.</li><li><code>embedding</code> (bool, default false): enable embedding pooling mode (<code>--embedding</code>). Only set for dedicated embedding models. Auto-disabled if MTP speculative decoding ends up active, since the two are incompatible.</li><li><code>pooling</code> (string, optional: none/mean/cls/last/rank): explicit <code>--pooling</code> override. Leave unset to trust the model\'s GGUF metadata — set explicitly for embedding models whose metadata is unreliable, e.g. bge-m3 needs <code>cls</code> (some quantized bge-m3 GGUFs omit pooling metadata entirely, which silently resolves to "none" and breaks <code>/v1/embeddings</code> outright).</li><li><code>reranking</code> (bool, default false): enable the reranking endpoint (<code>--reranking</code>) for a dedicated cross-encoder reranker model (e.g. bge-reranker-v2-m3). Implies <code>embedding=true</code>. Also auto-disabled alongside embedding if MTP ends up active.</li><li><code>jinja</code> (bool, default true): enable Jinja chat template rendering (<code>--jinja</code>).</li><li><code>draft_model_path</code> (string, optional): absolute path to a draft model for MTP speculative decoding. If omitted and <code>mtp</code> is not false, a sidecar draft is auto-detected next to <code>model_path</code>.</li><li><code>mtp</code> (bool, optional): set false to disable MTP auto-detection entirely (both sidecar draft lookup and bundled-nextn-tensor detection).</li><li><code>parallel</code> (int, optional): parallel slots (<code>-np</code>). Default 1 (<code>LLAMA_DEFAULT_PARALLEL</code>); with 1 slot concurrent requests queue inside llama-server. Set &gt;1 if you deliberately send concurrent requests to this instance (e.g. a benchmark harness); 0 = llama.cpp\'s auto choice (4).</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "started".</li><li><code>pid</code> (int): spawned process PID.</li><li><code>port</code> (int): the port actually used — the one you passed, or the auto-picked one if omitted.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/start -H "Content-Type: application/json" -d '{"model_path": "~/gguf_models/gemma-4-12b-it-Q5_K_M.gguf", "gpu_id": 0}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/stop',
            desc: 'Stop a managed llama-server instance (or the ComfyUI process, if its port is passed) by port.',
            payload: { port: 8001 },
            payloadSchema: '<ul><li><code>port</code> (int, required, query parameter): port of the instance to stop.</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "stopped".</li><li><code>port</code> (int).</li></ul>',
            curl: `curl -X POST "${BASE_URL}/api/stop?port=8001"`
        },
        {
            category: 'text',
            method: 'GET',
            path: '/api/models/active',
            desc: 'List models currently loaded and their keep-alive/idle-timeout state, including ComfyUI if it\'s running. Distinct from <code>/api/status</code>: this is the auto-unload-relevant view (last used, keep-alive remaining), not the raw process/health view.',
            responseSchema: '<ul><li><code>active_models</code> (array): each item has <code>port</code>, <code>model</code>, <code>gpu_id</code>, <code>active_requests</code> (in-flight request count), <code>last_used</code> (unix timestamp), <code>keep_alive</code> (seconds), <code>keep_alive_remaining</code> (seconds until auto-unload), <code>ctx_size</code>, <code>draft_model</code>, <code>mtp_active</code>. ComfyUI, if running, is included as a synthetic entry with <code>model</code> ending in " [ComfyUI]".</li></ul>',
            liveResponse: lastApiData.active,
            curl: `curl ${BASE_URL}/api/models/active`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/tokenize',
            desc: 'Tokenize text against a currently active model server and return the token count (does not load a model — the target must already be running).',
            payload: { content: "The quick brown fox jumps over the lazy dog." },
            payloadSchema: '<ul><li><code>content</code> (string, required): text to tokenize.</li><li><code>model_name</code> (string, optional): which active instance to use. Defaults to whichever model is currently active if omitted.</li></ul>',
            responseSchema: '<ul><li><code>success</code> (bool).</li><li><code>tokens_count</code> (int).</li><li><code>error</code> (string, present on failure): e.g. "No active model server found" if the target model isn\'t currently running.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/tokenize -H "Content-Type: application/json" -d '{"content": "The quick brown fox jumps over the lazy dog."}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/v1/embeddings',
            desc: 'OPENAI COMPATIBLE: Generate embedding vectors, auto-loading the model (in embedding-pooling mode) if not already running. Identical handler and behavior to <code>/api/embeddings</code> below — both paths are wired to the same function.',
            payload: {
                model: "nomic-embed-text-v1.5.Q8_0.gguf",
                input: "The quick brown fox jumps over the lazy dog."
            },
            payloadSchema: '<ul><li><code>model</code> (string, required): GGUF model name or absolute path.</li><li><code>input</code> (string or array of strings/token-id arrays, required): text(s) to embed.</li><li><code>encoding_format</code> (string, optional, default "float").</li><li><code>dimensions</code> (int, optional): request a specific output dimensionality, if the model supports Matryoshka truncation.</li><li><code>user</code> (string, optional).</li><li><code>gpu_id</code> (int, optional): GPU to auto-load the model onto, if not already running. No load balancing if the same model is already running on more than one GPU — leaving this unset always matches whichever instance was registered first, regardless of its current load; set explicitly to target a specific GPU\'s instance.</li><li><code>keep_alive</code> (int, optional, default 300): idle seconds before auto-unload.</li><li><code>ctx_size</code> (int, optional, default 32000): context size (<code>-c</code>) to launch the model with, if it needs auto-loading. If a matching model is already running with a smaller context, it\'s restarted with this larger one; equal-or-larger and it\'s reused as-is.</li><li><code>parallel</code> (int, optional): llama-server parallel slots (<code>-np</code>) used when this request auto-loads the model. Default 1 (<code>LLAMA_DEFAULT_PARALLEL</code>) — more slots cost single-stream speed (up to ~2x for a model that nearly fills the card). Set &gt;1 only if you send concurrent requests to this model; 0 = llama.cpp\'s auto choice (4). Asking for more slots than a running instance has restarts it, but only when idle.</li></ul>',
            responseSchema: '<ul><li><code>object</code>: "list".</li><li><code>data</code> (array of {object, index, embedding}): one embedding vector per input.</li><li><code>model</code> (string).</li><li><code>usage</code> (object): {prompt_tokens, total_tokens}.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/embeddings -H "Content-Type: application/json" -d '{"model": "nomic-embed-text-v1.5.Q8_0.gguf", "input": "The quick brown fox jumps over the lazy dog."}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/embeddings',
            desc: 'UNIFIED GATEWAY alias for <code>/v1/embeddings</code> above — same request/response shape, same handler, just under the app\'s own <code>/api/...</code> namespace.',
            payloadSchema: '<ul><li>Identical to <code>/v1/embeddings</code>.</li></ul>',
            responseSchema: '<ul><li>Identical to <code>/v1/embeddings</code>.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/embeddings -H "Content-Type: application/json" -d '{"model": "nomic-embed-text-v1.5.Q8_0.gguf", "input": "The quick brown fox jumps over the lazy dog."}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/v1/rerank',
            desc: 'Score/rank a list of documents against a query using a dedicated cross-encoder reranker model (e.g. bge-reranker-v2-m3) — <strong>not</strong> the same as an embedding model like bge-m3, which only produces vectors, not relevance scores. Auto-loads the model with <code>--reranking</code> (which implies <code>--embedding</code> and pooling type "rank" on llama-server\'s own side) if not already running. Identical handler and behavior to <code>/api/rerank</code> below.',
            payload: {
                model: "bge-reranker-v2-m3-Q8_0.gguf",
                query: "What is the capital of Bulgaria?",
                documents: [
                    "Sofia is the capital and largest city of Bulgaria.",
                    "Paris is the capital of France.",
                    "The Danube river flows through several European countries."
                ]
            },
            payloadSchema: '<ul><li><code>model</code> (string, required): GGUF model name or absolute path.</li><li><code>query</code> (string, required).</li><li><code>documents</code> (array of strings, required): candidates to score against the query.</li><li><code>top_n</code> (int, optional, default: all documents): only return the top-N scored results.</li><li><code>gpu_id</code> (int, optional): GPU to auto-load the model onto, if not already running. No load balancing if the same model is already running on more than one GPU — leaving this unset always matches whichever instance was registered first, regardless of its current load; set explicitly to target a specific GPU\'s instance.</li><li><code>keep_alive</code> (int, optional, default 300): idle seconds before auto-unload.</li><li><code>ctx_size</code> (int, optional, default 32000): context size (<code>-c</code>) to launch the model with, if it needs auto-loading. If a matching model is already running with a smaller context, it\'s restarted with this larger one; equal-or-larger and it\'s reused as-is.</li><li><code>pooling</code> (string, optional): explicit <code>--pooling</code> override. Rarely needed — <code>--reranking</code> already forces pooling type "rank" server-side regardless of GGUF metadata.</li><li><code>parallel</code> (int, optional): llama-server parallel slots (<code>-np</code>) used when this request auto-loads the model. Default 1 (<code>LLAMA_DEFAULT_PARALLEL</code>) — more slots cost single-stream speed (up to ~2x for a model that nearly fills the card). Set &gt;1 only if you send concurrent requests to this model; 0 = llama.cpp\'s auto choice (4). Asking for more slots than a running instance has restarts it, but only when idle.</li></ul>',
            responseSchema: '<ul><li><code>object</code>: "list".</li><li><code>model</code> (string).</li><li><code>results</code> (array of {index, relevance_score, document}, sorted by <code>relevance_score</code> descending): <code>index</code> refers to the position in the original <code>documents</code> array; <code>document</code> is filled in from the request for convenience (llama-server itself only returns it in TEI-compat mode).</li><li><code>usage</code> (object): {prompt_tokens, total_tokens}.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/rerank -H "Content-Type: application/json" -d '{"model": "bge-reranker-v2-m3-Q8_0.gguf", "query": "What is the capital of Bulgaria?", "documents": ["Sofia is the capital and largest city of Bulgaria.", "Paris is the capital of France."]}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/rerank',
            desc: 'UNIFIED GATEWAY alias for <code>/v1/rerank</code> above — same request/response shape, same handler, just under the app\'s own <code>/api/...</code> namespace.',
            payloadSchema: '<ul><li>Identical to <code>/v1/rerank</code>.</li></ul>',
            responseSchema: '<ul><li>Identical to <code>/v1/rerank</code>.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/rerank -H "Content-Type: application/json" -d '{"model": "bge-reranker-v2-m3-Q8_0.gguf", "query": "What is the capital of Bulgaria?", "documents": ["Sofia is the capital and largest city of Bulgaria.", "Paris is the capital of France."]}'`
        },
        {
            category: 'text',
            method: 'POST',
            path: '/api/chat/compare',
            desc: 'Broadcast one prompt (with optional image/audio) to every currently-loaded, ready llama-server instance in parallel, and return each model\'s response side-by-side with timing/token/tps stats. For comparing models that are already running — it does not load anything new, so unloaded models are simply skipped.',
            payload: {
                prompt: "What is the capital of France?",
                system_prompt: "You are a helpful assistant.",
                max_tokens: 512,
                temperature: 0.7
            },
            payloadSchema: '<ul><li><code>prompt</code> (string, required).</li><li><code>system_prompt</code> (string, optional, default "You are a helpful assistant.").</li><li><code>image_base64</code> (string, optional): base64 image for multimodal instances.</li><li><code>audio_base64</code> (string, optional): base64 audio for audio-input instances.</li><li><code>max_tokens</code> (int, optional, default 2048).</li><li><code>temperature</code> (float, optional, default 0.7).</li></ul>',
            responseSchema: '<ul><li><code>responses</code> (array): one entry per currently-running instance, each with <code>port</code>, <code>model</code>, <code>content</code>, <code>time_taken</code> (seconds), <code>tokens</code>, <code>tps</code>, and <code>status</code>.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/chat/compare -H "Content-Type: application/json" -d '{"prompt": "What is the capital of France?"}'`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/v1/images/generations',
            desc: 'OPENAI COMPATIBLE: Synchronous image generation via ComfyUI. Returns generated image in Base64.',
            payload: {
                prompt: "a majestic golden retriever sitting on a porch, highly detailed",
                n: 1,
                size: "1024x1024"
            },
            payloadSchema: '<ul><li><code>prompt</code> (string, required): Description of the image to generate.</li><li><code>n</code> (int, optional): Number of images to generate (default 1).</li><li><code>size</code> (string, optional): Resolution in "WIDTHxHEIGHT" format (e.g. "1024x1024").</li></ul>',
            responseSchema: '<ul><li>Standard OpenAI image generation response including <code>data[].b64_json</code> containing the Base64 image payload.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/images/generations \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "a majestic golden retriever sitting on a porch, highly detailed",
    "size": "1024x1024"
  }'`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/v1/images/edits',
            desc: 'OPENAI COMPATIBLE: Synchronous image editing via ComfyUI (Qwen-Image-Edit 2511 + its 4-step Lightning LoRA). Expects multipart/form-data with a single reference image. Returns the edited image in Base64. Default: qwen-edit-2511 (Q2_K); "qwen-edit-2511-q3" picks the Q3_K_S quant. Optional "seed" for reproducible results.',
            payloadSchema: '<ul><li><code>image</code> (binary file, required): The single source image to edit.</li><li><code>prompt</code> (string, required): Edit instruction (e.g. "make the sky sunset orange").</li><li><code>model</code> (string, optional): catalog id from GET /v1/images/models?task=edit (default "qwen-edit-2511"; also "qwen-edit-2511-q3"). Older names like "Qwen_Image_Edit-Q2_K.gguf" now mean qwen-edit-2511.</li><li><code>seed</code> (int, optional): same seed + image + prompt = same result.</li></ul>',
            responseSchema: '<ul><li>Standard OpenAI image generation response including <code>data[].b64_json</code> containing the Base64 edited image.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/images/edits \\
  -F "image=@input.png" \\
  -F "prompt=make the sky sunset orange" \\
  -F "model=qwen-edit-2511"`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/models',
            desc: 'COMFYUI MODELS: Lists the diffusion model checkpoints actually valid as model_name on /api/comfyui/generate — queried live from ComfyUI\'s own UnetLoaderGGUF/UNETLoader node schemas (not a hardcoded or independently-scanned list), so it can\'t drift out of sync with what ComfyUI would really accept. Starts ComfyUI automatically if it isn\'t already running (first call after a cold start takes ~10s).',
            responseSchema: '<ul><li><code>models</code> (array of strings): Checkpoint filenames, sorted. Includes both GGUF (Ernie Turbo, Z-Image-Turbo) and safetensors (Ideogram-4) checkpoints — not every filename returned here has a matching workflow template, though (e.g. Chroma1-HD/flux1-schnell/qwen-image-* are present as files but have no generate workflow wired up).</li></ul>',
            liveResponse: lastApiData.comfyModels,
            curl: `curl ${BASE_URL}/api/comfyui/models`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/workflows',
            desc: 'COMFYUI WORKFLOWS: Lists every workflow template (Ernie Turbo, Z-Image-Turbo, Ideogram-4, Qwen-Image-Edit, Qwen-Image-Style-Transfer) with the endpoint it\'s driven through, its full request body schema (field types/required/defaults, derived directly from the Pydantic request models so it can\'t drift out of sync with the code), the model_name routing rule where relevant, which checkpoints on disk are known to work with it (with an "available" flag checked live against ComfyUI), and free-text notes on quirks found during testing (quant reliability differences, LoRA compatibility, safety-filter behavior). Same-architecture quant/parameter variants are intentionally reported as options on one workflow entry rather than as separate workflows — use this instead of guessing at supported models or hardcoding a model list client-side. Starts ComfyUI automatically if not already running (first call after a cold start takes ~10s).',
            responseSchema: '<ul><li><code>workflows</code> (array of objects): Each has <code>name</code>, <code>workflow_file</code>, <code>endpoint</code>, <code>request_model</code> (field name → {type, required, default}), <code>routing_rule</code> (string or null), <code>known_models</code> (array of {name, available}), and an optional <code>notes</code> string.</li></ul>',
            liveResponse: lastApiData.comfyWorkflows,
            curl: `curl ${BASE_URL}/api/comfyui/workflows`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/generate',
            desc: 'COMFYUI GENERATE: Queues an image generation task. Returns a task_id. model_name selects the workflow template: "ernie" routes to Ernie Turbo, "ideogram" routes to Ideogram-4 (dual-model guidance, slower — needs both a conditional and unconditional checkpoint loaded), "qwen-image-2.1" routes to Qwen-Image-2.1 (needs cfg=1 — verified live guidance-distilled; aspect_ratio is ignored, square output only; symmetric subjects can come out mirrored/doubled, see CLAUDE.md), anything else routes to Z-Image-Turbo — the model_name must match a checkpoint that pairs with whichever CLIP/VAE that template bakes in (see /api/models for the installed set).',
            payload: {
                prompt: "majestic mountains",
                negative_prompt: "",
                model_name: "z_image_turbo-Q6_K.gguf",
                aspect_ratio: "1:1",
                base_resolution: 1024,
                steps: 4,
                seed: 1337,
                lora_name: "z-image-turbo/z-image-pen.safetensors",
                lora_strength: 0.8,
                filename: "majestic-mountains-1"
            },
            payloadSchema: '<ul><li><code>prompt</code> (string, required): Description of image to build.</li><li><code>negative_prompt</code> (string, optional): Styles/elements to avoid. Ignored by Ideogram-4, which derives its unconditional pass from the positive prompt instead.</li><li><code>model_name</code> (string, required): ComfyUI checkpoint filename — must match the CLIP/VAE pairing of whichever workflow template it routes to (Ernie Turbo, Z-Image-Turbo, or Ideogram-4). For Ideogram-4 use "ideogram4_nvfp4_mixed.safetensors" (the paired unconditional checkpoint is fixed, not selectable).</li><li><code>aspect_ratio</code> (string, optional): Aspect ratio (e.g. "1:1", "16:9").</li><li><code>base_resolution</code> (int, optional): Native generator size.</li><li><code>steps</code> (int, optional): Denoising steps override.</li><li><code>seed</code> (int, optional): Fixed generation seed.</li><li><code>lora_name</code> (string, optional): LoRA path from <code>/api/comfyui/loras</code> — include the subfolder if listed (e.g. "z-image-turbo/z-image-pen.safetensors"), that\'s the exact string ComfyUI itself expects. Only applies to Ernie Turbo / Z-Image-Turbo (both carry a LoraLoaderModelOnly node that\'s bypassed unless this is set); no-op for Ideogram-4, which has no LoRA node. A LoRA is architecture-specific — pairing one trained for the wrong base checkpoint will fail to load or produce garbage.</li><li><code>lora_strength</code> (float, optional): LoRA strength (default 1.0 when lora_name is set).</li><li><code>loras</code> (array of {name, strength}, optional): Stack more than one LoRA. When set, takes precedence over lora_name/lora_strength (which remain as shorthand for a single LoRA). The first entry activates the same template LoRA node lora_name would; any further entries are chained after it — none replace it, they stack. Only applies to Ernie Turbo / Z-Image-Turbo, same as lora_name; no-op for Ideogram-4.</li><li><code>filename</code> (string, optional): When set, the completed image is also written to disk under <code>generated_images/&lt;filename&gt;.png</code> (sanitized to a plain basename; any extension you give is replaced with .png since that\'s the actual format produced) — fetch it later via <code>GET /api/comfyui/images/{filename}</code>. Omit to keep the old behavior (base64 only, nothing written to disk). A repeated filename overwrites the previous file of that name.</li><li><code>gpu_id</code> (int, optional): Which GPU ComfyUI is allowed to see, via its own native <code>--cuda-device</code> flag — <strong>only takes effect if ComfyUI isn\'t already running</strong>, since it\'s one persistent process shared by every caller (unlike llama-server, which spins up a fresh instance per model/gpu_id combo). If ComfyUI is already up on a different GPU, this is ignored (logged as a warning server-side, not an error) rather than restarting it out from under whatever else might be using it. To force a different GPU: <code>POST /api/comfyui/stop</code>, then either <code>POST /api/comfyui/start</code> with the desired <code>gpu_id</code>, or just let the next generate/edit/style-transfer call with <code>gpu_id</code> set auto-start it correctly. Check <code>GET /api/comfyui/status</code>\'s <code>gpu_id</code> field to see what it\'s actually pinned to right now. Omitted/null: every GPU stays visible to the process — on this Linux host that means PyTorch\'s own default of cuda:0 among whatever\'s visible, not a most-free-VRAM auto-pick like llama-server\'s own gpu_id-omitted path.</li></ul>',
            responseSchema: '<ul><li><code>task_id</code> (string): UUID string of the queued ComfyUI run.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/generate -H "Content-Type: application/json" -d '{"prompt": "majestic mountains", "model_name": "z_image_turbo-Q6_K.gguf", "gpu_id": 0, "loras": [{"name": "z-image-turbo/z-image-pen.safetensors", "strength": 0.8}, {"name": "z-image-turbo/C64style_Z-Image_Turbo.safetensors", "strength": 0.5}], "filename": "majestic-mountains-1"}'`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/start',
            desc: 'COMFYUI START: Explicitly start the (single, shared) ComfyUI process, optionally pinned to a specific GPU. Normally unnecessary — every generate/edit/style-transfer/models/workflows call auto-starts ComfyUI on first use via this same endpoint — but this is the direct way to control which GPU it comes up on, or to restart it on a different one (stop first).',
            payload: { port: 8189, gpu_id: 1 },
            payloadSchema: '<ul><li><code>port</code> (int, default 8189): port to run ComfyUI on.</li><li><code>gpu_id</code> (int, optional): GPU index to pin via ComfyUI\'s native <code>--cuda-device</code> flag. No-op if ComfyUI is already running (returns "already running" with its actual current <code>gpu_id</code> — stop it first to change GPUs). Omitted: every GPU stays visible, PyTorch defaults to cuda:0 among them — there\'s no most-free-VRAM auto-pick here the way there is for llama-server\'s <code>/api/start</code>.</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "started" or "already running".</li><li><code>pid</code> (int).</li><li><code>port</code> (int).</li><li><code>gpu_id</code> (int or null): the GPU it\'s actually pinned to.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/start -H "Content-Type: application/json" -d '{"gpu_id": 1}'`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/status',
            desc: 'COMFYUI STATUS: Check whether ComfyUI is running/ready and which GPU it\'s currently pinned to.',
            responseSchema: '<ul><li><code>running</code> (bool).</li><li><code>ready</code> (bool): health-checked, not just process-alive.</li><li><code>port</code> (int).</li><li><code>pid</code> (int or null).</li><li><code>gpu_id</code> (int or null): the GPU it was started with (via <code>--cuda-device</code>), or null if unpinned or not running.</li></ul>',
            curl: `curl ${BASE_URL}/api/comfyui/status`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/unload',
            desc: 'COMFYUI UNLOAD: Free ComfyUI\'s VRAM (unload all models) but keep the process running, so the next generate skips the ~13s process start and only pays the model load (~11s for Z-Image-Turbo). The same thing happens automatically after 180s without any image job. Refuses (409) while a job is running unless force=true.',
            payloadSchema: '<ul><li><code>force</code> (bool query param, default false): unload even while busy. ComfyUI still finishes a running job first.</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "unloaded", or "not running" if ComfyUI isn\'t reachable.</li></ul><p>409 if busy: an image task tracked by this app is pending/processing, or ComfyUI\'s own queue is non-empty (also catches jobs queued from its native UI).</p>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/unload`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/stop',
            desc: 'COMFYUI STOP: Terminate the ComfyUI process entirely (frees all its VRAM and RAM). Refuses (409) while a job is running unless force=true — a forced stop drops the in-flight job. /api/stop?port=8189 and /api/models/unload with ComfyUI\'s port route here with the same guard. Prefer /api/comfyui/unload if you only need the VRAM back.',
            payloadSchema: '<ul><li><code>force</code> (bool query param, default false): stop even while busy.</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "stopped", or "stopped (no active managed process)" when it was found by cmdline rather than tracked.</li></ul><p>409 if busy (same check as unload).</p>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/stop`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/batch',
            desc: 'COMFYUI BATCH: Runs a whole JSON list of prompts through /api/comfyui/generate sequentially, one image at a time (same node-patching/workflow logic, same routing rule for model_name). Only one batch can be active at a time — a second call while one is running/paused gets HTTP 409. Rejects (400) if items is empty or over 500 entries. Also drives the Image Gen tab\'s Batch sub-tab: the dashboard just reads an uploaded .json file client-side and POSTs its parsed contents here — there is no separate multipart upload endpoint.',
            payload: {
                defaults: {
                    model_name: "z_image_turbo-Q6_K.gguf",
                    negative_prompt: "ugly, deformed, bad anatomy, blurry",
                    aspect_ratio: "1:1",
                    base_resolution: 1024
                },
                items: [
                    { prompt: "a cat playing with a ball of yarn, highly detailed", filename: "cat-yarn" },
                    { prompt: "a dog running on a beach at sunset", aspect_ratio: "16:9", filename: "dog-beach" },
                    { prompt: "a futuristic city skyline at night", seed: 42, steps: 6 }
                ]
            },
            payloadSchema: '<ul><li><code>defaults</code> (object, optional): Same fields as /api/comfyui/generate\'s payload minus prompt and filename (<code>model_name</code>, <code>negative_prompt</code>, <code>aspect_ratio</code>, <code>base_resolution</code>, <code>steps</code>, <code>seed</code>, <code>lora_name</code>, <code>lora_strength</code>, <code>gpu_id</code>) — applied to every item unless that item overrides the field itself. All have sane built-in fallbacks (model_name defaults to Z-Image-Turbo\'s checkpoint), so <code>defaults</code> can be omitted entirely. <code>gpu_id</code> in particular is batch-wide only (there\'s no per-item override) — it governs ComfyUI\'s auto-start for the whole batch, same rules as /api/comfyui/generate\'s <code>gpu_id</code> (no-op if ComfyUI is already running on a different GPU).</li><li><code>items</code> (array, required, 1-500 entries): Each item requires only <code>prompt</code>; every other /api/comfyui/generate field except <code>gpu_id</code> is accepted per-item and overrides <code>defaults</code> when present. <code>filename</code> (string, optional, per-item only — not a <code>defaults</code> field, since sharing one name across items would just mean each overwrites the last): saves that item\'s image to disk under <code>generated_images/&lt;filename&gt;.png</code> once it completes; omit to skip saving that item.</li></ul>',
            responseSchema: '<ul><li><code>batch_id</code> (string).</li><li><code>total</code> (int): number of items queued.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/batch -H "Content-Type: application/json" -d '{"items": [{"prompt": "a cat", "filename": "cat-yarn"}, {"prompt": "a dog", "aspect_ratio": "16:9"}]}'`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/batch/status',
            desc: 'COMFYUI BATCH STATUS: Current (or most recently finished) batch\'s progress — polled every ~1.5s by the Image Gen tab\'s Batch sub-tab while a batch is running/paused. image_base64 is deliberately not inlined per item here (would bloat this response fast on a large batch) — fetch a completed item\'s image via the existing GET /api/comfyui/task/{task_id} using that item\'s task_id.',
            responseSchema: '<ul><li><code>status</code> (string): "none" (no batch has run yet), "running", "paused", "cancelled", "completed", or "failed".</li><li><code>id</code>, <code>total</code>, <code>completed_count</code>, <code>current_index</code>.</li><li><code>items</code> (array): each has <code>index</code>, <code>prompt</code>, <code>model_name</code>, <code>status</code> ("pending"/"processing"/"completed"/"failed"/"cancelled"), <code>task_id</code> (null until queued; use it against GET /api/comfyui/task/{task_id} for the image), <code>error</code>, <code>runtime_seconds</code>, <code>filename</code> (null if that item didn\'t request one; the requested name until the item finishes, then the confirmed saved name — or null if the save itself failed — once <code>status</code> is "completed"; fetch it via GET /api/comfyui/images/{filename}).</li></ul>',
            liveResponse: lastApiData.comfyBatch,
            curl: `curl ${BASE_URL}/api/comfyui/batch/status`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/batch/pause',
            desc: 'COMFYUI BATCH PAUSE: Stops the batch from starting its next item once the current one finishes — does not interrupt the in-flight generation (a Z-Image-Turbo run is seconds long, so letting it finish is simpler than reconciling a half-finished ComfyUI job). HTTP 409 if no batch is currently running.',
            responseSchema: '<ul><li><code>status</code> (string): "paused".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/batch/pause`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/batch/resume',
            desc: 'COMFYUI BATCH RESUME: Continues a paused batch from exactly where it left off. HTTP 409 if no batch is currently paused.',
            responseSchema: '<ul><li><code>status</code> (string): "running".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/batch/resume`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/batch/cancel',
            desc: 'COMFYUI BATCH CANCEL: Aborts a running or paused batch immediately — unlike pause, this also calls ComfyUI\'s own /interrupt so the in-flight item is stopped right away rather than left to finish. Every not-yet-started item is marked "cancelled". HTTP 409 if no batch is active. Starting a new batch afterward is immediately allowed.',
            responseSchema: '<ul><li><code>status</code> (string): "cancelled".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/batch/cancel`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/loras',
            desc: 'COMFYUI LORAS: Recursively lists available LoRA .safetensors files (including per-model subfolders, e.g. "z-image-turbo/foo.safetensors") from the same directory ComfyUI itself loads LoRAs from. Each one can optionally be documented via a same-name .json sidecar (e.g. "z-image-pen.safetensors" + "z-image-pen.json") — the same convention voices/*.wav + voices/*.txt already uses, just JSON here since a LoRA has several distinct optional fields instead of one blob of text.',
            responseSchema: '<ul><li><code>loras</code> (array of objects, sorted by name): <code>name</code> (string, required) — pass this back as <code>lora_name</code>/inside <code>loras[].name</code> on /api/comfyui/generate, /edit, /style-transfer. Optional, only present when set in the sidecar: <code>trigger_words</code> (string — keyword(s) to include in your prompt for this LoRA to actually apply as trained), <code>description</code> (string), <code>base_model</code> (string — which checkpoint/architecture this LoRA is compatible with, e.g. "z-image-turbo"; no compatibility check is enforced, this is purely informational), <code>recommended_strength</code> (float). A LoRA with no sidecar is just <code>{"name": "..."}</code>, unchanged from before this existed.</li></ul>',
            liveResponse: lastApiData.loras,
            curl: `curl ${BASE_URL}/api/comfyui/loras`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/images',
            desc: 'COMFYUI SAVED IMAGES: Lists every image saved so far via the filename option on /api/comfyui/generate, /edit, /style-transfer, or a batch item.',
            responseSchema: '<ul><li><code>images</code> (array of strings): saved .png filenames, sorted.</li></ul>',
            curl: `curl ${BASE_URL}/api/comfyui/images`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/images/{filename}',
            desc: 'COMFYUI FETCH SAVED IMAGE: Returns the raw PNG bytes for a previously saved image. filename is sanitized the same way saving is (basename-only, safe charset) — a path-traversal attempt just resolves to its own basename rather than erroring. 404 if no file by that (sanitized) name exists.',
            curl: `curl ${BASE_URL}/api/comfyui/images/example.png -o example.png`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/edit',
            desc: 'COMFYUI EDIT: Queues a single-image edit task via Qwen-Image-Edit + a 4-step Lightning LoRA (fast: ~40-55s vs ~330s undistilled). Returns a task_id — poll the same /api/comfyui/task/{task_id} endpoint used for generation. Three quants tested (same prompt/image): Q2_K (7.1GB, ~43s, fully applied edit), Q4_K_S (12.1GB, ~54s, fully applied edit), Q3_K_S (9.2GB, ~40s, but under-applied the edit — fastest is not the most reliable here). Default is Q2_K.',
            payload: {
                image_base64: "<base64-encoded-image>",
                prompt: "make the sky sunset orange",
                model_name: "qwen-edit-2511",
                filename: "sunset-edit-1"
            },
            payloadSchema: '<ul><li><code>image_base64</code> (string, required): Base64-encoded source image (single image only, for now).</li><li><code>prompt</code> (string, required): Edit instruction.</li><li><code>model_name</code> (string, optional): catalog id ("qwen-edit-2511" default, "qwen-edit-2511-q3") or its file name. Older pre-2511 names map to qwen-edit-2511.</li><li><code>steps</code> (int, optional): Denoising steps override (default 4, tuned for the Lightning LoRA baked into the workflow).</li><li><code>seed</code> (int, optional): Fixed generation seed.</li><li><code>loras</code> (array of {name, strength}, optional): Stack one or more extra LoRAs on top of the workflow\'s own load-bearing 4-step Lightning speed-LoRA — they chain after it, never replace it (removing that LoRA would break the 4-step sampling this workflow is tuned for). Unlike /api/comfyui/generate, there\'s no separate lora_name shorthand here since the template\'s LoRA slot is already taken by the required Lightning LoRA.</li><li><code>filename</code> (string, optional): Also save the result to disk — see /api/comfyui/generate\'s filename field for the full rules (sanitized basename, always .png, GET /api/comfyui/images/{filename} to fetch).</li><li><code>gpu_id</code> (int, optional): Same rules as /api/comfyui/generate\'s <code>gpu_id</code> — only takes effect on ComfyUI\'s auto-start, ignored (with a server-side warning) if it\'s already running on a different GPU.</li></ul>',
            responseSchema: '<ul><li><code>task_id</code> (string): UUID string of the queued ComfyUI run.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/edit -H "Content-Type: application/json" -d '{"image_base64": "...", "prompt": "make the sky sunset orange"}'`
        },
        {
            category: 'image',
            method: 'POST',
            path: '/api/comfyui/style-transfer',
            desc: 'COMFYUI STYLE TRANSFER: Applies a reference image’s visual style to a separate content image (two-image conditioning via TextEncodeQwenImageEditPlus, unlike the single-image /api/comfyui/edit). Requires the 2511 quant — the unversioned Q2_K/Q4_K_S quants predate multi-image support and are untested here. Returns a task_id — poll the same /api/comfyui/task/{task_id} endpoint.',
            payload: {
                content_image_base64: "<base64-encoded-content-image>",
                style_image_base64: "<base64-encoded-style-reference-image>",
                prompt: "Repaint the first image in the style of the second image, keeping its composition.",
                model_name: "qwen-image-edit-2511-Q3_K_S.gguf",
                filename: "style-transfer-1"
            },
            payloadSchema: '<ul><li><code>content_image_base64</code> (string, required): Base64-encoded image whose composition/content is kept.</li><li><code>style_image_base64</code> (string, required): Base64-encoded image whose visual style is copied.</li><li><code>prompt</code> (string, optional): Instruction guiding the style transfer. Has a sensible default if omitted.</li><li><code>model_name</code> (string, optional): Must be a multi-image-capable GGUF. Default and only tested option: "qwen-image-edit-2511-Q3_K_S.gguf".</li><li><code>steps</code> (int, optional): Denoising steps override (default 4).</li><li><code>seed</code> (int, optional): Fixed generation seed.</li><li><code>loras</code> (array of {name, strength}, optional): Stack one or more extra LoRAs on top of the workflow\'s own load-bearing 4-step Lightning speed-LoRA — they chain after it, never replace it. Same rules as /api/comfyui/edit\'s loras field.</li><li><code>filename</code> (string, optional): Also save the result to disk — see /api/comfyui/generate\'s filename field for the full rules.</li><li><code>gpu_id</code> (int, optional): Same rules as /api/comfyui/generate\'s <code>gpu_id</code> — only takes effect on ComfyUI\'s auto-start, ignored (with a server-side warning) if it\'s already running on a different GPU.</li></ul>',
            responseSchema: '<ul><li><code>task_id</code> (string): UUID string of the queued ComfyUI run.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/comfyui/style-transfer -H "Content-Type: application/json" -d '{"content_image_base64": "...", "style_image_base64": "..."}'`
        },
        {
            category: 'image',
            method: 'GET',
            path: '/api/comfyui/task/{task_id}',
            desc: 'COMFYUI STATUS: Poll active progress and retrieve base64 image upon completion. Shared by /api/comfyui/generate, /edit, /style-transfer, and each item inside a batch.',
            responseSchema: '<ul><li><code>task_id</code> (string): Unique job UUID.</li><li><code>status</code> (string): "queued", "processing", "completed", or "failed".</li><li><code>progress</code> (object): <code>{ step, max_steps, node, value }</code>.</li><li><code>model_used</code> (string): Image generator model.</li><li><code>runtime_seconds</code> (float): Total processing time.</li><li><code>image_base64</code> (string): Base64 data URI of the generated picture.</li><li><code>error</code> (string): Error detail if execution crashed.</li><li><code>filename</code> (string or null): Confirmed saved basename if a filename was requested and the save succeeded (fetch via GET /api/comfyui/images/{filename}); null if no filename was requested, the save failed, or generation isn\'t done yet.</li></ul>',
            liveResponse: lastApiData.comfyTask || {
                task_id: "example-uuid-1234",
                status: "processing / completed / failed",
                progress: { step: 3, max_steps: 4, value: 75 },
                model_used: "z_image_turbo-Q6_K.gguf",
                runtime_seconds: 14.5,
                image_base64: "data:image/png;base64,...",
                error: null,
                filename: null
            },
            curl: `curl ${BASE_URL}/api/comfyui/task/3279f6a0-5903-4ba2-8352-05bfea52b225`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/v1/audio/speech',
            desc: 'OPENAI COMPATIBLE: Text-to-speech. Engine follows the model field: "qwen"→Qwen3-TTS, "chatter"→Chatterbox, "omni"→OmniVoice, anything else (incl. "tts-1")→server default (CosyVoice 3). voice = a voices/ preset filename (matched prefix), an OpenAI built-in name (ignored), or a free-form voice description. For CosyVoice a preset voice is cloned; a description drives instruct on DEFAULT_TTS_VOICE. OmniVoice descriptions use a narrow tag vocabulary (accent/age/pitch/whisper); Qwen3 → VoiceDesign (free-form).',
            payload: {
                model: "tts-1",
                input: "Hallo, das ist ein Test.",
                voice: "synth-default"
            },
            payloadSchema: '<ul><li><code>model</code> (string, required): Engine selector — "tts-1" (or anything without "qwen") = OmniVoice; contains "qwen" = Qwen3-TTS VoiceDesign.</li><li><code>input</code> (string, required): Text to read.</li><li><code>voice</code> (string, required): Reference voice preset filename (e.g. "synth-default.wav"), or a voice description. OmniVoice: instruct-design tags (e.g. "p-de-f, fast"). Qwen3: free-form, e.g. "a slow, calm voice with a low pitch".</li></ul>',
            responseSchema: '<ul><li>Returns binary audio file stream (WAV/MP3 format).</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -o output.wav \\
  -d '{
    "model": "tts-1",
    "input": "Hallo, das ist ein Test.",
    "voice": "synth-default"
  }'`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/v1/audio/transcriptions',
            desc: 'OPENAI COMPATIBLE: Audio transcription via WhisperX. Expects multipart/form-data with a binary file. Video files work too, not just audio — decoding goes through ffmpeg regardless of container, so it just extracts whatever audio track is present (e.g. an .mp4). Long recordings are handled by WhisperX\'s own VAD-based batching, not naive fixed-window chunking.',
            payloadSchema: '<ul><li><code>file</code> (binary file, required): The audio (or video) file to transcribe.</li><li><code>model</code> (string, required): Model name (e.g. "whisper-1").</li></ul>',
            responseSchema: '<ul><li><code>text</code> (string): Transcribed text string.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/v1/audio/transcriptions \\
  -F "file=@audio.wav" \\
  -F "model=whisper-1"`
        },
        {
            category: 'audio',
            method: 'GET',
            path: '/api/voices',
            desc: 'TTS VOICES: List all local voice presets, their absolute paths, and their stored transcriptions.',
            responseSchema: '<ul><li><code>voices</code> (array): List of speaker reference voice objects. Each has <code>name</code> (filename), <code>path</code> (absolute file path), and <code>transcript</code> (transcribed voice text if present).</li></ul>',
            liveResponse: lastApiData.voices,
            curl: `curl ${BASE_URL}/api/voices`
        },
        {
            category: 'audio',
            method: 'GET',
            path: '/api/tts/status',
            desc: 'TTS STATUS (engine-agnostic): One call for another process to check whether TTS is up, still loading, or busy — without caring which engine (OmniVoice, the default behind /v1/audio/speech, or the Qwen3-TTS worker) serves the request. Same payload is also the "tts" key of GET /api/system/status. For raw log lines use GET /api/logs/tts.',
            responseSchema: '<ul><li><code>status</code> (string): worst engine state — "error" | "starting" | "loading" | "generating" | "idle" | "unloaded".</li><li><code>ready</code> (bool): not starting/loading/error.</li><li><code>busy</code> (bool): starting/loading/generating.</li><li><code>default_engine</code> (string): "omnivoice".</li><li><code>engines</code> (object): per-engine <code>status</code>/<code>loaded</code>/<code>device</code> (plus <code>variant</code>/<code>pid</code> for qwen3tts).</li></ul>',
            curl: `curl ${BASE_URL}/api/tts/status`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/tts/generate',
            desc: 'TTS (engine-agnostic): Speech generation. Omit engine for the server default (DEFAULT_TTS_ENGINE, normally "cosyvoice"; every engine runs in tts-serve; 503 if it is unavailable). engine="cosyvoice" (CosyVoice 3 — clone + free-form style via variant="instruct"; Apache-2.0) | "omnivoice" (23 style tags OR clone; weights CC-BY-NC / non-commercial) | "qwen3" (3 checkpoints) | "chatterbox" (clone + exaggeration/cfg_weight; MIT, watermarked). Worker engines need a one-time scripts/setup_{cosyvoice,chatterbox}.sh. A default request with no reference clones DEFAULT_TTS_VOICE.',
            payload: {
                text: "Hallo, das ist ein Test der Stimme.",
                language: "de",
                ref_voice_path: "/opt/ai-serve/voices/synth-default.wav",
                ref_text: null,
                response_format: "json"
            },
            payloadSchema: '<ul><li><code>text</code> (string, required): Speech content to generate.</li><li><code>language</code> (string, optional): ISO language code (e.g. "de", "en", "fr"). For engine="qwen3" a code is mapped to the full language name it needs.</li><li><code>engine</code> (string, optional): omit for the server default ("cosyvoice"). "cosyvoice" | "omnivoice" | "qwen3" | "chatterbox".</li><li><code>variant</code> (string): engine="qwen3": "base"(default)/"voice_design"/"custom_voice". engine="cosyvoice": "zero_shot"(default, clone; needs ref_* + ref_text) / "instruct" (clone + free-form instruct — slower/angrier/…) / "cross_lingual" (clone; ref_* only). Ignored for omnivoice/chatterbox.</li><li><code>speaker</code> (string, engine="qwen3" + variant="custom_voice", required there): one of the 9 preset timbre names.</li><li><code>exaggeration</code> / <code>cfg_weight</code> / <code>temperature</code> (float, engine="chatterbox"): expressiveness knobs (~0..1). temperature is also forwarded to cosyvoice.</li><li><code>ref_voice_path</code> (string, optional): Voice-cloning reference — path to a preset WAV/MP3 under voices/. OmniVoice, or engine="qwen3" variant="base".</li><li><code>ref_audio_base64</code> (string, optional): Voice-cloning reference as base64 audio (alternative to ref_voice_path).</li><li><code>ref_text</code> (string): Transcript of the reference audio. Optional for OmniVoice (auto-transcribed if omitted); REQUIRED for engine="qwen3" variant="base" and engine="cosyvoice" variant="zero_shot" (a voices/ preset .txt sidecar or DEFAULT_TTS_VOICE fills it automatically).</li><li><code>instruct</code> (string, optional): natural-language voice description / style instruction. OmniVoice, or engine="qwen3" variant="voice_design" (required there) / "custom_voice" (optional).</li><li><code>response_format</code> (string, optional): "json" (returns base64 audio and transcript text) or "binary" (returns audio stream).</li><li><code>keep_alive</code> (int, optional, default 60): idle seconds before the engine auto-unloads, same convention as a llama-server instance\'s keep_alive (whose own default is 300 — TTS defaults lower since a generate call is a one-shot, seconds-long request rather than a multi-turn session someone keeps returning to). Reset on every call while the model stays loaded. Applies to whichever engine ends up handling the request — including engine="qwen3", which internally forwards this same value to the deprecated dedicated Qwen3-TTS endpoint below.</li></ul>',
            responseSchema: '<ul><li>If json format: Returns <code>{"audio_base64": "...", "transcript": "..."}</code>. If audio format: returns binary stream.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/tts/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Hallo, das ist ein Test der Stimme.",
    "language": "de",
    "ref_voice_path": "/opt/ai-serve/voices/synth-default.wav",
    "ref_text": null
  }'`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/tts/unload',
            desc: 'OMNIVOICE UNLOAD: Explicitly unload the OmniVoice model to free VRAM/RAM immediately, rather than waiting for the keep_alive idle timeout. Also fires automatically on idle timeout (default 60s after the last generate call, or whatever keep_alive that call set) — this endpoint is for forcing it early. HTTP 409 if a generation is currently in flight; returns {"status": "already unloaded"} (not an error) if nothing was loaded to begin with. Also reachable from the Servers & GPUs tab\'s Active Model Servers list once OmniVoice is loaded (Stop & Unload button) — same for the other three TTS engines via their own unload endpoints (/api/tts/qwen3/unload, /api/tts/cosyvoice/unload, /api/tts/chatterbox/unload), all surfaced there the same way.',
            responseSchema: '<ul><li><code>status</code> (string): "unloaded" or "already unloaded".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/tts/unload`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/tts/qwen3/generate',
            desc: 'DEPRECATED — use POST /api/tts/generate with engine="qwen3". QWEN3-TTS: A second, independent TTS engine alongside OmniVoice — 3 upstream checkpoints selected via variant. "base" clones a voice from ~3s of reference audio + its transcript. "voice_design" builds a voice from a natural-language description (timbre/emotion/prosody). "custom_voice" picks one of 9 preset premium timbres (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee), with an optional style instruct layered on top. Each variant is a separate ~1.7B checkpoint — only one is held in VRAM at a time, so switching variant unloads and reloads (a few seconds).',
            payload: {
                text: "Hello, this is a voice design test.",
                variant: "voice_design",
                language: "English",
                instruct: "A calm, warm female voice, mid-to-low pitch, slow pace."
            },
            payloadSchema: '<ul><li><code>text</code> (string, required): Speech content to generate.</li><li><code>variant</code> (string, optional, default "base"): "base" | "voice_design" | "custom_voice".</li><li><code>language</code> (string, optional, default "English"): Full language name (English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian) or "Auto".</li><li><code>ref_voice_path</code> / <code>ref_audio_base64</code> + <code>ref_text</code> (variant="base"): reference audio + its transcript, same voices/ path convention as /api/tts/generate.</li><li><code>instruct</code> (variant="voice_design", required; variant="custom_voice", optional): natural-language voice description or style instruction.</li><li><code>speaker</code> (variant="custom_voice", required): one of the 9 preset timbre names above.</li><li><code>response_format</code> (string, optional): "json" or "binary".</li><li><code>keep_alive</code> (int, optional, default 60): idle seconds before auto-unload, same convention as /api/tts/generate.</li></ul>',
            responseSchema: '<ul><li>If json format: Returns <code>{"status": "success", "audio_base64": "..."}</code>. If binary format: returns a WAV audio stream.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/tts/qwen3/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Hello, this is a voice design test.",
    "variant": "voice_design",
    "language": "English",
    "instruct": "A calm, warm female voice, mid-to-low pitch, slow pace."
  }'`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/tts/qwen3/unload',
            desc: 'QWEN3-TTS UNLOAD: Explicitly unload whichever Qwen3-TTS variant is currently loaded, freeing VRAM/RAM immediately rather than waiting for the keep_alive idle timeout. HTTP 409 if a generation is currently in flight; returns {"status": "already unloaded"} if nothing was loaded. Also reachable from the Servers & GPUs tab\'s Qwen3-TTS card (Stop & Unload button).',
            responseSchema: '<ul><li><code>status</code> (string): "unloaded" or "already unloaded".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/tts/qwen3/unload`
        },
        {
            category: 'audio',
            method: 'GET',
            path: '/api/stt/diarize/backends',
            desc: 'DIARIZATION BACKENDS: Reports which diarize_backend values are actually usable right now, rather than a caller discovering a missing isolated venv via a 500 mid-transcription. "nemotron".available is false (with a setup hint) until scripts/setup_nemotron_diarize.sh has been run.',
            responseSchema: '<ul><li><code>backends</code> (object): <code>whisperx</code> / <code>nemotron</code>, each <code>{available, loaded, description}</code>.</li></ul>',
            curl: `curl ${BASE_URL}/api/stt/diarize/backends`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/stt/transcribe',
            desc: 'WHISPERX STT: Custom audio transcription, optionally with speaker diarization. Can transcribe local audio_path or uploaded audio_base64. Video files work too (e.g. .mp4/.mkv) — decoding goes through ffmpeg regardless of container, so it just extracts whatever audio track is present; verified live with a real muxed MP4. Long recordings are handled by WhisperX\'s own VAD-based batched transcription, not naive fixed-window chunking. diarize=true labels each turn with a speaker — diarize_backend picks pyannote.audio (default) or nvidia/Nemotron-3-Diarization (an isolated-venv worker; see /api/stt/diarize/nemotron/unload). Verified live end-to-end with real two-speaker audio: both backends correctly split at the true speaker boundary.',
            payload: {
                audio_path: "/opt/ai-serve/voices/synth-default.wav",
                diarize: false
            },
            payloadSchema: '<ul><li><code>audio_path</code> (string, optional): Path to a local audio or video file.</li><li><code>audio_base64</code> (string, optional): Base64 audio or video data to transcribe.</li><li><code>keep_alive</code> (int, optional, default 300): idle seconds before WhisperX (and the diarization pipeline, if loaded) auto-unloads. Same convention as /api/tts/generate\'s keep_alive.</li><li><code>diarize</code> (bool, optional, default false): label each turn with a speaker.</li><li><code>diarize_backend</code> (string, optional, default "whisperx"): "whisperx" (pyannote.audio, in-process) or "nemotron" (nvidia/Nemotron-3-Diarization, isolated-venv subprocess worker — needs scripts/setup_nemotron_diarize.sh run once). Word-level alignment always runs via WhisperX either way. The nemotron backend does NOT support return_embeddings/known_speakers (400 if combined), and only approximates num_speakers/min_speakers/max_speakers (rank-by-duration truncation, not a native clustering constraint).</li><li><code>num_speakers</code> / <code>min_speakers</code> / <code>max_speakers</code> (int, optional, diarize=true only): speaker-count hint; num_speakers wins if set.</li></ul>',
            responseSchema: '<ul><li><code>status</code> (string): "success".</li><li><code>transcript</code> (string): Plain text normally; "SPEAKER_00: ...\\nSPEAKER_01: ..." per turn when diarize=true.</li><li><code>segments</code> (array): <code>{start, end, text}</code> objects — speaker turns (not raw whisper segments) with a <code>speaker</code> field added when diarize=true.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/stt/transcribe \\
  -H "Content-Type: application/json" \\
  -d '{
    "audio_path": "/opt/ai-serve/voices/synth-default.wav"
  }'`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/stt/unload',
            desc: 'WHISPERX UNLOAD: Explicitly unload the WhisperX model AND the diarization/alignment pipeline (if loaded) to free VRAM/RAM immediately, rather than waiting for the keep_alive idle timeout. HTTP 409 if a transcription is currently in flight; returns {"status": "already unloaded"} if nothing was loaded.',
            responseSchema: '<ul><li><code>status</code> (string): "unloaded" or "already unloaded".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/stt/unload`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/stt/diarize/unload',
            desc: 'DIARIZATION UNLOAD: Unload just the speaker-diarization + word-alignment pipeline (diarize_backend="whisperx"), keeping WhisperX itself loaded. HTTP 409 if diarization is currently running; returns {"status": "already unloaded"} if nothing was loaded.',
            responseSchema: '<ul><li><code>status</code> (string): "unloaded" or "already unloaded".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/stt/diarize/unload`
        },
        {
            category: 'audio',
            method: 'POST',
            path: '/api/stt/diarize/nemotron/unload',
            desc: 'NEMOTRON DIARIZATION UNLOAD: Stop the isolated-venv Nemotron-3-Diarization worker subprocess (diarize_backend="nemotron"), freeing VRAM immediately rather than waiting for keep_alive. HTTP 409 if diarization is currently running; returns {"status": "already unloaded"} if not running. Also reachable from the Servers & GPUs tab\'s "Nemotron Diarization" card (Stop & Unload button).',
            responseSchema: '<ul><li><code>status</code> (string): "unloaded" or "already unloaded".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/stt/diarize/nemotron/unload`
        },
        {
            category: 'document',
            method: 'POST',
            path: '/api/ocr',
            desc: 'OCR / DOCUMENT EXTRACTION: Run one document (one or more page images) through NuExtract3, a vision-language model finetuned specifically for document understanding — structured JSON extraction against a caller-supplied template, image-to-Markdown transcription, document-type detection, or template generation. Auto-loads the model (+ its mmproj vision sidecar) if not already running. Synchronous — for many documents at once, use /api/ocr/batch instead so a large run doesn\'t block on one HTTP call. <strong>Every page is sent to the model as its own request, never multiple pages together</strong> — verified live that presenting several pages in one request measurably degrades accuracy (a real DOI fabricated into a fake ISBN, ~15 of ~60 names in a list garbled, two adjacent pages silently merged together), 100% reproducibly, and that isolating pages fixes all of it. More requests per document as a result, but correct ones.',
            payload: {
                images_base64: ["<base64-encoded-page-image>"],
                mode: "markdown"
            },
            payloadSchema: '<ul><li><code>images_base64</code> (array of strings, required): one or more page images for this document (multi-page = one call, one combined result). Any entry can be a PDF instead of an image — content-sniffed via magic bytes, not filename/extension — which is expanded server-side into one rendered page image per PDF page, in order (capped at 50 pages per document; split a bigger one across /api/ocr/batch items).</li><li><code>mode</code> (string, optional, default "markdown"): <code>structured</code> (extract against <code>template</code>), <code>content</code> (free-form transcription), <code>markdown</code> (image-to-Markdown conversion — this is the closest thing to plain OCR), <code>document-detection</code> (classify document type), or <code>template-generation</code> (have the model propose an extraction template).</li><li><code>pdf_dpi</code> (int, optional, default 170): render resolution for any PDF pages. 170 matches numind\'s own reference conversion approach for this model; only raise it if you know you need more fidelity for dense/small text.</li><li><code>template</code> (string, required only for mode="structured"): JSON extraction template — see NuExtract3\'s own TYPES.md for the template DSL.</li><li><code>instructions</code> (string, optional): extra guidance, mode="structured" only. <strong>Set this explicitly when the document describes the same figure both in a data table AND in narrative/discussion text</strong> — confirmed live that a bare field-name-only template gives the model no signal to prefer the exact table value over a rounded prose paraphrase of the same metric (table says "45", a later discussion paragraph says "just under half", model literalizes that into a wrong "50"). Effective instructions value: "Extract numbers ONLY from explicit data tables. Ignore approximate or rounded figures mentioned in narrative/discussion text."</li><li><code>previous_output</code> (string, optional): a prior structured result to refine/continue. Only has an effect in mode="structured".</li><li><code>system_prompt</code> (string, optional): guidance injected as a system message. NuExtract3\'s template only auto-injects mode-specific instructions for mode="structured" — every other mode gets nothing beyond the bare task label unless you provide this (confirmed live: mode="document-detection" with no system_prompt just describes the document instead of classifying it). Defaults to numind\'s own shipped markdown task instructions when mode="markdown" and this is left unset (matches how they benchmarked the model); no default for document-detection/content/template-generation.</li><li><code>enable_thinking</code> (boolean, optional, default false): only valid for mode="structured"/"content" — NuExtract3\'s own chat template rejects it for the other modes (this endpoint returns a 400 rather than letting that fail server-side).</li><li><code>max_tokens</code> (int, optional, default 4096).</li><li><code>temperature</code> (float, optional): defaults to numind\'s own reference values when unset — 0 for mode="markdown", 0.2 for other modes non-thinking, 0.6 when enable_thinking is set.</li><li><code>model</code> (string, optional, default "NuExtract3-Q8_0.gguf"): swap in a different quant if you\'ve downloaded one.</li><li><code>gpu_id</code> (int, optional): which GPU to load the OCR model onto — unset auto-picks the GPU with the most free VRAM (same as every other gateway endpoint). Set explicitly to pin OCR off a GPU that\'s also serving a model you don\'t want it contending with for *compute* (not VRAM — both can coexist in VRAM fine, but two models sharing one GPU\'s compute serialize each other\'s requests instead of running in parallel). Caveat: only affects where a *new* instance spawns — an already-running instance on a different GPU won\'t be moved, a second instance spawns on the requested GPU instead (the old one idles out via its own keep_alive). <code>keep_alive</code>: same auto-load knob every other gateway endpoint has.</li><li><code>ctx_size</code> (int, optional): defaults to 16384 (numind\'s own reference "memory-constrained" --max-model-len value — comfortably covers one page: measured ~3405 prompt tokens for a single page at pdf_dpi=170 regardless of content density, plus up to max_tokens of completion). Every request here is one page, never multiple, so numind\'s much larger 131072 reference figure — sized for their own approach of all pages in one request — doesn\'t apply here and would just cost unneeded VRAM (measured ~18.6GB vs ~6.6GB for the same page). Raise it explicitly if you set a much bigger max_tokens or a higher pdf_dpi.</li></ul>',
            responseSchema: '<ul><li><code>status</code>: "completed" (every page succeeded), "partial" (some pages failed — inspect <code>pages</code> for which/why, the rest is still usable), or "failed" (raises HTTP 502 instead of returning here — no page succeeded, or a setup error before any page ran).</li><li><code>content</code> (string or null): a single-page document\'s own content, unwrapped, same as before this endpoint had multi-page support. Multiple pages, text modes (markdown/content): each page\'s content joined with an explicit <code>&lt;!-- PAGE BREAK --&gt;</code> inserted between them by this endpoint itself (not requested from or trusted to the model). Multiple pages, JSON modes: null — use <code>parsed</code> instead.</li><li><code>parsed</code> (object, array, or null): single page + JSON-producing mode → that page\'s own parsed object, unwrapped. Multiple pages + JSON-producing mode → an array of one parsed object (or null) per page, in page order. Text modes: always null.</li><li><code>pages</code> (array, always populated, length 1 for a single-image document): the full per-page breakdown <code>content</code>/<code>parsed</code> above are derived from — each has <code>page_index</code>, <code>status</code>, <code>content</code>, <code>reasoning_content</code>, <code>parsed</code>, <code>parse_error</code>, <code>error</code>, <code>runtime_seconds</code>. Check this when <code>status</code>="partial" or to see exactly what one specific page produced.</li><li><code>runtime_seconds</code> (float): total across all pages.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/ocr -H "Content-Type: application/json" -d '{"images_base64": ["<base64>"], "mode": "markdown"}'`
        },
        {
            category: 'document',
            method: 'POST',
            path: '/api/ocr/batch',
            desc: 'OCR BATCH: Runs a whole list of documents through NuExtract3 sequentially (same reasoning as the ComfyUI batch endpoint — one GPU, no benefit to concurrent items). Each item\'s images_base64 accepts PDFs too, same as /api/ocr (auto-expanded to one page image per PDF page), and — also same as /api/ocr — every page of every item is its own request to the model, never batched together (see /api/ocr\'s own note on why). Returns a batch_id immediately; poll /api/ocr/batch/status for progress and per-item results. Only one batch can run at a time — a second call while one is running gets HTTP 409.',
            payload: {
                defaults: { mode: "document-detection", system_prompt: "Classify the type of this document (e.g. invoice, receipt, contract, letter, form, resume, id_card, other). Return a JSON object: {\"document_type\": \"...\", \"confidence\": \"high|medium|low\"}." },
                items: [
                    { id: "invoice-1", images_base64: ["<base64-page-1>"] },
                    { id: "invoice-2", images_base64: ["<base64-page-1>", "<base64-page-2>"] }
                ]
            },
            payloadSchema: '<ul><li><code>defaults</code> (object, optional): same fields as /api/ocr\'s payload minus <code>images_base64</code>/<code>id</code>, applied to every item that doesn\'t override them.</li><li><code>items</code> (array, required, 1-500 entries): each requires only <code>images_base64</code>; <code>id</code> (string, optional) is a caller-supplied label to match a result back to its source document (e.g. a filename) — echoed verbatim in the status response, never interpreted. Every other /api/ocr field is accepted per-item and overrides <code>defaults</code>.</li></ul>',
            responseSchema: '<ul><li><code>batch_id</code> (string).</li><li><code>total</code> (int): number of items queued.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/ocr/batch -H "Content-Type: application/json" -d '{"defaults": {"mode": "markdown"}, "items": [{"id": "doc-1", "images_base64": ["<base64>"]}]}'`
        },
        {
            category: 'document',
            method: 'GET',
            path: '/api/ocr/batch/status',
            desc: 'OCR BATCH STATUS: Current (or most recently finished) batch\'s progress and per-item results.',
            responseSchema: '<ul><li><code>status</code> (string): "none" (nothing has run yet), "running", "cancelled", "completed", or "failed".</li><li><code>id</code>, <code>total</code>, <code>completed_count</code>, <code>current_index</code>.</li><li><code>items</code> (array): each has <code>index</code>, <code>id</code> (the caller-supplied label, or null), <code>status</code> ("pending"/"processing"/"completed"/"partial"/"failed"/"cancelled"), and — once processed — <code>content</code>, <code>parsed</code>, <code>pages</code>, <code>error</code>, <code>runtime_seconds</code> (same shape as /api/ocr\'s response — see there for what <code>pages</code>/<code>content</code>/<code>parsed</code> mean for a multi-page item).</li></ul>',
            curl: `curl ${BASE_URL}/api/ocr/batch/status`
        },
        {
            category: 'document',
            method: 'POST',
            path: '/api/ocr/batch/cancel',
            desc: 'OCR BATCH CANCEL: Stops the batch from starting its next item — the in-flight item is left to finish (llama-server has no interrupt-current-generation endpoint to reach for, unlike ComfyUI\'s /interrupt). Every not-yet-started item is marked "cancelled". HTTP 409 if no batch is running.',
            responseSchema: '<ul><li><code>status</code> (string): "cancelled".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/ocr/batch/cancel`
        },
        {
            category: 'training',
            method: 'GET',
            path: '/api/training/health',
            desc: 'REMOTE FINE-TUNING: Health check + live GPU state for the training service. All /api/training/* endpoints require a Bearer token in TRAINING_SERVICE_TOKEN (set in .env.training) — a SEPARATE token from the dashboard\'s general API_KEY, since submitting jobs is a materially higher-stakes capability (subprocess execution, dataset upload, GPU time). Returns HTTP 500 if TRAINING_SERVICE_TOKEN is unset server-side, 401 if the wrong token is supplied.',
            responseSchema: '<ul><li><code>status</code> (string): "ok".</li><li><code>running_job</code> (string or null): job_id of the first currently running job, if any (kept for backwards compatibility — the lock is per-GPU now, so there can be more than one; see <code>running_jobs</code>).</li><li><code>running_jobs</code> (array of {job_id, gpu}): every currently running job.</li><li><code>gpus</code> (array): same shape as <code>GET /api/gpu/status</code>.</li><li><code>disk_free_gb</code> (float).</li></ul>',
            curl: `curl ${BASE_URL}/api/training/health -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'POST',
            path: '/api/training/jobs',
            desc: 'Submit a job. job_type "dpo_train" runs Unsloth+TRL DPO training on an uploaded JSONL dataset ({"prompt": [...chat messages...], "chosen": str, "rejected": str} per line), producing a LoRA adapter. job_type "sft_train" runs Unsloth+TRL SFT (supervised fine-tuning) on an uploaded JSONL dataset ({"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]} per line — a full chat transcript), also producing a LoRA adapter; loss is masked to only the assistant turns via unsloth\'s train_on_responses_only. job_type "kto_train" runs Unsloth+TRL KTO (Kahneman-Tversky Optimization) training on an uploaded JSONL dataset ({"prompt": [...chat messages...], "completion": str, "label": bool} per line — single examples, NOT paired like DPO: label true = desirable ("thumbs up"), false = undesirable ("thumbs down")), also producing a LoRA adapter. KTO\'s appeal over DPO is that it doesn\'t need matched chosen/rejected pairs on the same prompt — plain accept/reject signal (e.g. user thumbs up/down, or a pass/fail heuristic) is enough, which is often much easier to collect in practice. <strong>config.batch_size must be &gt; 1 for kto_train</strong> (default 2, unlike dpo_train\'s default 1) — confirmed live: TRL\'s KTOTrainer raises an error at construction otherwise, since its KL term needs multiple examples in the same batch to build mismatched pairs against; grad_accum doesn\'t substitute for this since it only raises the effective batch size, not the actual per-step one this check cares about. job_type "orpo_train" runs Unsloth+TRL ORPO (Odds Ratio Preference Optimization) training on an uploaded JSONL dataset — same {"prompt": [...chat messages...], "chosen": str, "rejected": str} format as dpo_train, also producing a LoRA adapter. ORPO combines SFT-style and preference learning into one training pass instead of two, and — unlike dpo_train/kto_train — needs NO reference model at all (no ref_model, no extra forward pass for a KL term), so its per-example VRAM cost is lower than DPO/KTO on the same model. There\'s no dedicated Unsloth patch for ORPO (unlike PatchDPOTrainer/PatchKTOTrainer), so it runs as a standard TRL trainer on an Unsloth-loaded PEFT model — still functional, just without whatever ORPO-specific speed optimization Unsloth\'s dedicated patches provide for DPO/KTO. <strong>config.beta means something different for orpo_train than for dpo_train/kto_train</strong> — ORPO has no KL term, so beta here is the "relative ratio loss weight" (lambda in the ORPO paper) controlling how strongly the odds-ratio preference penalty is weighted against the plain NLL/SFT loss component, not a KL regularization strength. <strong>orpo_train has a confirmed numerical failure mode on short/templated completions</strong> (e.g. classifier-style judge verdicts like "VERDICT: YES"/"VERDICT: NO"): TRL\'s ORPOTrainer.odds_ratio_loss computes log1p(-exp(logp)) from each completion\'s mean per-token logprob, a genuine singularity (-inf) once that logprob saturates near 0 — which a model reaches within a handful of steps on short, near-deterministic labels. Two guards exist for this: at submit time, a dataset preflight heuristic (median completion word count ≤6, or ≤30% distinct completions) adds a message to the job\'s <code>warnings</code> field and log if the uploaded dataset looks risky (does NOT block submission — it\'s a heuristic, not a guarantee either way); at runtime, orpo_train.py aborts training within its first 10 steps if any logged metric goes NaN/Inf, landing the job on status "failed" with a clear diagnostic in <code>error</code> instead of silently burning GPU time toward a garbage adapter. ORPO is the right choice when chosen/rejected completions are full, distributionally distinct generative responses — prefer dpo_train/kto_train instead for short-label/classifier-style tasks, since their KL-to-reference term dampens this. job_type "merge_export" merges a LoRA adapter (either freshly uploaded or referenced via config.source_job_id from a completed dpo_train/sft_train/kto_train/orpo_train job) into full precision and, unless config.output_format is "hf", converts+quantizes to GGUF via llama.cpp. "hf" stops after merging and returns the full-precision HF/safetensors checkpoint instead — MLX cannot be produced on this machine (the mlx package is Apple Silicon/Metal-only), so the remote caller runs mlx_lm.convert itself against that HF checkpoint. The concurrency lock is per-GPU, not global: two jobs pinned to different GPUs can run at once, but a second submit targeting a GPU that already has a running job gets HTTP 409. If config.gpu is omitted, an idle GPU is preferred over the most-free-VRAM one (falling back to most-free-VRAM only if every GPU already has a job running).',
            payloadSchema: '<ul><li><code>job_type</code> (string, required, form field): "dpo_train", "sft_train", "kto_train", "orpo_train", or "merge_export".</li><li><code>config</code> (string, required, form field): JSON-encoded config dict. dpo_train/sft_train/kto_train/orpo_train share most fields: <code>base_model</code> (required), <code>max_seq_length</code>, <code>lora_r</code>, <code>lora_alpha</code>, <code>lora_dropout</code>, <code>target_modules</code>, <code>learning_rate</code> (dpo_train/kto_train/orpo_train default 5e-6, sft_train default 2e-4 — very different scales, don\'t reuse one for the other), <code>num_train_epochs</code>, <code>max_steps</code>, <code>user_marker</code>, <code>assistant_marker</code>, <code>gpu</code>, <code>save_steps</code> (checkpoint every N steps, default 50 — also bounds pause/crash progress loss), <code>val_split</code> (fraction of the dataset held out as a never-trained-on validation set, fixed seed, default 0 = disabled — opt-in since it\'s meaningless on a tiny pilot dataset; tracks eval_loss, the only in-run signal that can catch a model matching the training format\'s surface pattern rather than genuinely learning the task), <code>eval_steps</code> (only used if val_split > 0; defaults to save_steps). <strong>Best-checkpoint export is automatic, not a config key you set</strong> — whenever val_split > 0 AND save_steps is a round multiple of eval_steps (e.g. both 50, or save_steps=250/eval_steps=250 — NOT save_steps=50/eval_steps=250, that combo doesn\'t qualify), the job internally turns on HF\'s load_best_model_at_end, and the adapter GET /result serves for a done job is the best-eval checkpoint, not just whichever one happened to be last (training loss can keep falling toward 0 well past the point eval_loss stopped improving, so "last" and "best" are often different checkpoints). If the save_steps/eval_steps combo doesn\'t qualify, this is silently skipped — no error, the result is just the last checkpoint instead, same as when val_split is disabled entirely. GET /jobs and GET /jobs/{id} both include a <code>best_checkpoint</code> field ({best_step, best_metric, is_latest}, or null if not applicable) so you can confirm which one actually won without inspecting internal files. <code>batch_size</code> (per_device_train_batch_size; dpo_train default 1, sft_train default 2, kto_train default 2 and <strong>must be &gt; 1</strong> for kto_train specifically — TRL\'s KTOTrainer raises at construction time otherwise, since its KL term needs multiple examples per batch to build mismatched pairs against. DPO/KTO both need a policy + implicit-reference forward pass per example, so need roughly half the per-example VRAM budget SFT gets away with on the same model — KTO needs somewhat more again on top of that for the additional KL-dataset forward pass. orpo_train default 1 — ORPO has no reference model at all (no ref_model, no extra forward pass), so its per-example VRAM cost is lower than DPO/KTO on the same model, with more headroom to raise this than DPO/KTO have), <code>grad_accum</code> (gradient_accumulation_steps, default 4 for both — raising this OR batch_size increases the effective batch size (batch_size × grad_accum) and reduces gradient-norm noise equally; only batch_size also buys wall-clock speed, since grad_accum just runs more sequential micro-batches per optimizer step), <code>early_stopping_patience</code> (int, optional, default disabled): stop training after this many consecutive evals that fail to beat the best eval_loss by more than early_stopping_threshold — wraps transformers.EarlyStoppingCallback directly. Only takes effect if load_best_model_at_end ends up active for the run (val_split > 0 AND save_steps is a multiple of eval_steps); otherwise there\'s no tracked "best" to compare against and it\'s silently skipped with a log note, same as load_best_model_at_end\'s own fallback. <code>early_stopping_threshold</code> (float, default 0.0): minimum eval_loss improvement to NOT count as a no-improvement eval — calibrate to the task\'s actual eval_loss scale, not left at 0.0 (rarely triggers) or set coarsely like 0.2 (triggers almost immediately once losses are small), <code>warmup_ratio</code> (float, default 0.0 = disabled): fraction of total steps to linearly ramp learning_rate up from 0 before the scheduler takes over — a small value like 0.03-0.05 is a common, cheap way to reduce early-training instability, <code>lr_scheduler_type</code> (string, default "linear"): HF scheduler type, e.g. "cosine" holds the learning rate higher for longer before tapering, <code>weight_decay</code> (float, default 0.01 for sft_train, 0.0 for dpo_train/kto_train/orpo_train — never reconciled across all four, left as each script\'s own default), <code>resume_from_job_id</code> (set automatically by POST .../resume — don\'t set by hand, use that endpoint instead). dpo_train/kto_train: <code>beta</code> (KL regularization strength, same meaning/default 0.1 for both). kto_train-only: <code>desirable_weight</code>/<code>undesirable_weight</code> (float, default 1.0 each) — raise one to counter a lopsided ratio of thumbs-up vs thumbs-down examples in the dataset, something DPO doesn\'t need since paired examples are balanced by construction. orpo_train also has a <code>beta</code> (default 0.1) but it means something different — ORPO has no KL term at all, so this is the "relative ratio loss weight" (lambda in the paper) controlling how strongly the odds-ratio preference penalty is weighted against ORPO\'s combined NLL/SFT loss component, not a KL regularization strength. merge_export: <code>base_model</code> (required), <code>source_job_id</code> or an uploaded adapter — if the source job is still RUNNING, its latest checkpoint is used instead of a final adapter (lets you quantize a snapshot without pausing, e.g. on a second free GPU — the per-GPU lock already prevents submitting this onto the same GPU the source job is using), <code>output_format</code> ("gguf"|"hf"), <code>quant_type</code>, <code>output_name</code>, <code>llama_cpp_dir</code>, <code>max_seq_length</code>, <code>gpu</code>.</li><li><code>dataset</code> (file, required for dpo_train/sft_train/kto_train/orpo_train unless resume_from_job_id is set): JSONL upload, format depends on job_type (see above — dpo_train, orpo_train, and kto_train all look similar but are NOT interchangeable: dpo_train/orpo_train need paired chosen/rejected on the same prompt, kto_train needs single prompt/completion/label examples).</li></ul>',
            responseSchema: '<ul><li><code>job_id</code>, <code>job_type</code>, <code>status</code> ("running"), <code>pid</code>, <code>created_at</code>, <code>started_at</code>, <code>config</code>.</li><li><code>warnings</code> (array of strings, currently orpo_train-only): non-blocking preflight risk messages computed at submit time from the uploaded dataset (see this endpoint\'s orpo_train description above) — empty for every other job_type and for orpo_train datasets that don\'t look risky.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/training/jobs -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN" -F "job_type=dpo_train" -F 'config={"base_model":"unsloth/granite-4.1-3b","max_seq_length":2048}' -F "dataset=@train_dpo.jsonl"`
        },
        {
            category: 'training',
            method: 'GET',
            path: '/api/training/jobs',
            desc: 'List all training jobs, newest first.',
            responseSchema: '<ul><li>Array of job objects (same shape as GET /api/training/jobs/{id}), each with added <code>disk_usage</code> and (dpo_train/sft_train/kto_train/orpo_train only) <code>best_checkpoint</code> objects — see GET /api/training/jobs/{id} for both. <code>disk_usage.freeable_bytes</code> depends on job_type/output_format, not always "output/ + merged_hf/": for merge_export with output_format "gguf" it\'s just merged_hf/ (output/ holds the actual .gguf deliverable), for output_format "hf" it\'s 0 (merged_hf/ IS the deliverable there), for dpo_train/sft_train/kto_train/orpo_train it\'s output/ (checkpoints + adapter). Both fields computed at request time (not cached, not stored on the job) — fast even across dozens of multi-GB jobs since disk_usage never reads file content and best_checkpoint only reads one small JSON file per job.</li></ul>',
            curl: `curl ${BASE_URL}/api/training/jobs -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'GET',
            path: '/api/training/jobs/{job_id}',
            desc: 'Get one job\'s status. Reconstructed from PID existence + log content on read (not in-memory state), so it survives a backend restart.',
            responseSchema: '<ul><li><code>status</code>: "running", "done", "failed", "cancelled", or "paused".</li><li><code>exit_code</code>, <code>error</code>, <code>finished_at</code>.</li><li><code>disk_usage</code>: {total_bytes, freeable_bytes} — see GET /jobs.</li><li><code>best_checkpoint</code> (dpo_train/sft_train/kto_train/orpo_train only, null if not applicable): {best_step, best_metric, is_latest} — read from the latest checkpoint\'s trainer_state.json (HF updates best_model_checkpoint/best_metric cumulatively as training progresses). null if there\'s no checkpoint yet or val_split was never used (nothing to track a "best" against). is_latest tells you whether the best checkpoint IS the most recent one (no ongoing overfitting past that point) or an earlier one (training kept going past its actual best).</li><li><code>warnings</code> (array of strings, currently orpo_train-only): see POST /jobs\' response schema — set once at submit time, doesn\'t change as the job runs.</li></ul>',
            curl: `curl ${BASE_URL}/api/training/jobs/<job_id> -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'GET',
            path: '/api/training/jobs/{job_id}/log',
            desc: 'Get a job\'s full stdout/stderr log, or the last N lines via ?tail=N. No streaming/SSE — polling is enough at this scale.',
            responseSchema: '<ul><li><code>log</code> (string).</li></ul>',
            curl: `curl "${BASE_URL}/api/training/jobs/<job_id>/log?tail=50" -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'POST',
            path: '/api/training/jobs/{job_id}/cancel',
            desc: 'Cancel a running job — hard stop (SIGTERM to the process group, SIGKILL after a 10s grace period if still alive). NOT resumable: an abruptly-killed process has no guaranteed-clean checkpoint. Use /pause instead if you might want to continue later. No-op if the job already finished.',
            responseSchema: '<ul><li>Updated job status object (status becomes "cancelled").</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/training/jobs/<job_id>/cancel -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'POST',
            path: '/api/training/jobs/{job_id}/pause',
            desc: 'Gracefully pause a running dpo_train, sft_train, kto_train, or orpo_train job: writes a sentinel file that the shared PauseCallback checks after every step — the trainer saves a checkpoint and exits cleanly on its own, actually freeing the GPU (unlike an OS-level suspend, which would leave all its VRAM allocated). Returns immediately while still "running"; the status becomes "paused" once the subprocess notices the flag and exits (poll GET /jobs/{job_id} to see the transition). Not available for merge_export jobs (nothing to checkpoint).',
            responseSchema: '<ul><li>Job status object — still "running" in this response; check again shortly for "paused".</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/training/jobs/<job_id>/pause -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'POST',
            path: '/api/training/jobs/{job_id}/resume',
            desc: 'Continue a paused dpo_train, sft_train, kto_train, or orpo_train job as a brand-new job of the same type (not a modification of the paused one) — reuses its exact config and already-uploaded dataset (no re-upload needed) and resumes from its latest checkpoint (TRL/HF\'s own resume_from_checkpoint mechanism: restores model/optimizer/scheduler state and step count). Optional ?gpu=N query param overrides which GPU the resumed job uses. HTTP 409 if another job is already running on the target GPU, 404 if the source job has no checkpoint to resume from.',
            responseSchema: '<ul><li>The newly created job\'s status object (a different job_id from the one you resumed).</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/training/jobs/<job_id>/resume -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'GET',
            path: '/api/training/jobs/{job_id}/result',
            desc: 'Download a finished (or paused) job\'s result. dpo_train/sft_train/kto_train/orpo_train and merge_export with output_format "hf" return a zip (LoRA adapter or merged HF checkpoint respectively, both directories with multiple files); merge_export with output_format "gguf" (default) returns the raw .gguf file. For a done/paused dpo_train/sft_train/kto_train/orpo_train job, this is the BEST-eval checkpoint\'s weights whenever load_best_model_at_end was active for that run (see POST /jobs\' config.val_split/save_steps/eval_steps), not just whichever checkpoint happened to be last — HF Trainer reloads the best weights into the model before writing this adapter, automatically, no separate download/flag needed. Check the job\'s best_checkpoint field (GET /jobs/{job_id}) to see which step actually won. dpo_train/sft_train/kto_train/orpo_train jobs can also be downloaded while still RUNNING — this returns the latest checkpoint (an in-progress adapter snapshot, not the final result, since trainer.save_model() hasn\'t run yet), filename tagged with its checkpoint step number. merge_export has no meaningful partial result, so it still requires done/paused. HTTP 409 if the job doesn\'t qualify for any of the above (failed/cancelled, or a running merge_export); HTTP 404 if the expected result file is missing on disk (e.g. moved/deleted externally after completion).',
            curl: `curl ${BASE_URL}/api/training/jobs/<job_id>/result -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN" -o result.zip`
        },
        {
            category: 'training',
            method: 'DELETE',
            path: '/api/training/jobs/{job_id}',
            desc: 'Delete a job entirely — job entry, log, data.jsonl, checkpoints, and output all removed; the job disappears from GET /jobs. Blocked (HTTP 409) if the job is currently running (stop or pause it first) or if any other job still depends on it via config.resume_from_job_id (a Continue) or config.source_job_id (a merge_export) — delete those first if you\'re sure, or leave this one alone. Irreversible.',
            responseSchema: '<ul><li><code>status</code>: "deleted".</li><li><code>job_id</code>.</li></ul>',
            curl: `curl -X DELETE ${BASE_URL}/api/training/jobs/<job_id> -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'POST',
            path: '/api/training/jobs/{job_id}/free-disk',
            desc: 'Lighter alternative to full delete: removes only output/ (checkpoints + the final adapter) and merged_hf/ (merge_export\'s full-precision intermediate) — the big files — while keeping job.json/log.txt/data.jsonl, so the job stays visible in the list with its config and history intact. Same blocking rules as delete (must not be running, must not be depended on by another job\'s resume_from_job_id/source_job_id). Irreversible.',
            responseSchema: '<ul><li><code>status</code>: "freed".</li><li><code>job_id</code>.</li><li><code>freed_bytes</code> (int): total size of the removed files.</li></ul>',
            curl: `curl -X POST ${BASE_URL}/api/training/jobs/<job_id>/free-disk -H "Authorization: Bearer $TRAINING_SERVICE_TOKEN"`
        },
        {
            category: 'training',
            method: 'GET',
            path: '/api/training-monitor/*',
            desc: 'Unauthenticated mirror of the routes above, for this dashboard\'s own "Training" tab — NOT for remote/external use. Same paths minus the /api/training prefix (e.g. <code>GET /api/training-monitor/jobs</code>, <code>POST /api/training-monitor/jobs/{job_id}/pause</code>). Covers list/log/download/pause/cancel/resume/delete/free-disk with no Bearer token required, but deliberately excludes job *submission* — there\'s no <code>POST /api/training-monitor/jobs</code>, so an unauthenticated caller can control the lifecycle of (or delete) a job someone already legitimately submitted (via the token-gated routes above) but can never inject an arbitrary base_model/dataset to run new code. Exists because embedding TRAINING_SERVICE_TOKEN in this page\'s own JS would leak it to anyone viewing source, defeating the point of having a separate, stronger secret than the dashboard\'s general API_KEY. Shares this dashboard\'s own (currently unauthenticated) trust boundary — covered by API_KEY\'s middleware if that\'s ever set, same as every other /api/* route.',
            curl: `curl ${BASE_URL}/api/training-monitor/jobs`
        }
    ];

    window.apiEndpointsList = endpoints;

    const categories = [
        { key: 'text', title: '🔤 Text & LLM Gateway Endpoints', desc: 'Models, Chat Completions, GPU status, Does It Work? check' },
        { key: 'image', title: '🖼️ Image Generation & Edit Endpoints', desc: 'ComfyUI image generation, editing, style transfer, task status' },
        { key: 'audio', title: '🎙️ Speech & Audio Endpoints', desc: 'OmniVoice TTS, WhisperX STT, local voice presets' },
        { key: 'document', title: '📄 OCR / Document Extraction Endpoints', desc: 'NuExtract3 structured extraction, image-to-Markdown, document-type detection, batch processing' },
        { key: 'training', title: '🧪 Remote Fine-Tuning Endpoints', desc: 'DPO training + merge/GGUF export jobs, gated by a separate TRAINING_SERVICE_TOKEN' }
    ];

    // Jump-to nav: one pill per non-empty category, scrolls its header into view.
    // Plain (not sticky) -- avoids stacking-context/z-index edge cases with the
    // modals this tab can also open; easy to make sticky later if wanted.
    let html = `
        <div style="grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 0.5rem; padding-bottom: 0.85rem; margin-bottom: 0.5rem; border-bottom: 1px solid var(--border);">
            ${categories
                .filter(cat => endpoints.some(ep => ep.category === cat.key))
                .map(cat => `<button type="button" class="btn btn-secondary" style="padding:0.35rem 0.75rem;font-size:0.78rem;" onclick="document.getElementById('api-cat-${cat.key}').scrollIntoView({behavior:'smooth',block:'start'})">${cat.title}</button>`)
                .join('')}
        </div>
    `;

    categories.forEach(cat => {
        const catEndpoints = endpoints.filter(ep => ep.category === cat.key);
        if (catEndpoints.length === 0) return;

        html += `
            <div id="api-cat-${cat.key}" class="api-category-header" style="grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center; background: var(--bg-hover); padding: 0.75rem 1rem; border-radius: 8px; margin-top: 1rem; scroll-margin-top: 1rem;">
                <div>
                    <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">${cat.title}</h3>
                    <span style="font-size: 0.85rem; color: var(--text-muted);">${cat.desc}</span>
                </div>
                <button type="button" class="btn btn-secondary" onclick="copySectionApiDocs('${cat.key}')" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">📋 Copy ${cat.key.toUpperCase()} API Specs</button>
            </div>
        `;

        catEndpoints.forEach(ep => {
            html += `
                <div class="api-explorer-card">
                    <div class="api-endpoint">
                        <span><span style="color: ${ep.method === 'GET' ? 'var(--success)' : 'var(--primary)'}">${ep.method}</span> ${ep.path}</span>
                    </div>
                    <div class="api-desc">${ep.desc}</div>
                    
                    ${ep.payload ? `
                        <div class="api-section-title">Request Payload Example:</div>
                        <div class="api-payload-box">
                            <pre>${JSON.stringify(ep.payload, null, 2)}</pre>
                        </div>
                    ` : ''}

                    ${ep.payloadSchema ? `
                        <div class="api-section-title">Request Body Schema:</div>
                        <div class="api-schema-desc">${ep.payloadSchema}</div>
                    ` : ''}

                    ${ep.responseSchema ? `
                        <div class="api-section-title">Response Schema:</div>
                        <div class="api-schema-desc">${ep.responseSchema}</div>
                    ` : ''}

                    ${ep.examples ? `
                        <div class="api-section-title">Examples:</div>
                        ${apiExamples(ep.examples)}
                    ` : ''}

                    ${ep.method === 'GET' ? `
                        <div class="api-section-title">Live Response Data:</div>
                        <div class="api-live-data">
                            ${ep.liveResponse ? `<pre>${JSON.stringify(ep.liveResponse, null, 2)}</pre>` : '<div class="api-desc" style="font-style: italic">Waiting for data...</div>'}
                        </div>
                    ` : ''}

                    <div class="api-actions">
                        <button class="btn-copy" onclick="copyToClipboard('${ep.curl.replace(/'/g, "\\'").replace(/"/g, '&quot;')}')">Copy cURL</button>
                    </div>
                </div>
            `;
        });
    });

    UI.apiExplorerList.innerHTML = html;
}

function copyToClipboard(text) {
    copyTextToClipboard(text).then(ok => logCopyResult(ok, 'API command'));
}

window.copyToClipboard = copyToClipboard;

// ----------------------------------------------------
// Global Polling loop
// ----------------------------------------------------
async function pollSystemStatus() {
    try {
        // GPU status
        const gpuData = await api('/api/gpu/status');
        lastApiData.gpu = gpuData;
        renderGPUContainer(gpuData.gpus || []);
        
        // Instances status
        const statusData = await api('/api/status');
        lastApiData.status = statusData;
        renderInstancesList(statusData.instances || []);

        // Update indicators
        UI.statusBackend.className = 'status-indicator online';
        
        // Check if ComfyUI is active
        const comfyInstance = statusData.instances.find(inst => inst.port === 8189);
        const comfyActive = !!comfyInstance;
        if (comfyActive) {
            UI.statusComfyui.className = 'status-indicator online';
            UI.comfyuiStatusIndicator.className = 'status-indicator online';
            const comfyGpuLabel = comfyInstance.gpu_id !== null && comfyInstance.gpu_id !== undefined ? `GPU ${comfyInstance.gpu_id}` : 'Auto';
            UI.comfyuiStatusIndicator.querySelector('.status-text').textContent = `Running (${comfyGpuLabel})`;
            UI.startComfyuiBtn.style.display = 'none';
            UI.stopComfyuiBtn.style.display = 'inline-block';
            UI.openComfyuiLink.style.display = 'inline-block';
            UI.openComfyuiLink.href = '#';  // resolved on click, see openComfyuiNativeUi()
            // The GPU picker only takes effect on ComfyUI's next fresh start -- see
            // ComfyUIStartRequest.gpu_id -- so disable it while already running rather
            // than let it silently do nothing if changed and "started" again.
            UI.comfyuiGpuSelect.disabled = true;
        } else {
            UI.statusComfyui.className = 'status-indicator offline';
            UI.comfyuiStatusIndicator.className = 'status-indicator offline';
            UI.comfyuiGpuSelect.disabled = false;
            UI.comfyuiStatusIndicator.querySelector('.status-text').textContent = 'Stopped';
            UI.startComfyuiBtn.style.display = 'inline-block';
            UI.stopComfyuiBtn.style.display = 'none';
            UI.openComfyuiLink.style.display = 'none';
        }

        // Re-render Developer API Explorer with fresh live system status
        renderAPIExplorer();
    } catch (err) {
        UI.statusBackend.className = 'status-indicator offline';
        UI.statusComfyui.className = 'status-indicator offline';
        console.error('System status poll failed', err);
    }
}

// GGUF Header Inspector — dumps a model's raw KV metadata (architecture, context
// length, quantization, rope/vision/MTP config, chat template, etc.) so you don't
// have to reverse-engineer it from the filename or a one-off script.
function initGgufHeaderInspector() {
    if (!UI.inspectHeaderBtn) return;

    UI.inspectHeaderBtn.addEventListener('click', async () => {
        const modelPath = UI.modelSelect.value;
        if (!modelPath) return addLog('Select a model first.', 'warn');

        const modelName = UI.modelSelect.selectedOptions[0]?.textContent || modelPath;
        UI.ggufHeaderTitle.textContent = `GGUF Header — ${modelName}`;
        UI.ggufHeaderContent.textContent = 'Loading...';
        UI.ggufHeaderModal.style.display = 'flex';

        try {
            const data = await api(`/api/models/header?path=${encodeURIComponent(modelPath)}`);
            UI.ggufHeaderContent.textContent = JSON.stringify(data, null, 2);
        } catch (err) {
            UI.ggufHeaderContent.textContent = `Failed to read header: ${err.message}`;
        }
    });

    UI.closeGgufHeaderBtn.addEventListener('click', () => {
        UI.ggufHeaderModal.style.display = 'none';
    });

    UI.ggufHeaderModal.addEventListener('click', (e) => {
        if (e.target === UI.ggufHeaderModal) {
            UI.ggufHeaderModal.style.display = 'none';
        }
    });

    UI.copyGgufHeaderBtn.addEventListener('click', async () => {
        logCopyResult(await copyTextToClipboard(UI.ggufHeaderContent.textContent), 'GGUF header JSON');
    });
}

// Initial Loading sequence
(async function init() {
    addLog('System Console Initialized.', 'system');
    initGgufHeaderInspector();
    await loadModels();
    await pollSystemStatus();
    setInterval(pollSystemStatus, 3000);
    try {
        lastApiData.version = await api('/api/version');
        renderAPIExplorer();
    } catch (err) {
        console.error('Failed to load version info', err);
    }
})();



// ComfyUI listens on 127.0.0.1 only; beyond this machine it is reachable only via
// the Tailscale level chosen for it in the Services tab.
async function openComfyuiNativeUi(e) {
    e.preventDefault();
    const local = ['localhost', '127.0.0.1', '::1', '[::1]'].includes(window.location.hostname);
    if (local) {
        window.open('http://localhost:8189', '_blank', 'noopener');
        return;
    }
    try {
        const data = await api('/api/services');
        const comfy = data.services.find(s => s.id === 'comfyui');
        if (comfy && comfy.exposure && comfy.exposure.url) {
            window.open(comfy.exposure.url, '_blank', 'noopener');
            return;
        }
    } catch (err) {
        console.error('services lookup failed', err);
    }
    addLog('ComfyUI is only reachable on the host itself -- choose "Reachable for" on its card in the Services tab.', 'error');
    document.querySelector('.tab-link[data-tab="services"]').click();
}
UI.openComfyuiLink.addEventListener('click', openComfyuiNativeUi);
