// Spielwiese -- moved out of the ai-serve-ui dashboard (2026-10-16): chat, image
// generation and speech tools against the gateway. Calls go to this app's own
// server, which forwards only the playground endpoints (spielwiese/policy.py).

const BASE_URL = window.location.origin;

// Global States
let activeTab = 'overview';
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
let trainingPollInterval = null;
let gatewayChatHistory = [];

// Multimodal input base64 states
let currentMultimodalBase64 = null;
let currentMultimodalFormat = null; // "image" or "audio"
let currentMultimodalFilename = "";


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
// Behind the door, door-session.js (loaded first) keeps requests working when the login runs out: it renews
// the login in the background and sends the request again -- no reload, nothing typed is lost. What's left
// here: Accept: application/json, so the door answers a lost login with 401 instead of starting a login flow
// for every background poll (that overwrote the login being completed -- "pkce verifier expired or missing").
async function swFetch(url, options = {}) {
    try {
        return await fetch(url, { ...options, headers: { Accept: 'application/json', ...(options.headers || {}) } });
    } catch (err) {
        throw new Error(tx('No connection to the Spielwiese.'));
    }
}

async function api(path, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    if (body) {
        options.body = JSON.stringify(body);
    }
    const resp = await swFetch(`${BASE_URL}${path}`, options);
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
    if (type === 'error' || type === 'warn') showToast(msg, type);
}

// Errors and warnings must be seen, not only land in the (usually folded) log: a note at the bottom right,
// gone after a while or on click. Found on a Mac: an image request answered 507 and the page showed nothing.
function showToast(msg, type = 'error') {
    let box = document.getElementById('sw-toasts');
    if (!box) {
        box = document.createElement('div');
        box.id = 'sw-toasts';
        box.setAttribute('role', 'status');
        box.setAttribute('aria-live', 'polite');
        document.body.appendChild(box);
    }
    const t = document.createElement('div');
    t.className = `sw-toast ${type}`;
    t.textContent = msg;
    t.title = tx('Click to close');
    t.addEventListener('click', () => t.remove());
    box.appendChild(t);
    while (box.children.length > 4) box.firstChild.remove();
    setTimeout(() => t.remove(), type === 'error' ? 15000 : 8000);
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
    addLog(ok ? tx('{label} copied to the clipboard.', { label }) : tx('Copy failed — clipboard access is blocked on this connection (try opening the page via https:// or localhost).'), ok ? 'system' : 'error');
}

// ----------------------------------------------------
// Modes & theme
// ----------------------------------------------------
UI.tabLinks.forEach(link => {
    link.addEventListener('click', () => {
        UI.tabLinks.forEach(l => { l.classList.remove('active'); l.setAttribute('aria-selected', 'false'); });
        UI.tabContents.forEach(c => c.classList.remove('active'));
        link.classList.add('active');
        link.setAttribute('aria-selected', 'true');
        activeTab = link.getAttribute('data-tab');
        document.getElementById(`${activeTab}-tab`).classList.add('active');
        try { localStorage.setItem('spielwiese-tab', activeTab); } catch { /* private mode */ }
    });
});

UI.themeToggle.addEventListener('click', () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    try { localStorage.setItem('spielwiese-theme', currentTheme); } catch { /* private mode */ }
});
try { currentTheme = localStorage.getItem('spielwiese-theme') || currentTheme; } catch { /* private mode */ }
document.documentElement.setAttribute('data-theme', currentTheme);

// ----------------------------------------------------
// Gateway Chat Logic
// ----------------------------------------------------
UI.clearChatBtn.addEventListener('click', () => {
    UI.chatMessages.innerHTML = `
        <div class="chat-system-message">
            ${tx('🤖 Write a message below – the chosen model is loaded automatically the first time.')}
        </div>
    `;
    gatewayChatHistory = [];
    addLog(tx('Chat history cleared.'), 'system');
    updateTokenEstimate();
});

// Drag & Drop / File select for Gateway Multimodal
UI.multimodalUploadZone.addEventListener('click', () => UI.multimodalInput.click());
UI.multimodalInput.addEventListener('change', handleGatewayFileSelect);

// Handle Gateway Multimodal File Uploader
function handleGatewayFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    currentMultimodalFilename = file.name;
    const reader = new FileReader();
    reader.onload = function(evt) {
        const base64Content = evt.target.result.split(',')[1];
        currentMultimodalBase64 = base64Content;
        
        UI.multimodalPlaceholder.style.display = 'none';
        UI.multimodalPreviewContainer.style.display = 'block';
        UI.multimodalFilename.textContent = file.name;

        if (file.type.startsWith('image/')) {
            currentMultimodalFormat = 'image';
            UI.multimodalImgPreview.src = evt.target.result;
            UI.multimodalImgPreview.style.display = 'block';
            UI.multimodalAudioPreview.style.display = 'none';
            addLog(tx('Attached: {name}', { name: file.name }), 'system');
        } else if (file.type.startsWith('audio/')) {
            currentMultimodalFormat = 'audio';
            UI.multimodalAudioPreview.src = evt.target.result;
            UI.multimodalAudioPreview.style.display = 'block';
            UI.multimodalImgPreview.style.display = 'none';
            addLog(tx('Audio attached: {name}', { name: file.name }), 'system');
        } else {
            addLog(tx('This file type doesn\'t work – please choose an image or audio.'), 'warn');
            clearGatewayFile();
        }
    };
    reader.readAsDataURL(file);
}

UI.removeMultimodal.addEventListener('click', (e) => {
    e.stopPropagation();
    clearGatewayFile();
});

function clearGatewayFile() {
    currentMultimodalBase64 = null;
    currentMultimodalFormat = null;
    currentMultimodalFilename = "";
    UI.multimodalInput.value = "";
    UI.multimodalPlaceholder.style.display = 'block';
    UI.multimodalPreviewContainer.style.display = 'none';
    UI.multimodalImgPreview.src = "";
    UI.multimodalAudioPreview.src = "";
    UI.multimodalFilename.textContent = "";
}

// Send Chat Message to Gateway
UI.chatSendBtn.addEventListener('click', sendGatewayChatMessage);
UI.chatUserInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendGatewayChatMessage();
    }
});

// Live Token Listeners
UI.chatUserInput.addEventListener('input', updateTokenEstimate);
UI.chatUserInput.addEventListener('keyup', updateTokenEstimate);
UI.chatUserInput.addEventListener('paste', () => setTimeout(updateTokenEstimate, 20));
UI.chatUserInput.addEventListener('change', updateTokenEstimate);

UI.gatewaySystemPrompt.addEventListener('input', updateTokenEstimate);
UI.gatewaySystemPrompt.addEventListener('keyup', updateTokenEstimate);
UI.gatewaySystemPrompt.addEventListener('paste', () => setTimeout(updateTokenEstimate, 20));
UI.gatewaySystemPrompt.addEventListener('change', updateTokenEstimate);

function selectedChatModel() {
    return (lastApiData.llm || []).find(m => m.id === UI.gatewayModelSelect.value);
}

// What the chosen model can do: a note for Strata / device-only models, thinking only where the model has it.
function updateChatModelHints() {
    const m = selectedChatModel();
    const hint = document.getElementById('gateway-model-hint');
    if (hint) hint.textContent = ChatModels.note(m);
    const canThink = !m || m.thinking;
    UI.gatewayThinking.disabled = !canThink;
    if (!canThink) UI.gatewayThinking.checked = false;
}

UI.gatewayModelSelect.addEventListener('change', () => { updateChatModelHints(); updateTokenEstimate(); });

// Strategy Buttons Listeners
if (UI.strategyPruneBtn) {
    UI.strategyPruneBtn.addEventListener('click', () => {
        if (pruneOldestMessagePair()) {
            reRenderChat();
            updateTokenEstimate();
        } else {
            addLog(tx('The history is empty or too short to trim.'), 'warn');
        }
    });
}
if (UI.strategyClearBtn) {
    UI.strategyClearBtn.addEventListener('click', () => {
        UI.clearChatBtn.click();
    });
}
if (UI.strategyAutoshrinkBtn) {
    UI.strategyAutoshrinkBtn.addEventListener('click', () => {
        autoPruneEnabled = !autoPruneEnabled;
        if (autoPruneEnabled) {
            autoSummarizeEnabled = false;
            if (UI.strategyAutosummarizeBtn) {
                UI.strategyAutosummarizeBtn.classList.remove('active');
                UI.strategyAutosummarizeBtn.textContent = tx('📝 Auto-summarise: off');
            }
        }
        UI.strategyAutoshrinkBtn.classList.toggle('active', autoPruneEnabled);
        UI.strategyAutoshrinkBtn.textContent = autoPruneEnabled ? tx('🤖 Auto-trim: on') : tx('🤖 Auto-trim: off');
        addLog(autoPruneEnabled ? tx('Auto-trim switched on.') : tx('Auto-trim switched off.'), 'system');
        updateTokenEstimate();
    });
}
if (UI.strategyAutosummarizeBtn) {
    UI.strategyAutosummarizeBtn.addEventListener('click', () => {
        autoSummarizeEnabled = !autoSummarizeEnabled;
        if (autoSummarizeEnabled) {
            autoPruneEnabled = false;
            if (UI.strategyAutoshrinkBtn) {
                UI.strategyAutoshrinkBtn.classList.remove('active');
                UI.strategyAutoshrinkBtn.textContent = tx('🤖 Auto-trim: off');
            }
        }
        UI.strategyAutosummarizeBtn.classList.toggle('active', autoSummarizeEnabled);
        UI.strategyAutosummarizeBtn.textContent = autoSummarizeEnabled ? tx('📝 Auto-summarise: on') : tx('📝 Auto-summarise: off');
        addLog(autoSummarizeEnabled ? tx('Auto-summarise switched on.') : tx('Auto-summarise switched off.'), 'system');
        updateTokenEstimate();
    });
}

// Token Estimation State & Functions
let autoPruneEnabled = false;
let autoSummarizeEnabled = false;
let isSummarizing = false;

function estimateTokens(text) {
    if (!text) return 0;
    // Standard rule of thumb: ~4 characters per token or 0.75 words per token.
    // We average them for slightly better robustness.
    const charEstimate = text.length / 4;
    const words = text.trim().split(/\s+/).filter(Boolean);
    const wordEstimate = words.length * 1.3;
    return Math.ceil((charEstimate + wordEstimate) / 2);
}

function getDefaultCtxSizeForModel(modelName) {
    if (!modelName) return 32000;
    const name = modelName.toLowerCase();
    if (name.includes('gemma-2') || name.includes('gemma2')) return 8192;
    if (name.includes('gemma')) return 8192;
    if (name.includes('llama-3.1') || name.includes('llama-3.2') || name.includes('llama3.1') || name.includes('llama3.2')) return 131072;
    if (name.includes('llama-3') || name.includes('llama3')) return 8192;
    if (name.includes('llama-2') || name.includes('llama2')) return 4096;
    if (name.includes('mistral') || name.includes('mixtral')) return 32768;
    if (name.includes('phi-3') || name.includes('phi3') || name.includes('phi-4')) return 131072;
    if (name.includes('qwen')) return 32000;
    return 32000; // default fallback
}

function getActiveModelLimit() {
    const known = ChatModels.contextFor(selectedChatModel());
    if (known) return known;
    let selectedModel = UI.gatewayModelSelect.value;
    
    // If set to Auto, determine model using running instances or first configured instance
    if (!selectedModel && lastApiData.status && lastApiData.status.instances) {
        const runningInst = lastApiData.status.instances.find(inst => inst.running);
        if (runningInst) {
            selectedModel = runningInst.model.split('/').pop();
        } else {
            const firstInst = lastApiData.status.instances[0];
            if (firstInst) {
                selectedModel = firstInst.model.split('/').pop();
            }
        }
    }
    
    // Check instances for configured ctx_size (prefer running, fallback to configured)
    if (lastApiData.status && lastApiData.status.instances) {
        if (selectedModel) {
            // First check running instance
            let match = lastApiData.status.instances.find(inst => 
                inst.running && 
                (inst.model.toLowerCase().includes(selectedModel.toLowerCase()) || 
                 selectedModel.toLowerCase().includes(inst.model.toLowerCase()))
            );
            // Fallback to configured but stopped instance
            if (!match) {
                match = lastApiData.status.instances.find(inst => 
                    (inst.model.toLowerCase().includes(selectedModel.toLowerCase()) || 
                     selectedModel.toLowerCase().includes(inst.model.toLowerCase()))
                );
            }
            if (match && match.ctx_size) {
                return match.ctx_size;
            }
        } else {
            const runningInst = lastApiData.status.instances.find(inst => inst.running);
            if (runningInst && runningInst.ctx_size) {
                return runningInst.ctx_size;
            }
            const firstInst = lastApiData.status.instances[0];
            if (firstInst && firstInst.ctx_size) {
                return firstInst.ctx_size;
            }
        }
    }
    
    // Fallback heuristisch nach Modellname
    if (selectedModel) {
        return getDefaultCtxSizeForModel(selectedModel);
    }
    
    return 32000; // Default limit fallback
}

let lastUserText = null;
let lastHistoryText = null;
let lastMaxLimit = null;
let lastWasExact = false;

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const debouncedFetchExactTokens = debounce(async (userText, historyText, maxLimit) => {
    const modelName = UI.gatewayModelSelect.value || null;
    
    try {
        const promises = [];
        if (userText) {
            promises.push(api('/api/tokenize', 'POST', { content: userText, model_name: modelName }));
        } else {
            promises.push(Promise.resolve({ success: true, tokens_count: 0 }));
        }
        
        if (historyText) {
            promises.push(api('/api/tokenize', 'POST', { content: historyText, model_name: modelName }));
        } else {
            promises.push(Promise.resolve({ success: true, tokens_count: 0 }));
        }
        
        const [userRes, historyRes] = await Promise.all(promises);
        
        if (userRes.success && historyRes.success) {
            const exactInput = userRes.tokens_count;
            const exactHistory = historyRes.tokens_count;
            const exactTotal = exactInput + exactHistory;
            
            // Ensure UI is only updated if the user has not typed something new in the meantime
            if (UI.chatUserInput.value === userText) {
                lastWasExact = true;
                if (UI.tokenInputCount) UI.tokenInputCount.textContent = exactInput;
                if (UI.tokenHistoryCount) UI.tokenHistoryCount.textContent = exactTotal;
                
                if (UI.tokenProgressFill) {
                    const percentage = Math.min(100, (exactTotal / maxLimit) * 100);
                    UI.tokenProgressFill.style.width = `${percentage}%`;
                    
                    UI.tokenProgressFill.classList.remove('warning', 'danger');
                    if (percentage >= 90) {
                        UI.tokenProgressFill.classList.add('danger');
                    } else if (percentage >= 75) {
                        UI.tokenProgressFill.classList.add('warning');
                    }
                }
                
                if (UI.tokenWarningActions) {
                    if (exactTotal >= maxLimit) {
                        UI.tokenWarningActions.style.display = 'flex';
                        UI.tokenWarningActions.classList.add('danger');
                        if (UI.tokenWarningMsg) {
                            UI.tokenWarningMsg.innerHTML = `⚠️ <strong>${tx('Limit exceeded!')}</strong> (${exactTotal} > ${maxLimit})`;
                        }
                        if (autoPruneEnabled) {
                            if (pruneOldestMessagePair()) {
                                updateTokenEstimate();
                            }
                        } else if (autoSummarizeEnabled) {
                            summarizeOldestTurns();
                        }
                    } else if (exactTotal >= maxLimit * 0.8) {
                        UI.tokenWarningActions.style.display = 'flex';
                        UI.tokenWarningActions.classList.remove('danger');
                        if (UI.tokenWarningMsg) {
                            UI.tokenWarningMsg.innerHTML = `⚠️ <strong>${tx('Limit almost reached!')}</strong> (${exactTotal} / ${maxLimit})`;
                        }
                    } else {
                        UI.tokenWarningActions.style.display = 'none';
                    }
                }
            }
        } else {
            lastWasExact = false;
        }
    } catch (err) {
        lastWasExact = false;
        console.warn("Exact tokenization request failed, falling back to heuristic:", err);
    }
}, 350);

function updateTokenEstimate() {
    const userText = UI.chatUserInput.value;
    
    const systemPrompt = UI.gatewaySystemPrompt.value.trim();
    let historyText = "";
    if (systemPrompt) {
        historyText += systemPrompt + "\n";
    }
    gatewayChatHistory.forEach(msg => {
        if (typeof msg.content === 'string') {
            historyText += msg.content + "\n";
        } else if (Array.isArray(msg.content)) {
            msg.content.forEach(part => {
                if (part.type === 'text' && part.text) {
                    historyText += part.text + "\n";
                }
            });
        }
    });
    
    const maxLimit = getActiveModelLimit();
    
    const hasActiveModel = lastApiData.status && lastApiData.status.instances && lastApiData.status.instances.some(inst => inst.running);
    
    if (userText === lastUserText && historyText === lastHistoryText && maxLimit === lastMaxLimit) {
        if (lastWasExact || !hasActiveModel) {
            return;
        }
    }
    
    lastUserText = userText;
    lastHistoryText = historyText;
    lastMaxLimit = maxLimit;
    lastWasExact = false;
    
    const inputTokens = estimateTokens(userText);
    const historyTokens = estimateTokens(historyText);
    const totalTokens = inputTokens + historyTokens;
    
    // Update heuristic immediately
    if (UI.tokenInputCount) UI.tokenInputCount.textContent = inputTokens;
    if (UI.tokenHistoryCount) UI.tokenHistoryCount.textContent = totalTokens;
    if (UI.tokenMaxLimit) UI.tokenMaxLimit.textContent = maxLimit;
    
    if (UI.tokenProgressFill) {
        const percentage = Math.min(100, (totalTokens / maxLimit) * 100);
        UI.tokenProgressFill.style.width = `${percentage}%`;
        
        UI.tokenProgressFill.classList.remove('warning', 'danger');
        if (percentage >= 90) {
            UI.tokenProgressFill.classList.add('danger');
        } else if (percentage >= 75) {
            UI.tokenProgressFill.classList.add('warning');
        }
    }
    
    if (UI.tokenWarningActions) {
        if (totalTokens >= maxLimit) {
            UI.tokenWarningActions.style.display = 'flex';
            UI.tokenWarningActions.classList.add('danger');
            if (UI.tokenWarningMsg) {
                UI.tokenWarningMsg.innerHTML = `⚠️ <strong>${tx('Limit exceeded!')}</strong> (${totalTokens} > ${maxLimit})`;
            }
            if (autoPruneEnabled) {
                if (pruneOldestMessagePair()) {
                    updateTokenEstimate();
                }
            } else if (autoSummarizeEnabled) {
                summarizeOldestTurns();
            }
        } else if (totalTokens >= maxLimit * 0.8) {
            UI.tokenWarningActions.style.display = 'flex';
            UI.tokenWarningActions.classList.remove('danger');
            if (UI.tokenWarningMsg) {
                UI.tokenWarningMsg.innerHTML = `⚠️ <strong>${tx('Limit almost reached!')}</strong> (${totalTokens} / ${maxLimit})`;
            }
        } else {
            UI.tokenWarningActions.style.display = 'none';
        }
    }

    // Trigger exact token calculation from server (debounced)
    debouncedFetchExactTokens(userText, historyText, maxLimit);
}

function pruneOldestMessagePair() {
    if (gatewayChatHistory.length >= 2) {
        const removed = gatewayChatHistory.splice(0, 2);
        addLog(tx('Trimmed the oldest turn: “{text}…”', { text: removed[0].content.substring(0, 30) }), 'info');
        return true;
    }
    return false;
}

async function summarizeOldestTurns() {
    if (isSummarizing) return false;
    if (gatewayChatHistory.length < 4) {
        return false;
    }
    isSummarizing = true;
    
    UI.chatSendBtn.disabled = true;
    addLog(tx('Summarising older messages…'), 'system');
    
    // Summarize all except the last 2 messages (the most recent user/assistant pair)
    const messagesToSummarize = gatewayChatHistory.slice(0, -2);
    const remainingMessages = gatewayChatHistory.slice(-2);
    
    let textToSummarize = "";
    messagesToSummarize.forEach(m => {
        const roleName = m.role === 'user' ? 'User' : 'Assistant';
        let textVal = "";
        if (typeof m.content === 'string') {
            textVal = m.content;
        } else if (Array.isArray(m.content)) {
            m.content.forEach(part => {
                if (part.type === 'text' && part.text) {
                    textVal += part.text + " ";
                }
            });
        }
        textToSummarize += `${roleName}: ${textVal.trim()}\n`;
    });
    
    const summaryPrompt = `Summarise the following part of this conversation briefly, in at most 2-3 sentences, in the conversation's own language. Keep all important facts:\n\n${textToSummarize}\n\nSummary:`;
    
    const modelName = UI.gatewayModelSelect.value || null;
    
    try {
        // the standard OpenAI call, so this works with a plain Ollama / LM Studio too
        const response = await api('/v1/chat/completions', 'POST', {
            ...(modelName ? { model: modelName } : {}),
            messages: [{ role: "user", content: summaryPrompt }],
            temperature: 0.3,
            max_tokens: 256
        });
        
        const summaryText = (response.choices?.[0]?.message?.content || '').trim();
        addLog(tx('Summarised: “{text}…”', { text: summaryText.substring(0, 40) }), 'info');
        
        // Rebuild history
        gatewayChatHistory = [
            { 
                role: "system", 
                content: `[Summary of the earlier conversation: ${summaryText}]`,
                meta: { systemSummary: true }
            },
            ...remainingMessages
        ];
        
        reRenderChat();
        updateTokenEstimate();
        return true;
    } catch (err) {
        addLog(tx('Summary failed: {err}', { err: err.message }), 'error');
        return false;
    } finally {
        isSummarizing = false;
        UI.chatSendBtn.disabled = false;
    }
}

function reRenderChat() {
    UI.chatMessages.innerHTML = `
        <div class="chat-system-message">
            ${tx('🤖 Write a message below – the chosen model is loaded automatically the first time.')}
        </div>
    `;
    gatewayChatHistory.forEach(msg => {
        appendChatMessage(msg.role, msg.content, msg.meta || {});
    });
}

async function sendGatewayChatMessage() {
    const userText = UI.chatUserInput.value.trim();
    if (!userText && !currentMultimodalBase64) return;

    // Build payload config
    const systemPrompt = UI.gatewaySystemPrompt.value.trim();
    const modelName = UI.gatewayModelSelect.value || null;
    const thinking = UI.gatewayThinking.checked;
    const temp = parseFloat(UI.gatewayTemp.value) || 0.7;
    const maxTokens = parseInt(UI.gatewayMaxTokens.value) || 2048;
    const retries = parseInt(UI.gatewayRetries.value) || 3;
    const keepalive = parseInt(UI.gatewayKeepalive.value) || 300;
    
    let responseFormat = null;
    const rawSchema = UI.gatewaySchema.value.trim();
    if (rawSchema) {
        try {
            responseFormat = JSON.parse(rawSchema);
        } catch (err) {
            return addLog(tx('The schema is not valid JSON: {err}', { err: err.message }), 'error');
        }
    }

    // Deactivate Inputs
    UI.chatSendBtn.disabled = true;
    UI.chatUserInput.disabled = true;

    // Append User message to UI chat log
    // Append User message to UI chat log
    const userMeta = { filename: currentMultimodalFilename, format: currentMultimodalFormat };
    appendChatMessage('user', userText, userMeta);
    UI.chatUserInput.value = "";

    // Build user message content block
    let userMessage;
    if (currentMultimodalBase64) {
        if (currentMultimodalFormat === 'image') {
            userMessage = {
                role: "user",
                content: [
                    { type: "image_url", image_url: { url: `data:image/jpeg;base64,${currentMultimodalBase64}` } },
                    { type: "text", text: userText || "Describe this image." }
                ],
                meta: userMeta
            };
        } else if (currentMultimodalFormat === 'audio') {
            userMessage = {
                role: "user",
                content: [
                    { type: "input_audio", input_audio: { data: currentMultimodalBase64, format: "wav" } },
                    { type: "text", text: userText || "Transcribe this audio." }
                ],
                meta: userMeta
            };
        }
    } else {
        userMessage = { role: "user", content: userText, meta: userMeta };
    }

    // Add user message to history
    gatewayChatHistory.push(userMessage);

    // Clear file selection visual state
    clearGatewayFile();

    addLog(tx('Sending the request to the gateway…'), 'system');
    
    // Auto-pruning check before sending (if exceeded limit)
    if (autoPruneEnabled) {
        let attemptsLeft = 10; // prevent infinite loop
        while (attemptsLeft > 0) {
            const systemPromptText = systemPrompt.trim();
            let textSum = systemPromptText ? (systemPromptText + "\n") : "";
            gatewayChatHistory.forEach(msg => {
                if (typeof msg.content === 'string') {
                    textSum += msg.content + "\n";
                } else if (Array.isArray(msg.content)) {
                    msg.content.forEach(part => {
                        if (part.type === 'text' && part.text) {
                            textSum += part.text + "\n";
                        }
                    });
                }
            });
            const est = estimateTokens(textSum);
            const limit = getActiveModelLimit();
            if (est >= limit && gatewayChatHistory.length >= 3) { // keep at least current message (User) and 2 older ones
                // remove oldest pair from history (elements 0 and 1)
                const removed = gatewayChatHistory.splice(0, 2);
                addLog(tx('Trimmed the oldest turn before sending: “{text}…”', { text: removed[0].content.substring(0, 30) }), 'info');
                reRenderChat();
            } else {
                break;
            }
            attemptsLeft--;
        }
    } else if (autoSummarizeEnabled) {
        const systemPromptText = systemPrompt.trim();
        let textSum = systemPromptText ? (systemPromptText + "\n") : "";
        gatewayChatHistory.forEach(msg => {
            if (typeof msg.content === 'string') {
                textSum += msg.content + "\n";
            } else if (Array.isArray(msg.content)) {
                msg.content.forEach(part => {
                    if (part.type === 'text' && part.text) {
                        textSum += part.text + "\n";
                    }
                });
            }
        });
        const est = estimateTokens(textSum);
        const limit = getActiveModelLimit();
        if (est >= limit) {
            await summarizeOldestTurns();
        }
    }

    // Use SSE streaming when no schema validation or multimodal post-processing is needed.
    const useStreaming = !responseFormat && !currentMultimodalBase64;

    try {
        if (useStreaming) {
            // Inject system prompt client-side for the streaming path.
            const streamMessages = gatewayChatHistory.map(m => ({ role: m.role, content: m.content }));
            if (systemPrompt && !streamMessages.some(m => m.role === 'system')) {
                streamMessages.unshift({ role: 'system', content: systemPrompt });
            }

            // Create a streaming placeholder bubble immediately.
            const streamDiv = document.createElement('div');
            streamDiv.className = 'chat-message assistant';
            const streamBubble = document.createElement('div');
            streamBubble.className = 'message-bubble';
            streamBubble.textContent = '…';
            streamDiv.appendChild(streamBubble);
            UI.chatMessages.appendChild(streamDiv);
            UI.chatMessages.scrollTop = UI.chatMessages.scrollHeight;

            const streamStart = performance.now();
            const fetchResp = await swFetch('/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: modelName || 'default',
                    messages: streamMessages,
                    temperature: temp,
                    max_tokens: maxTokens,
                    stream: true,
                    thinking: thinking,
                    keep_alive: keepalive
                })
            });

            if (!fetchResp.ok) {
                const errText = await fetchResp.text();
                UI.chatMessages.removeChild(streamDiv);
                throw new Error(`HTTP ${fetchResp.status}: ${errText}`);
            }

            const reader = fetchResp.body.getReader();
            const decoder = new TextDecoder();
            let fullContent = '';
            let fullReasoning = '';   // "thinking" models stream it separately; shown folded under the answer
            let sseBuffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                sseBuffer += decoder.decode(value, { stream: true });
                const lines = sseBuffer.split('\n');
                sseBuffer = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const payload = line.slice(6).trim();
                    if (payload === '[DONE]') continue;
                    try {
                        const chunk = JSON.parse(payload);
                        fullReasoning += chunk.choices?.[0]?.delta?.reasoning_content || '';
                        const delta = chunk.choices?.[0]?.delta?.content || '';
                        if (!fullContent && !delta && fullReasoning) streamBubble.textContent = tx('… thinking …');
                        if (delta) {
                            fullContent += delta;
                            streamBubble.textContent = fullContent;
                            UI.chatMessages.scrollTop = UI.chatMessages.scrollHeight;
                        }
                    } catch (_) {}
                }
            }

            // Replace streaming placeholder with the properly formatted final bubble.
            UI.chatMessages.removeChild(streamDiv);
            const assistantMeta = { model: modelName || 'auto',
                                    runtime: ((performance.now() - streamStart) / 1000).toFixed(1) };
            if (fullReasoning.trim()) assistantMeta.thinking = fullReasoning.trim();
            if (!fullContent.trim()) fullContent = fullReasoning.trim() ? tx('(Only reasoning, no answer – maybe raise “Max. tokens”.)') : tx('(Empty answer)');
            appendChatMessage('assistant', fullContent, assistantMeta);
            gatewayChatHistory.push({ role: 'assistant', content: fullContent, meta: assistantMeta });
            addLog(tx('Answer received.'), 'info');
        } else {
            const response = await api('/api/chat/completions', 'POST', {
                model_name: modelName,
                messages: gatewayChatHistory.map(m => ({ role: m.role, content: m.content })),
                system_prompt: systemPrompt || null,
                temperature: temp,
                max_tokens: maxTokens,
                thinking: thinking,
                keep_alive: keepalive,
                response_format: responseFormat,
                retries: retries
            });

            const assistantMeta = {
                thinking: response.thinking,
                model: response.model_used,
                tps: response.tps,
                runtime: response.runtime_seconds,
                tokens: response.tokens
            };

            appendChatMessage('assistant', response.content, assistantMeta);
            gatewayChatHistory.push({ role: 'assistant', content: response.content, meta: assistantMeta });
            addLog(tx('Answer received ({tps} tokens/s).', { tps: response.tps }), 'info');
        }
    } catch (err) {
        // Remove the failed user message from history to prevent corrupting future attempts
        gatewayChatHistory.pop();
        addLog(tx('Chat failed: {err}', { err: err.message }), 'error');
        appendChatMessage('assistant', '⚠️ ' + tx('Error: {err}', { err: err.message }), { error: true });
    } finally {
        UI.chatSendBtn.disabled = false;
        UI.chatUserInput.disabled = false;
        UI.chatUserInput.focus();
        updateTokenEstimate();
    }
}

function appendChatMessage(sender, text, meta = {}) {
    const msgDiv = document.createElement('div');
    if (sender === 'system' || meta.systemSummary) {
        msgDiv.className = 'chat-system-message-container';
        msgDiv.style.width = '100%';
        msgDiv.style.display = 'flex';
        msgDiv.style.justifyContent = 'center';
        msgDiv.style.margin = '0.5rem 0';
        msgDiv.innerHTML = `<div class="chat-system-message" style="background: rgba(148, 163, 184, 0.08); border: 1px dashed var(--border); border-radius: 8px; padding: 0.5rem 1rem; max-width: 90%; font-size: 0.8rem; color: var(--text-dim); text-align: left; line-height: 1.4;">📝 <em>${text}</em></div>`;
        UI.chatMessages.appendChild(msgDiv);
        UI.chatMessages.scrollTop = UI.chatMessages.scrollHeight;
        return;
    }

    msgDiv.className = `chat-message ${sender}`;

    let innerHTML = '';
    
    // Render multimodal preview inside bubble if sent by user
    if (sender === 'user' && meta.filename) {
        innerHTML += `
            <div style="font-size: 0.75rem; background: rgba(0,0,0,0.1); padding: 0.35rem 0.6rem; border-radius: 6px; margin-bottom: 0.4rem; font-weight: 600;">
                📎 ${meta.format === 'image' ? tx('Image attached: {name}', { name: meta.filename }) : tx('Audio attached: {name}', { name: meta.filename })}
            </div>
        `;
    }

    // Render reasoning blocks if provided by assistant
    if (sender === 'assistant' && meta.thinking) {
        innerHTML += `
            <details class="thinking-block">
                <summary>Gedankengang</summary>
                <div class="thinking-content">${meta.thinking}</div>
            </details>
        `;
    }

    innerHTML += `<div class="message-bubble">${text}</div>`;

    // Metadaten
    if (sender === 'assistant' && !meta.error) {
        // streamed answers carry no speed/time/token figures -- show only what is known
        const stats = [
            meta.model ? `<span>Model: ${meta.model}</span>` : '',
            meta.tps != null ? `<span>Speed: ${meta.tps} TPS</span>` : '',
            meta.runtime != null ? `<span>Time: ${meta.runtime}s</span>` : '',
            meta.tokens != null ? `<span>Tokens: ${meta.tokens}</span>` : '',
        ].join('');
        if (stats) innerHTML += `<div class="message-metadata">${stats}</div>`;
    }

    msgDiv.innerHTML = innerHTML;
    UI.chatMessages.appendChild(msgDiv);
    UI.chatMessages.scrollTop = UI.chatMessages.scrollHeight;
}


// ----------------------------------------------------
// Models, voices, LoRAs
// ----------------------------------------------------
async function loadModels() {
    // Chat models: everything the gateway can reach (this machine, devices, other model servers) -- static/models.js.
    // With a standard OpenAI server instead of ai-serve (Ollama, LM Studio) the Spielwiese's server rebuilds this list
    // from /v1/models, and there is no "Auto": such a server needs a model name.
    try {
        lastApiData.backend = await api('/sw/backend').catch(() => ({ kind: 'ai-serve' }));
        const data = await api('/v1/llm/models');
        lastApiData.llm = data.models || [];
        const keep = UI.gatewayModelSelect.value;
        UI.gatewayModelSelect.innerHTML = lastApiData.backend.kind === 'openai' ? '' : '<option value="">Auto (Gemma 4 12B / aktiv)</option>';
        for (const g of ChatModels.groups(lastApiData.llm)) {
            const og = document.createElement('optgroup');
            og.label = g.label;
            for (const o of g.options) {
                const opt = document.createElement('option');
                opt.value = o.value; opt.textContent = o.text; opt.disabled = o.disabled;
                og.append(opt);
            }
            UI.gatewayModelSelect.append(og);
        }
        if ([...UI.gatewayModelSelect.options].some(o => o.value === keep)) UI.gatewayModelSelect.value = keep;
        updateChatModelHints();
        if (typeof populateOcrModels === 'function') {
            populateOcrModels(lastApiData.llm.filter(m => m.kind === 'chat' && m.vision).map(m => ({ name: m.id })));
        }
    } catch (err) {
        addLog(tx('Couldn\'t load models: {err}', { err: err.message }), 'error');
    }
    try {
        const loraData = await api('/api/comfyui/loras');
        lastApiData.loras = loraData;
        UI.comfyLora.innerHTML = `<option value="">${tx('none')}</option>` +
            (loraData.loras || []).map(l => {
                const tooltipParts = [];
                if (l.trigger_words) tooltipParts.push(`Trigger: ${l.trigger_words}`);
                if (l.description) tooltipParts.push(l.description);
                if (l.base_model) tooltipParts.push(tx('Base model: {m}', { m: l.base_model }));
                const title = tooltipParts.length ? ` title="${tooltipParts.join(' — ').replace(/"/g, '&quot;')}"` : '';
                return `<option value="${l.name}"${title}>${l.name}</option>`;
            }).join('');
    } catch (err) {
        console.error('loras', err);
    }
}

// What runs on the gateway (for the chat's context limits) and whether it answers.
async function pollSystemStatus() {
    if (lastApiData.backend && lastApiData.backend.kind === 'openai') {   // a standard server: no /api/status
        try {
            lastApiData.backend = await api('/sw/backend');
            UI.statusBackend.className = `status-indicator ${lastApiData.backend.kind === 'down' ? 'offline' : 'online'}`;
        } catch { UI.statusBackend.className = 'status-indicator offline'; }
        UI.statusComfyui.className = 'status-indicator offline';
        return;
    }
    try {
        lastApiData.status = await api('/api/status');
        UI.statusBackend.className = 'status-indicator online';
        const comfy = (lastApiData.status.instances || []).some(i => i.port === 8189);
        UI.statusComfyui.className = `status-indicator ${comfy ? 'online' : 'offline'}`;
        updateTokenEstimate();
    } catch (err) {
        UI.statusBackend.className = 'status-indicator offline';
        UI.statusComfyui.className = 'status-indicator offline';
    }
}

(async function init() {
    let tab = null;
    try { tab = localStorage.getItem('spielwiese-tab'); } catch { /* private mode */ }
    const link = tab && document.querySelector(`.tab-link[data-tab="${tab}"]`);
    if (link) link.click();
    initSchemaHelper();
    await loadModels();
    await pollSystemStatus();
    setInterval(pollSystemStatus, 5000);
    const dash = document.getElementById('dashboard-link');
    // the dashboard answers plain http on port 8030 of the same machine -- also when this
    // page came through a Tailscale https URL (https://<machine>.ts.net:<port>)
    if (dash) dash.href = `http://${location.hostname}:8030/`;
})();

// ----------------------------------------------------
// Image Gen (ComfyUI) Logic
// ----------------------------------------------------
let comfyMode = 'generate';
let comfyEditImageBase64 = null;
let comfyStyleContentBase64 = null;
let comfyStyleRefBase64 = null;

const COMFY_GENERATE_SAMPLE_PROMPT = 'cute cat playing with a ball of yarn, highly detailed';

function setComfyMode(mode) {
    const prevMode = comfyMode;
    comfyMode = mode;
    UI.comfyModeGenerateBtn.classList.toggle('active', mode === 'generate');
    UI.comfyModeEditBtn.classList.toggle('active', mode === 'edit');
    UI.comfyModeStyleBtn.classList.toggle('active', mode === 'style');
    UI.comfyModeBatchBtn.classList.toggle('active', mode === 'batch');
    UI.comfyGenerateOnlyFields.style.display = mode === 'generate' ? 'block' : 'none';
    UI.comfyEditFields.style.display = mode === 'edit' ? 'block' : 'none';
    UI.comfyStyleFields.style.display = mode === 'style' ? 'block' : 'none';
    UI.comfyBatchFields.style.display = mode === 'batch' ? 'block' : 'none';
    UI.comfyNegativePromptGroup.style.display = mode === 'generate' ? 'block' : 'none';
    if (UI.comfyDenoiseGroup) UI.comfyDenoiseGroup.style.display = (mode === 'style' || mode === 'edit') ? 'block' : 'none';
    // Batch mode has its own start button + its own progress/results panel — the
    // single-shot config grid, prompt card, and generate button all apply to exactly
    // one image at a time and don't make sense once a whole JSON list is driving it.
    UI.comfySingleConfigGrid.style.display = mode === 'batch' ? 'none' : 'grid';
    UI.comfySingleFilenameGroup.style.display = mode === 'batch' ? 'none' : 'block';
    UI.comfyGenerateBtn.style.display = mode === 'batch' ? 'none' : 'inline-flex';
    UI.comfySingleResults.style.display = mode === 'batch' ? 'none' : 'block';
    UI.comfyBatchResults.style.display = mode === 'batch' ? 'block' : 'none';
    UI.comfyPromptLabel.textContent = mode === 'generate' ? tx('What should be in the picture?') : mode === 'edit' ? tx('What should change?') : tx('How should the style be transferred?');
    UI.comfyGenerateBtn.textContent = mode === 'generate' ? tx('Generate image') : mode === 'edit' ? tx('Edit image') : tx('Transfer style');

    // The prompt box is shared across modes — leaving the Generate-mode sample text in
    // place would otherwise get silently submitted as an edit/style instruction.
    if (prevMode === 'generate' && mode !== 'generate' && UI.comfyPrompt.value.trim() === COMFY_GENERATE_SAMPLE_PROMPT) {
        UI.comfyPrompt.value = '';
    } else if (mode === 'generate' && prevMode !== 'generate' && UI.comfyPrompt.value.trim() === '') {
        UI.comfyPrompt.value = COMFY_GENERATE_SAMPLE_PROMPT;
    }

    if (mode === 'batch') {
        pollComfyBatchStatus();
    }
}

UI.comfyLora.addEventListener('change', () => {
    UI.comfyLoraStrengthGroup.style.display = UI.comfyLora.value ? 'block' : 'none';
    // Pre-fill the strength field from the selected LoRA's own recommended_strength
    // (its .json sidecar), if it has one -- still just a starting point, not locked.
    const selected = (lastApiData.loras?.loras || []).find(l => l.name === UI.comfyLora.value);
    if (selected?.recommended_strength !== undefined && selected.recommended_strength !== null) {
        UI.comfyLoraStrength.value = selected.recommended_strength;
    }
});

UI.comfyModeGenerateBtn.addEventListener('click', () => setComfyMode('generate'));
UI.comfyModeEditBtn.addEventListener('click', () => setComfyMode('edit'));
UI.comfyModeStyleBtn.addEventListener('click', () => setComfyMode('style'));

// Drag & Drop / File select for the image-to-edit
UI.comfyEditUploadZone.addEventListener('click', () => UI.comfyEditInput.click());
UI.comfyEditInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
        comfyEditImageBase64 = evt.target.result.split(',')[1];
        UI.comfyEditPlaceholder.style.display = 'none';
        UI.comfyEditPreviewContainer.style.display = 'block';
        UI.comfyEditImgPreview.src = evt.target.result;
        addLog(tx('Image to edit: {name}', { name: file.name }), 'system');
    };
    reader.readAsDataURL(file);
});

UI.comfyEditRemoveBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    comfyEditImageBase64 = null;
    UI.comfyEditInput.value = "";
    UI.comfyEditPlaceholder.style.display = 'block';
    UI.comfyEditPreviewContainer.style.display = 'none';
    UI.comfyEditImgPreview.src = "";
});

// Drag & Drop / File select for the style-transfer content and reference images
function wireImageUploadZone(zone, input, previewContainer, imgPreview, removeBtn, placeholder, onLoaded, onCleared) {
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            onLoaded(evt.target.result.split(',')[1]);
            placeholder.style.display = 'none';
            previewContainer.style.display = 'block';
            imgPreview.src = evt.target.result;
        };
        reader.readAsDataURL(file);
    });
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        onCleared();
        input.value = "";
        placeholder.style.display = 'block';
        previewContainer.style.display = 'none';
        imgPreview.src = "";
    });
}

wireImageUploadZone(
    UI.comfyStyleContentUploadZone, UI.comfyStyleContentInput, UI.comfyStyleContentPreviewContainer,
    UI.comfyStyleContentImgPreview, UI.comfyStyleContentRemoveBtn, UI.comfyStyleContentPlaceholder,
    (b64) => { comfyStyleContentBase64 = b64; }, () => { comfyStyleContentBase64 = null; }
);
wireImageUploadZone(
    UI.comfyStyleRefUploadZone, UI.comfyStyleRefInput, UI.comfyStyleRefPreviewContainer,
    UI.comfyStyleRefImgPreview, UI.comfyStyleRefRemoveBtn, UI.comfyStyleRefPlaceholder,
    (b64) => { comfyStyleRefBase64 = b64; }, () => { comfyStyleRefBase64 = null; }
);

UI.comfyGenerateBtn.addEventListener('click', () => {
    if (comfyMode === 'edit') {
        startImageEdit();
    } else if (comfyMode === 'style') {
        startStyleTransfer();
    } else {
        startImageGeneration();
    }
});

async function startStyleTransfer() {
    const prompt = UI.comfyPrompt.value.trim();
    if (!comfyStyleContentBase64) return addLog(tx('Please attach a content image.'), 'warn');
    if (!comfyStyleRefBase64) return addLog(tx('Please attach a style reference.'), 'warn');

    UI.comfyGenerateBtn.disabled = true;
    UI.comfyLoading.style.display = 'flex';
    UI.imagePlaceholderBox.style.display = 'none';
    UI.comfyResultImg.style.display = 'none';
    UI.comfyProgressPanel.style.display = 'block';

    addLog(tx('Starting the style transfer…'), 'system');

    try {
        const payload = {
            content_image_base64: comfyStyleContentBase64,
            style_image_base64: comfyStyleRefBase64,
            model_name: UI.comfyStyleModel.value,
            steps: UI.comfySteps.value ? parseInt(UI.comfySteps.value) : null,
            seed: UI.comfySeed.value ? parseInt(UI.comfySeed.value) : null,
            denoise: UI.comfyDenoise.value ? parseFloat(UI.comfyDenoise.value) : null,
            filename: UI.comfyFilename.value.trim() || null
        };
        if (prompt) payload.prompt = prompt;
        const resp = await api('/api/comfyui/style-transfer', 'POST', payload);

        addLog(tx('Task queued: {id}. Following its progress…', { id: resp.task_id }), 'system');

        if (comfyPollInterval) clearInterval(comfyPollInterval);
        comfyPollInterval = setInterval(() => pollComfyTask(resp.task_id), 1500);
    } catch (err) {
        addLog(tx('Style transfer failed: {err}', { err: err.message }), 'error');
        UI.comfyGenerateBtn.disabled = false;
        UI.comfyLoading.style.display = 'none';
        UI.imagePlaceholderBox.style.display = 'block';
        UI.comfyProgressPanel.style.display = 'none';
    }
}

async function startImageEdit() {
    const prompt = UI.comfyPrompt.value.trim();
    if (!prompt) return addLog(tx('Please describe what should change.'), 'warn');
    if (!comfyEditImageBase64) return addLog(tx('Please attach an image to edit.'), 'warn');

    UI.comfyGenerateBtn.disabled = true;
    UI.comfyLoading.style.display = 'flex';
    UI.imagePlaceholderBox.style.display = 'none';
    UI.comfyResultImg.style.display = 'none';
    UI.comfyProgressPanel.style.display = 'block';

    addLog(tx('Editing started …'), 'system');

    try {
        const resp = await api('/api/comfyui/edit', 'POST', {
            image_base64: comfyEditImageBase64,
            prompt,
            model_name: UI.comfyEditModel.value,
            steps: UI.comfySteps.value ? parseInt(UI.comfySteps.value) : null,
            seed: UI.comfySeed.value ? parseInt(UI.comfySeed.value) : null,
            denoise: UI.comfyDenoise.value ? parseFloat(UI.comfyDenoise.value) : null,
            filename: UI.comfyFilename.value.trim() || null
        });

        addLog(tx('Task queued: {id}. Following its progress…', { id: resp.task_id }), 'system');

        if (comfyPollInterval) clearInterval(comfyPollInterval);
        comfyPollInterval = setInterval(() => pollComfyTask(resp.task_id), 1500);
    } catch (err) {
        addLog(tx('Editing failed: {err}', { err: err.message }), 'error');
        UI.comfyGenerateBtn.disabled = false;
        UI.comfyLoading.style.display = 'none';
        UI.imagePlaceholderBox.style.display = 'block';
        UI.comfyProgressPanel.style.display = 'none';
    }
}

async function startImageGeneration() {
    const prompt = UI.comfyPrompt.value.trim();
    if (!prompt) return addLog(tx('Please describe what should be in the picture.'), 'warn');
    
    UI.comfyGenerateBtn.disabled = true;
    UI.comfyLoading.style.display = 'flex';
    UI.imagePlaceholderBox.style.display = 'none';
    UI.comfyResultImg.style.display = 'none';
    UI.comfyProgressPanel.style.display = 'block';
    
    const gatewayOpt = UI.comfyModel.selectedOptions[0];
    if (gatewayOpt && gatewayOpt.dataset.gateway) return generateViaGateway(prompt, gatewayOpt.value, gatewayOpt.textContent);
    // kept in "Meine Bilder" once the ComfyUI task is done (pollComfyTask)
    pendingGalleryMeta = { prompt, model: UI.comfyModel.value, label: (UI.comfyModel.selectedOptions[0] || {}).textContent || UI.comfyModel.value,
                           steps: UI.comfySteps.value ? parseInt(UI.comfySteps.value) : null,
                           seed: UI.comfySeed.value ? parseInt(UI.comfySeed.value) : null,
                           aspect_ratio: UI.comfyAspectRatio.value, source: 'bild' };
    addLog(tx('Generating the image …'), 'system');
    
    try {
        const resp = await api('/api/comfyui/generate', 'POST', {
            prompt,
            negative_prompt: UI.comfyNegativePrompt.value.trim(),
            model_name: UI.comfyModel.value,
            aspect_ratio: UI.comfyAspectRatio.value,
            base_resolution: parseInt(UI.comfyResolution.value) || 1024,
            steps: UI.comfySteps.value ? parseInt(UI.comfySteps.value) : null,
            seed: UI.comfySeed.value ? parseInt(UI.comfySeed.value) : null,
            lora_name: UI.comfyLora.value || null,
            lora_strength: UI.comfyLora.value ? parseFloat(UI.comfyLoraStrength.value) : null,
            filename: UI.comfyFilename.value.trim() || null
        });
        
        addLog(tx('Task queued: {id}. Following its progress…', { id: resp.task_id }), 'system');
        
        if (comfyPollInterval) clearInterval(comfyPollInterval);
        comfyPollInterval = setInterval(() => pollComfyTask(resp.task_id), 1500);
    } catch (err) {
        addLog(tx('Generating failed: {err}', { err: err.message }), 'error');
        UI.comfyGenerateBtn.disabled = false;
        UI.comfyLoading.style.display = 'none';
        UI.imagePlaceholderBox.style.display = 'block';
        UI.comfyProgressPanel.style.display = 'none';
    }
}

// All three model lists of the Bild tab come from the gateway's catalog (GET /v1/images/models, by task), so
// a new or removed model shows up here without a change (owner, 2026-10-20). Main-ComfyUI models keep their
// checkpoint filename as value (they go through /api/comfyui/generate with LoRA, resolution, negative
// prompt); the others (MOSS-Image, Iris, ...) their catalog id and answer on /v1/images/generations.
// Edit and style models: catalog ids, which /api/comfyui/edit|style-transfer accept.
function fillImageModelSelect(select, rows, valueOf, noteEl) {
    const keep = select.value;
    select.replaceChildren();
    for (const m of rows) {
        const o = document.createElement('option');
        o.value = valueOf(m);
        if (m.provider !== 'comfyui' && select === UI.comfyModel) o.dataset.gateway = '1';
        o.textContent = m.available ? m.label : tx('{label} (not installed)', { label: m.label });
        o.title = m.description ? tx(m.description) : '';
        o.disabled = !m.available;
        o.dataset.description = m.description ? tx(m.description) : '';
        select.append(o);
    }
    const def = rows.find(m => m.default && m.available);
    const options = [...select.options];
    // nothing usable: still show what exists (greyed, "not installed") instead of an empty field
    select.value = options.some(o => o.value === keep && !o.disabled) ? keep
        : (def ? valueOf(def) : (options.find(o => !o.disabled) || options[0] || {}).value);
    const showNote = () => { if (noteEl) noteEl.textContent = (select.selectedOptions[0] || {}).dataset?.description || ''; };
    select.onchange = showNote;
    showNote();
}

async function loadImageModels() {
    const comfyFile = m => (m.aliases || []).slice(-1)[0];   // the checkpoint filename (last alias)
    try {
        const [gen, edit, style] = await Promise.all(['generate', 'edit', 'style'].map(t => api(`/v1/images/models?task=${t}`)));
        fillImageModelSelect(UI.comfyModel, gen.data || [], m => m.provider === 'comfyui' ? comfyFile(m) : m.id);
        fillImageModelSelect(UI.comfyEditModel, edit.data || [], m => m.id, document.getElementById('comfy-edit-model-note'));
        fillImageModelSelect(UI.comfyStyleModel, style.data || [], m => m.id, document.getElementById('comfy-style-model-note'));
        UI.comfyModel.dispatchEvent(new Event('change'));
        // missing models: what can be downloaded, and a plain word when an ability isn't there at all
        const usable = rows => (rows || []).some(m => m.available);
        Downloads.panel(document.getElementById('dl-image-gen'), 'image',
            usable(gen.data) ? '' : tx('No image model is installed yet.'));
        Downloads.panel(document.getElementById('dl-image-edit'), false, usable(edit.data) ? '' :
            tx('No model on this gateway can edit images. That needs an image plugin with editing (e.g. ComfyUI with Qwen-Image-Edit) – it isn\'t part of this installation (docs/plugins.md).'));
        Downloads.panel(document.getElementById('dl-image-style'), false, usable(style.data) ? '' :
            tx('No model on this gateway can transfer a style. That needs an image plugin with style transfer (e.g. ComfyUI with Qwen-Image-Edit 2511) – it isn\'t part of this installation (docs/plugins.md).'));
    } catch (err) {
        addLog(tx('Couldn\'t load image models: {err}', { err: err.message }), 'warn');
    }
}

async function generateViaGateway(prompt, model, label) {
    addLog(tx('{label}: generating … (the first image loads the model)', { label }), 'system');
    const t0 = Date.now();
    try {
        const body = { model, prompt, aspect_ratio: UI.comfyAspectRatio.value, response_format: 'b64_json' };
        if (UI.comfySeed.value) body.seed = parseInt(UI.comfySeed.value);
        const data = await api('/v1/images/generations', 'POST', body);
        UI.comfyResultImg.src = `data:image/png;base64,${data.data[0].b64_json}`;
        UI.comfyResultImg.style.display = 'inline-block';
        addLog(tx('{label}: done after {s} s.', { label, s: Math.round((Date.now() - t0) / 100) / 10 }), 'success');
        if (typeof saveToGallery === 'function') {
            saveToGallery(data.data[0].b64_json, { prompt, model, label, seed: body.seed, aspect_ratio: body.aspect_ratio,
                                                   seconds: (Date.now() - t0) / 1000, source: 'bild' });
        }
    } catch (err) {
        addLog(tx('{label} failed: {err}', { label, err: err.message }), 'error');
        UI.imagePlaceholderBox.style.display = 'block';
        UI.imagePlaceholderBox.textContent = '❌ ' + tx('{label} failed: {err}', { label, err: err.message });
    } finally {
        UI.comfyGenerateBtn.disabled = false;
        UI.comfyLoading.style.display = 'none';
        UI.comfyProgressPanel.style.display = 'none';
    }
}

loadImageModels();
window.addEventListener('sw-installed', (e) => { if (e.detail.capability === 'image') loadImageModels(); });

let pendingGalleryMeta = null;   // set by a Bild "Generate", consumed when its task completes

async function pollComfyTask(taskId) {
    try {
        const data = await api(`/api/comfyui/task/${taskId}`);
        lastApiData.comfyTask = data;
        // (the API explorer stayed in the dashboard)
        
        if (data.status === 'processing') {
            const prog = data.progress;
            UI.comfyProgressNode.textContent = prog.node ? `Node: ${prog.node}` : tx('Generating…');
            UI.comfyProgressPercentage.textContent = `${prog.value}%`;
            UI.comfyProgressFill.style.width = `${prog.value}%`;
        } else if (data.status === 'completed') {
            clearInterval(comfyPollInterval);
            addLog(tx('Done after {s} s.', { s: data.runtime_seconds }), 'success');
            
            UI.comfyResultImg.src = data.image_base64.startsWith('data:') ? data.image_base64 : `data:image/png;base64,${data.image_base64}`;
            UI.comfyResultImg.style.display = 'inline-block';
            if (pendingGalleryMeta && typeof saveToGallery === 'function') {
                saveToGallery(data.image_base64, { ...pendingGalleryMeta, seconds: data.runtime_seconds });
            }
            pendingGalleryMeta = null;
            UI.comfyLoading.style.display = 'none';
            UI.comfyGenerateBtn.disabled = false;
            UI.comfyProgressPanel.style.display = 'none';
        } else if (data.status === 'failed') {
            clearInterval(comfyPollInterval);
            addLog(tx('Failed: {err}', { err: data.error }), 'error');
            UI.comfyLoading.style.display = 'none';
            UI.comfyGenerateBtn.disabled = false;
            UI.imagePlaceholderBox.style.display = 'block';
            UI.imagePlaceholderBox.textContent = '❌ ' + tx('Failed: {err}', { err: data.error });
            UI.comfyProgressPanel.style.display = 'none';
        }
    } catch (err) {
        console.error('Comfy polling error', err);
    }
}

// ----------------------------------------------------
// Image Gen — Batch mode (JSON upload -> sequential generation)
// ----------------------------------------------------

UI.comfyModeBatchBtn.addEventListener('click', () => setComfyMode('batch'));

UI.comfyBatchFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
        try {
            // Round-trip through JSON.parse/stringify so a pasted/dropped file that's
            // valid but oddly formatted still renders as clean, readable JSON.
            const parsed = JSON.parse(evt.target.result);
            UI.comfyBatchJson.value = JSON.stringify(parsed, null, 2);
            addLog(tx('Loaded the batch JSON from {name}.', { name: file.name }), 'system');
        } catch (err) {
            addLog(tx('That file isn\'t valid JSON: {err}', { err: err.message }), 'error');
        }
    };
    reader.readAsText(file);
});

UI.comfyBatchStartBtn.addEventListener('click', startComfyBatch);
UI.comfyBatchPauseBtn.addEventListener('click', () => comfyBatchAction('pause'));
UI.comfyBatchResumeBtn.addEventListener('click', () => comfyBatchAction('resume'));
UI.comfyBatchCancelBtn.addEventListener('click', () => comfyBatchAction('cancel'));

async function startComfyBatch() {
    let payload;
    try {
        payload = JSON.parse(UI.comfyBatchJson.value);
    } catch (err) {
        return addLog(tx('The batch JSON is invalid: {err}', { err: err.message }), 'error');
    }

    UI.comfyBatchStartBtn.disabled = true;
    for (const key in comfyBatchImageCache) delete comfyBatchImageCache[key];
    UI.comfyBatchGallery.innerHTML = '';

    try {
        const resp = await api('/api/comfyui/batch', 'POST', payload);
        addLog(tx('Batch started: {n} image(s).', { n: resp.total }), 'system');
        if (comfyBatchPollInterval) clearInterval(comfyBatchPollInterval);
        comfyBatchPollInterval = setInterval(pollComfyBatchStatus, 1500);
        await pollComfyBatchStatus();
    } catch (err) {
        addLog(tx('Couldn\'t start the batch: {err}', { err: err.message }), 'error');
        UI.comfyBatchStartBtn.disabled = false;
    }
}

async function comfyBatchAction(action) {
    try {
        await api(`/api/comfyui/batch/${action}`, 'POST', {});
        addLog(tx('Batch: {action} requested.', { action }), 'system');
        await pollComfyBatchStatus();
    } catch (err) {
        addLog(tx('Batch ({action}) failed: {err}', { action, err: err.message }), 'error');
    }
}

function renderComfyBatchItem(item) {
    let el = document.getElementById(`comfy-batch-item-${item.index}`);
    if (!el) {
        el = document.createElement('div');
        el.id = `comfy-batch-item-${item.index}`;
        el.style.cssText = 'border:1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--card-bg);';
        UI.comfyBatchGallery.appendChild(el);
    }
    const statusBadgeClass = {
        completed: 'success', failed: 'error', cancelled: 'omitted',
        processing: 'badge-generating', pending: 'omitted',
    }[item.status] || 'omitted';
    const img = comfyBatchImageCache[item.task_id];
    el.innerHTML = `
        <div style="aspect-ratio: 1; display: flex; align-items: center; justify-content: center; background: rgba(148,163,184,0.08); ${img ? '' : 'font-size: 1.5rem;'}">
            ${img ? `<img src="${img}" style="width: 100%; height: 100%; object-fit: cover;">` : (item.status === 'failed' ? '❌' : item.status === 'processing' ? '⏳' : '🖼️')}
        </div>
        <div style="padding: 0.5rem;">
            <div style="font-size: 0.75rem; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${item.prompt.replace(/"/g, '&quot;')}">${item.prompt}</div>
            <span class="badge ${statusBadgeClass}" style="font-size: 0.65rem; padding: 0.1rem 0.35rem; margin-top: 0.25rem; display: inline-block;">${item.status}</span>
            ${item.status === 'completed' && item.filename ? `<div style="font-size: 0.7rem; margin-top: 0.25rem;"><a href="${BASE_URL}/api/comfyui/images/${encodeURIComponent(item.filename)}" target="_blank" rel="noopener">${item.filename}</a></div>` : ''}
            ${item.error ? `<div style="font-size: 0.7rem; color: var(--danger); margin-top: 0.25rem;">${item.error}</div>` : ''}
        </div>
    `;
}

async function pollComfyBatchStatus() {
    let data;
    try {
        data = await api('/api/comfyui/batch/status');
    } catch (err) {
        console.error('Batch status poll error', err);
        return;
    }
    lastApiData.comfyBatch = data;
    // (the API explorer stayed in the dashboard)

    if (data.status === 'none') {
        UI.comfyBatchStatusLabel.textContent = tx('Ready');
        UI.comfyBatchStatusLabel.className = 'badge success';
        UI.comfyBatchPauseBtn.style.display = 'none';
        UI.comfyBatchResumeBtn.style.display = 'none';
        UI.comfyBatchCancelBtn.style.display = 'none';
        UI.comfyBatchStartBtn.disabled = false;
        if (comfyBatchPollInterval) { clearInterval(comfyBatchPollInterval); comfyBatchPollInterval = null; }
        return;
    }

    const percent = data.total > 0 ? Math.round((data.completed_count / data.total) * 100) : 0;
    UI.comfyBatchProgressFill.style.width = `${percent}%`;
    UI.comfyBatchProgressText.textContent = `${data.completed_count} / ${data.total}`;
    UI.comfyBatchStatusLabel.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    UI.comfyBatchStatusLabel.className = `badge ${data.status === 'running' ? 'badge-generating' : data.status === 'failed' ? 'error' : data.status === 'cancelled' ? 'omitted' : 'success'}`;

    UI.comfyBatchPauseBtn.style.display = data.status === 'running' ? 'inline-flex' : 'none';
    UI.comfyBatchResumeBtn.style.display = data.status === 'paused' ? 'inline-flex' : 'none';
    UI.comfyBatchCancelBtn.style.display = (data.status === 'running' || data.status === 'paused') ? 'inline-flex' : 'none';
    UI.comfyBatchStartBtn.disabled = data.status === 'running' || data.status === 'paused';

    for (const item of data.items) {
        renderComfyBatchItem(item);
        if (item.status === 'completed' && item.task_id && !comfyBatchImageCache[item.task_id] && !comfyBatchImageFetching.has(item.task_id)) {
            // Fetch the actual image once per completed item -- the status poll
            // itself deliberately omits image_base64 to stay cheap on a large batch.
            comfyBatchImageFetching.add(item.task_id);
            api(`/api/comfyui/task/${item.task_id}`).then((task) => {
                if (task.image_base64) {
                    comfyBatchImageCache[item.task_id] = task.image_base64.startsWith('data:')
                        ? task.image_base64 : `data:image/png;base64,${task.image_base64}`;
                    renderComfyBatchItem(item);
                }
            }).catch((err) => console.error('Batch thumbnail fetch failed', err)).finally(() => {
                comfyBatchImageFetching.delete(item.task_id);
            });
        }
    }

    if (data.status === 'running' || data.status === 'paused') {
        if (!comfyBatchPollInterval) comfyBatchPollInterval = setInterval(pollComfyBatchStatus, 1500);
    } else if (comfyBatchPollInterval) {
        clearInterval(comfyBatchPollInterval);
        comfyBatchPollInterval = null;
    }
}

let activeProvider = 'llamacpp';
let builderFields = [
    { name: "character_name", type: "string", desc: "The character's name", required: true },
    { name: "class", type: "enum", enumOptions: "warrior, mage, rogue", desc: "The character class", required: true },
    { name: "level", type: "integer", desc: "The character's level", required: true },
    { name: "skills", type: "array", desc: "Special abilities", required: false },
    { name: "is_hero", type: "boolean", desc: "Is a main hero", required: false }
];

const presets = {
    character: [
        { name: "character_name", type: "string", desc: "The character's name", required: true },
        { name: "class", type: "enum", enumOptions: "warrior, mage, rogue", desc: "The character class", required: true },
        { name: "level", type: "integer", desc: "The character's level", required: true },
        { name: "skills", type: "array", desc: "Special abilities", required: false },
        { name: "is_hero", type: "boolean", desc: "Is a main hero", required: false }
    ],
    sentiment: [
        { name: "sentiment", type: "enum", enumOptions: "positive, negative, neutral", desc: "The sentiment", required: true },
        { name: "confidence", type: "number", desc: "Confidence (0.0 - 1.0)", required: true },
        { name: "summary", type: "string", desc: "Summary", required: false },
        { name: "keywords", type: "array", desc: "Most important keywords", required: false }
    ],
    simple: [
        { name: "first_name", type: "string", desc: "First name", required: true },
        { name: "last_name", type: "string", desc: "Last name", required: true },
        { name: "age", type: "integer", desc: "Age in years", required: false },
        { name: "email", type: "string", desc: "Email address", required: false }
    ],
    catalog: [
        { name: "product_name", type: "string", desc: "Product name", required: true },
        { name: "price", type: "number", desc: "Price in euros", required: true },
        { name: "category", type: "enum", enumOptions: "electronics, clothing, books, food", desc: "Product category", required: true },
        { name: "in_stock", type: "boolean", desc: "In stock", required: false }
    ]
};

const providerDocs = {
    llamacpp: `
        <div class="doc-header">llama.cpp / OpenAI Structured Outputs</div>
        <p data-i18n>The gateway and llama.cpp support standard OpenAI-compatible JSON schemas through the <code>response_format</code> parameter. The gateway turns the schema into a GBNF grammar at inference time, which guarantees a 100% correct structure.</p>
        
        <div class="doc-section">API payload structure</div>
        <div class="code-snippet">"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "custom_schema",
    "strict": true,
    "schema": { ... }
  }
}</div>

        <div class="doc-section">Examples & presets</div>
        <details class="doc-details" open>
            <summary>🟢 Simple example: user profile</summary>
            <div class="doc-details-content">
                <p>A simple, flat schema with basic data types.</p>
                <div class="code-snippet">{
  "type": "object",
  "properties": {
    "first_name": { "type": "string" },
    "last_name": { "type": "string" },
    "age": { "type": "integer" }
  },
  "required": ["first_name", "last_name"]
}</div>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="simple" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>

        <details class="doc-details">
            <summary>🟡 Medium example: sentiment (with enums)</summary>
            <div class="doc-details-content">
                <p>A schema with a fixed choice (enum) and numeric values.</p>
                <div class="code-snippet">{
  "type": "object",
  "properties": {
    "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral"] },
    "confidence": { "type": "number" },
    "summary": { "type": "string" }
  },
  "required": ["sentiment", "confidence"]
}</div>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="sentiment" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>

        <details class="doc-details">
            <summary>🔴 Complex example: product catalogue</summary>
            <div class="doc-details-content">
                <p>A larger schema with booleans, enums and numbers.</p>
                <div class="code-snippet">{
  "type": "object",
  "properties": {
    "product_name": { "type": "string" },
    "price": { "type": "number" },
    "category": { "type": "string", "enum": ["electronics", "clothing", "books", "food"] },
    "in_stock": { "type": "boolean" }
  },
  "required": ["product_name", "price", "category"]
}</div>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="catalog" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>
    `,
    ollama: `
        <div class="doc-header">Ollama JSON Schema Format</div>
        <p data-i18n>Ollama supports structured output when a JSON schema is passed directly in the <code>format</code> parameter. This is fully built in since Ollama 0.2.7.</p>
        
        <div class="doc-section">API payload structure</div>
        <div class="code-snippet">"format": {
  "type": "object",
  "properties": { ... }
}</div>

        <div class="doc-section">Examples & presets</div>
        <details class="doc-details" open>
            <summary>🟢 Simple example: user profile</summary>
            <div class="doc-details-content">
                <p data-i18n>A simple JSON schema passed directly to Ollama's <code>format</code> field.</p>
                <div class="code-snippet">"format": {
  "type": "object",
  "properties": {
    "first_name": { "type": "string" },
    "last_name": { "type": "string" }
  },
  "required": ["first_name", "last_name"]
}</div>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="simple" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>

        <details class="doc-details">
            <summary>🟡 Medium example: sentiment (with enums)</summary>
            <div class="doc-details-content">
                <p>Ollama enforces the enum values at inference time too.</p>
                <div class="code-snippet">"format": {
  "type": "object",
  "properties": {
    "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral"] }
  }
}</div>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="sentiment" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>

        <details class="doc-details">
            <summary>🔴 Complex example: RPG character</summary>
            <div class="doc-details-content">
                <p>A more complex schema for generating characters.</p>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="character" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>
    `,
    lmstudio: `
        <div class="doc-header">LM Studio Structured Outputs</div>
        <p data-i18n>LM Studio is OpenAI-compatible and expects the schema under <code>response_format.json_schema.schema</code>.</p>
        
        <div class="doc-section">API payload structure</div>
        <div class="code-snippet">"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "schema",
    "schema": { ... }
  }
}</div>

        <div class="doc-section">Examples & presets</div>
        <details class="doc-details" open>
            <summary>🟢 Simple example: user profile</summary>
            <div class="doc-details-content">
                <p>A standard OpenAI-compatible schema for profiles.</p>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="simple" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>

        <details class="doc-details">
            <summary>🟡 Medium example: sentiment (with enums)</summary>
            <div class="doc-details-content">
                <p>A schema for classifications.</p>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="sentiment" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>

        <details class="doc-details">
            <summary>🔴 Complex example: RPG character</summary>
            <div class="doc-details-content">
                <p>A large RPG schema for structured generation.</p>
                <button type="button" class="btn btn-primary btn-load-preset" data-preset-load="character" style="padding: 0.25rem 0.5rem; font-size: 0.7rem; margin-top: 0.25rem;">Load into the builder</button>
            </div>
        </details>
    `
};

function initSchemaHelper() {
    if (!UI.openSchemaHelperBtn) return;

    // Open Modal
    UI.openSchemaHelperBtn.addEventListener('click', () => {
        UI.schemaHelperModal.style.display = 'flex';
        updateProviderDocs();
        loadPreset('character');
        addLog(tx('Opened the JSON schema builder.'), 'system');
    });

    // Close Modal
    UI.closeSchemaHelperBtn.addEventListener('click', () => {
        UI.schemaHelperModal.style.display = 'none';
    });

    // Close on click outside
    UI.schemaHelperModal.addEventListener('click', (e) => {
        if (e.target === UI.schemaHelperModal) {
            UI.schemaHelperModal.style.display = 'none';
        }
    });

    // Provider Tabs
    UI.providerTabLinks.forEach(link => {
        link.addEventListener('click', () => {
            UI.providerTabLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            activeProvider = link.getAttribute('data-provider');
            updateProviderDocs();
            generateSchema();
        });
    });

    // Preset Buttons
    UI.presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetName = btn.getAttribute('data-preset');
            loadPreset(presetName);
        });
    });

    // Delegate "Load in Builder" clicks in documentation box
    if (UI.providerDescContent) {
        UI.providerDescContent.addEventListener('click', (e) => {
            const loadBtn = e.target.closest('.btn-load-preset');
            if (loadBtn) {
                const presetName = loadBtn.getAttribute('data-preset-load');
                loadPreset(presetName);
            }
        });
    }

    // Add Field button
    UI.builderAddFieldBtn.addEventListener('click', () => {
        builderFields.push({ name: `field_${builderFields.length + 1}`, type: "string", desc: "", required: false });
        renderBuilderFields();
        generateSchema();
    });

    // Copy actions
    UI.copySchemaOnlyBtn.addEventListener('click', async () => {
        const code = UI.schemaPreviewCode.textContent;
        logCopyResult(await copyTextToClipboard(code), 'JSON Schema');
    });

    UI.copyPayloadBtn.addEventListener('click', async () => {
        const code = UI.payloadPreviewCode.textContent;
        logCopyResult(await copyTextToClipboard(code), 'API Request Payload cURL');
    });

    // Apply to Gateway Chat
    UI.applySchemaBtn.addEventListener('click', () => {
        const schema = buildJSONSchema();
        UI.gatewaySchema.value = JSON.stringify(schema, null, 2);
        UI.schemaHelperModal.style.display = 'none';
        addLog(tx('Schema taken over into the chat.'), 'success');
    });
}

function loadPreset(presetName) {
    if (presets[presetName]) {
        builderFields = JSON.parse(JSON.stringify(presets[presetName]));
        renderBuilderFields();
        generateSchema();

        UI.presetButtons.forEach(btn => {
            if (btn.getAttribute('data-preset') === presetName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        addLog(tx('Loaded the template “{name}” into the schema builder.', { name: presetName }), 'system');
    }
}

function updateProviderDocs() {
    UI.providerDescContent.innerHTML = providerDocs[activeProvider] || "";
    I18N.apply(UI.providerDescContent);
}

function renderBuilderFields() {
    UI.builderFieldsList.innerHTML = "";
    if (builderFields.length === 0) {
        UI.builderFieldsList.innerHTML = `<div class="empty-state-small" style="padding: 1rem; text-align: center; color: var(--text-muted); font-style: italic;">${tx('No fields yet – “+ Add field” below.')}</div>`;
        return;
    }

    builderFields.forEach((field, index) => {
        const row = document.createElement('div');
        row.className = 'builder-field-row';
        
        const isEnum = field.type === 'enum';

        row.innerHTML = `
            <input type="text" class="field-name-input" value="${field.name}" placeholder="field_name" title="Property Name" style="width: 20%;">
            <select class="field-type-select" style="width: 18%; padding: 0.4rem 0.6rem;">
                <option value="string" ${field.type === 'string' ? 'selected' : ''}>string</option>
                <option value="integer" ${field.type === 'integer' ? 'selected' : ''}>integer</option>
                <option value="number" ${field.type === 'number' ? 'selected' : ''}>number</option>
                <option value="boolean" ${field.type === 'boolean' ? 'selected' : ''}>boolean</option>
                <option value="array" ${field.type === 'array' ? 'selected' : ''}>array</option>
                <option value="enum" ${field.type === 'enum' ? 'selected' : ''}>enum</option>
            </select>
            ${isEnum ? `
                <input type="text" class="field-enum-input" value="${field.enumOptions || ''}" placeholder="opt1, opt2" title="Enum Options (comma-separated)" style="width: 20%;">
            ` : ''}
            <input type="text" class="field-desc-input" value="${field.desc || ''}" placeholder="${tx('Description (optional)')}" title="${tx('Description')}" style="width: ${isEnum ? '24%' : '44%'};">
            <label class="checkbox-label-inline" style="width: 10%; margin-left: auto; justify-content: flex-end;">
                <input type="checkbox" class="field-required-checkbox" ${field.required ? 'checked' : ''}>
                Req
            </label>
            <button type="button" class="btn-remove-row" title="${tx('Remove field')}">&times;</button>
        `;

        const nameInput = row.querySelector('.field-name-input');
        const typeSelect = row.querySelector('.field-type-select');
        const descInput = row.querySelector('.field-desc-input');
        const reqCheck = row.querySelector('.field-required-checkbox');
        const removeBtn = row.querySelector('.btn-remove-row');
        const enumInput = row.querySelector('.field-enum-input');

        nameInput.addEventListener('input', () => {
            field.name = nameInput.value.replace(/[^a-zA-Z0-9_]/g, '');
            generateSchema();
        });

        typeSelect.addEventListener('change', () => {
            field.type = typeSelect.value;
            if (field.type === 'enum' && !field.enumOptions) {
                field.enumOptions = "option1, option2";
            }
            renderBuilderFields();
            generateSchema();
        });

        if (enumInput) {
            enumInput.addEventListener('input', () => {
                field.enumOptions = enumInput.value;
                generateSchema();
            });
        }

        descInput.addEventListener('input', () => {
            field.desc = descInput.value;
            generateSchema();
        });

        reqCheck.addEventListener('change', () => {
            field.required = reqCheck.checked;
            generateSchema();
        });

        removeBtn.addEventListener('click', () => {
            builderFields.splice(index, 1);
            renderBuilderFields();
            generateSchema();
        });

        UI.builderFieldsList.appendChild(row);
    });
}

function buildJSONSchema() {
    const properties = {};
    const required = [];

    builderFields.forEach(field => {
        if (!field.name) return;

        const propObj = {};
        
        if (field.type === 'array') {
            propObj.type = 'array';
            propObj.items = { type: 'string' };
        } else if (field.type === 'enum') {
            propObj.type = 'string';
            const options = (field.enumOptions || '')
                .split(',')
                .map(o => o.trim())
                .filter(o => o.length > 0);
            propObj.enum = options.length > 0 ? options : ['option1', 'option2'];
        } else {
            propObj.type = field.type;
        }

        if (field.desc) {
            propObj.description = field.desc;
        }

        properties[field.name] = propObj;

        if (field.required) {
            required.push(field.name);
        }
    });

    const schema = {
        type: "object",
        properties: properties,
        additionalProperties: false
    };

    if (required.length > 0) {
        schema.required = required;
    }

    return schema;
}

function generateSchema() {
    const schema = buildJSONSchema();
    UI.schemaPreviewCode.textContent = JSON.stringify(schema, null, 2);

    let payload = "";
    if (activeProvider === 'llamacpp') {
        payload = `curl -X POST ${BASE_URL}/api/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_name": "gemma",
    "messages": [{"role": "user", "content": "Generate structured output."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "custom_schema",
        "strict": true,
        "schema": ${JSON.stringify(schema, null, 8).trim().replace(/\n/g, '\n        ')}
      }
    }
  }'`;
    } else if (activeProvider === 'ollama') {
        payload = `curl -X POST http://localhost:11434/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Generate structured output."}],
    "format": ${JSON.stringify(schema, null, 4).replace(/\n/g, '\n    ')},
    "stream": false
  }'`;
    } else if (activeProvider === 'lmstudio') {
        payload = `curl -X POST http://localhost:1234/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "meta-llama-3-8b-instruct",
    "messages": [{"role": "user", "content": "Generate structured output."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "custom_schema",
        "schema": ${JSON.stringify(schema, null, 8).trim().replace(/\n/g, '\n        ')}
      }
    }
  }'`;
    }

    UI.payloadPreviewCode.textContent = payload;
}

