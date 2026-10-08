import {
    WatermarkEngine,
    detectWatermarkConfig,
    calculateWatermarkPosition
} from './core/watermarkEngine.js';
import { WatermarkWorkerClient, canUseWatermarkWorker } from './core/workerClient.js';
import {
    isConfirmedWatermarkDecision,
    resolveDisplayWatermarkInfo,
    resolveProcessedStatusPresentation
} from './core/watermarkDisplay.js';
import { canvasToBlob } from './core/canvasBlob.js';
import {
    loadImage,
    setStatusMessage,
    showLoading,
    hideLoading
} from './utils.js';
import {
    consumeDebugFileHandoff,
    getDebugFileKind,
    saveDebugFileHandoff
} from './shared/debugFileHandoff.js';


const TEXT = {
    loading: 'Loading resources...',
    size: 'Size',
    watermark: 'Detected Watermark',
    position: 'Position',
    status: 'Status',
    removed: 'Watermark Removed',
    skipped: 'No removable watermark detected, original retained',
    visibleResidual: 'Processed, visible residual may remain',
    possibleContentDamage: 'Best result generated, please check watermark area',
    mixedQualityWarning: 'Best result generated, residual or localized distortion possible',
    unsupported: 'Browser does not support copying images',
    copied: 'Copied!',
    copy: 'Copy Result',
    copyFailed: 'Copy Failed',
    unsupportedFile: 'Please select JPG, PNG, WebP images, or MP4/WebM/MOV videos.',
    fileTooLarge: 'Image preview does not support images larger than 20MB. Videos will redirect to video preview.',
    skippedLargeImages: 'Skipped images over 20MB.',
    handoffVideo: 'Entering video debug workflow...',
    progress: 'Processing Progress',
    pending: 'Pending',
    loadingImage: 'Reading image...',
    processing: 'Processing...',
    processFailed: 'Processing Failed'
};

let enginePromise = null;
let workerClient = null;
let currentItem = null;
let imageQueue = [];
let processedCount = 0;
let activeBatchId = 0;
let liveUpdateTimer = null;

const LOCAL_STORAGE_KEY = 'gwr_classic_corner_settings';

function loadSavedPresetState() {
    const defaults = {
        mode: 'fixed-corner',
        gain: 1.00,
        sizeScale: 1.00,
        positionX: 0,
        positionY: 0
    };
    try {
        const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            return {
                mode: 'fixed-corner',
                gain: typeof parsed.gain === 'number' && !isNaN(parsed.gain) ? parsed.gain : defaults.gain,
                sizeScale: typeof parsed.sizeScale === 'number' && !isNaN(parsed.sizeScale) ? parsed.sizeScale : defaults.sizeScale,
                positionX: typeof parsed.positionX === 'number' && !isNaN(parsed.positionX) ? parsed.positionX : defaults.positionX,
                positionY: typeof parsed.positionY === 'number' && !isNaN(parsed.positionY) ? parsed.positionY : defaults.positionY
            };
        }
    } catch (e) {
        console.warn('Failed to load preset settings from localStorage:', e);
    }
    return defaults;
}

function savePresetState() {
    try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify({
            gain: presetState.gain,
            sizeScale: presetState.sizeScale,
            positionX: presetState.positionX,
            positionY: presetState.positionY
        }));
    } catch (e) {
        console.warn('Failed to save preset settings to localStorage:', e);
    }
}

const presetState = loadSavedPresetState();

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const singlePreview = document.getElementById('singlePreview');
const multiPreview = document.getElementById('multiPreview');
const imageList = document.getElementById('imageList');
const progressText = document.getElementById('progressText');
const originalImage = document.getElementById('originalImage');
const processedImage = document.getElementById('processedImage');
const originalInfo = document.getElementById('originalInfo');
const processedInfo = document.getElementById('processedInfo');
const downloadBtn = document.getElementById('downloadBtn');
const copyBtn = document.getElementById('copyBtn');
const resetBtn = document.getElementById('resetBtn');
const batchResetBtn = document.getElementById('batchResetBtn');
const processedOverlay = document.getElementById('processedOverlay');
const sliderHandle = document.getElementById('sliderHandle');

// New Preset Controls & Overlay Elements
const modelPresetSection = document.getElementById('modelPresetSection');
const presetAdaptiveBtn = document.getElementById('presetAdaptiveBtn');
const presetFixedBtn = document.getElementById('presetFixedBtn');
const fixedControlsPanel = document.getElementById('fixedControlsPanel');
const processAllBtn = document.getElementById('processAllBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');

const strengthGainInput = document.getElementById('strengthGainInput');
const strengthGainVal = document.getElementById('strengthGainVal');
const sizeScaleInput = document.getElementById('sizeScaleInput');
const sizeScaleVal = document.getElementById('sizeScaleVal');
const positionXInput = document.getElementById('positionXInput');
const positionXVal = document.getElementById('positionXVal');
const positionYInput = document.getElementById('positionYInput');
const positionYVal = document.getElementById('positionYVal');

const watermarkBoxOverlay = document.getElementById('watermarkBoxOverlay');
const zoomPreviewCard = document.getElementById('zoomPreviewCard');
const zoomCanvas = document.getElementById('zoomCanvas');

async function getEngine() {
    if (!enginePromise) {
        enginePromise = WatermarkEngine.create().catch((error) => {
            enginePromise = null;
            throw error;
        });
    }
    return enginePromise;
}

function getCurrentOptions() {
    return {
        presetMode: 'fixed-corner',
        gain: presetState.gain,
        sizeScale: presetState.sizeScale,
        positionX: presetState.positionX,
        positionY: presetState.positionY
    };
}

function disableWorkerClient(reason) {
    if (!workerClient) return;
    console.warn('disable worker path, fallback to main thread:', reason);
    workerClient.dispose();
    workerClient = null;
}

function cleanupCurrentItem() {
    if (!currentItem) return;
    if (currentItem.originalUrl) URL.revokeObjectURL(currentItem.originalUrl);
    if (currentItem.processedUrl) URL.revokeObjectURL(currentItem.processedUrl);
    currentItem = null;
}

function cleanupBatchItems() {
    activeBatchId++;
    imageQueue.forEach((item) => {
        if (item.originalUrl) URL.revokeObjectURL(item.originalUrl);
        if (item.processedUrl) URL.revokeObjectURL(item.processedUrl);
    });
    imageQueue = [];
    processedCount = 0;
}

async function init() {
    try {
        showLoading(TEXT.loading);

        if (canUseWatermarkWorker()) {
            try {
                workerClient = new WatermarkWorkerClient({
                    workerUrl: './workers/watermark-worker.js'
                });
            } catch (workerError) {
                console.warn('worker unavailable, fallback to main thread:', workerError);
                workerClient = null;
            }
        }

        if (!workerClient) {
            getEngine().catch((error) => {
                console.warn('main thread engine warmup failed:', error);
            });
        }

        hideLoading();
        setupEventListeners();
        setupPresetEventListeners();
        setupSlider();
        await consumePendingImageHandoff();
    } catch (error) {
        hideLoading();
        console.error('initialize error:', error);
    }
}

function setupPresetEventListeners() {
    if (strengthGainInput) {
        strengthGainInput.value = presetState.gain;
        if (strengthGainVal) strengthGainVal.textContent = presetState.gain.toFixed(2) + 'x';
        strengthGainInput.addEventListener('input', (e) => {
            presetState.gain = parseFloat(e.target.value);
            if (strengthGainVal) strengthGainVal.textContent = presetState.gain.toFixed(2) + 'x';
            savePresetState();
            triggerLivePreviewUpdate();
        });
    }

    if (sizeScaleInput) {
        sizeScaleInput.value = presetState.sizeScale;
        if (sizeScaleVal) sizeScaleVal.textContent = presetState.sizeScale.toFixed(2) + 'x';
        sizeScaleInput.addEventListener('input', (e) => {
            presetState.sizeScale = parseFloat(e.target.value);
            if (sizeScaleVal) sizeScaleVal.textContent = presetState.sizeScale.toFixed(2) + 'x';
            savePresetState();
            triggerLivePreviewUpdate();
        });
    }

    if (positionXInput) {
        positionXInput.value = presetState.positionX;
        if (positionXVal) positionXVal.textContent = presetState.positionX + 'px';
        positionXInput.addEventListener('input', (e) => {
            presetState.positionX = parseInt(e.target.value, 10);
            if (positionXVal) positionXVal.textContent = presetState.positionX + 'px';
            savePresetState();
            triggerLivePreviewUpdate();
        });
    }

    if (positionYInput) {
        positionYInput.value = presetState.positionY;
        if (positionYVal) positionYVal.textContent = presetState.positionY + 'px';
        positionYInput.addEventListener('input', (e) => {
            presetState.positionY = parseInt(e.target.value, 10);
            if (positionYVal) positionYVal.textContent = presetState.positionY + 'px';
            savePresetState();
            triggerLivePreviewUpdate();
        });
    }

    if (downloadZipBtn) {
        downloadZipBtn.addEventListener('click', downloadAllAsZip);
    }
}

function updatePresetUi() {
    if (fixedControlsPanel) {
        fixedControlsPanel.style.display = 'grid';
    }
}

function triggerLivePreviewUpdate() {
    clearTimeout(liveUpdateTimer);
    liveUpdateTimer = setTimeout(() => {
        const item = currentItem || imageQueue[0];
        if (item) {
            reprocessSingleItem(item);
        }
    }, 40);
}

async function reprocessSingleItem(item) {
    try {
        const img = item.originalImg || await loadImage(item.file);
        item.originalImg = img;
        item.originalUrl = img.src;

        const options = getCurrentOptions();
        const processed = await processImageWithBestPath(item.file, img, options);
        if (item.processedUrl) URL.revokeObjectURL(item.processedUrl);
        item.processedMeta = processed.meta;
        item.processedBlob = processed.blob;
        item.processedUrl = URL.createObjectURL(processed.blob);

        if (currentItem === item || !currentItem) {
            originalImage.src = img.src;
            processedImage.src = item.processedUrl;
            processedOverlay.style.display = 'block';
            sliderHandle.style.display = 'flex';
            processedInfo.style.display = 'block';
            renderSingleProcessedMeta(item);
            updateWatermarkBoxOverlay(item);
        }

        const cardProcessed = document.getElementById(`processed-${item.id}`);
        if (cardProcessed) {
            cardProcessed.src = item.processedUrl;
        }
    } catch (err) {
        console.error('Failed to reprocess preview:', err);
    }
}

function getFixedCornerPosition(imgWidth, imgHeight, options = {}) {
    const baseConfig = detectWatermarkConfig(imgWidth, imgHeight);
    const sizeScale = Number.isFinite(options.sizeScale) ? options.sizeScale : 1.0;
    const targetSize = Math.max(8, Math.round(baseConfig.logoSize * sizeScale));
    const positionX = Number.isFinite(options.positionX) ? options.positionX : 0;
    const positionY = Number.isFinite(options.positionY) ? options.positionY : 0;
    const baseX = imgWidth - baseConfig.marginRight - targetSize;
    const baseY = imgHeight - baseConfig.marginBottom - targetSize;
    return {
        x: Math.max(0, Math.min(imgWidth - targetSize, Math.round(baseX + positionX))),
        y: Math.max(0, Math.min(imgHeight - targetSize, Math.round(baseY + positionY))),
        width: targetSize,
        height: targetSize
    };
}

function updateWatermarkBoxOverlay(item) {
    if (!item?.originalImg || !originalImage) return;

    const imgW = item.originalImg.width;
    const imgH = item.originalImg.height;
    const options = getCurrentOptions();
    const pos = item.processedMeta?.selectedCandidate?.position || getFixedCornerPosition(imgW, imgH, options);

    const applyBox = () => {
        const rect = originalImage.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        const scaleX = rect.width / imgW;
        const scaleY = rect.height / imgH;

        const bLeft = pos.x * scaleX;
        const bTop = pos.y * scaleY;
        const bW = pos.width * scaleX;
        const bH = pos.height * scaleY;

        if (watermarkBoxOverlay) {
            watermarkBoxOverlay.style.left = `${bLeft}px`;
            watermarkBoxOverlay.style.top = `${bTop}px`;
            watermarkBoxOverlay.style.width = `${bW}px`;
            watermarkBoxOverlay.style.height = `${bH}px`;
            watermarkBoxOverlay.style.display = 'block';
        }
    };

    applyBox();

    if (zoomPreviewCard && zoomCanvas && item.processedUrl) {
        zoomPreviewCard.style.display = 'flex';
        const ctx = zoomCanvas.getContext('2d');
        const zoomImg = new Image();
        zoomImg.crossOrigin = 'anonymous';
        zoomImg.onload = () => {
            const pad = 16;
            const cropX = Math.max(0, pos.x - pad);
            const cropY = Math.max(0, pos.y - pad);
            const cropW = Math.min(imgW - cropX, pos.width + pad * 2);
            const cropH = Math.min(imgH - cropY, pos.height + pad * 2);

            ctx.clearRect(0, 0, zoomCanvas.width, zoomCanvas.height);
            ctx.drawImage(zoomImg, cropX, cropY, cropW, cropH, 0, 0, zoomCanvas.width, zoomCanvas.height);

            const scaleZoomX = zoomCanvas.width / cropW;
            const scaleZoomY = zoomCanvas.height / cropH;
            const zBoxX = (pos.x - cropX) * scaleZoomX;
            const zBoxY = (pos.y - cropY) * scaleZoomY;
            const zBoxW = pos.width * scaleZoomX;
            const zBoxH = pos.height * scaleZoomY;

            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.strokeRect(zBoxX, zBoxY, zBoxW, zBoxH);
        };
        zoomImg.src = item.processedUrl;
    }
}

function setupEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-primary', 'bg-emerald-50');
    });

    document.addEventListener('dragleave', (e) => {
        if (e.clientX === 0 && e.clientY === 0) {
            uploadArea.classList.remove('border-primary', 'bg-emerald-50');
        }
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-primary', 'bg-emerald-50');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });

    document.addEventListener('paste', (e) => {
        const items = e.clipboardData.items;
        const files = [];
        for (let i = 0; i < items.length; i++) {
            if (items[i].kind === 'file') {
                files.push(items[i].getAsFile());
            }
        }
        if (files.length > 0) handleFiles(files);
    });

    if (resetBtn) resetBtn.addEventListener('click', reset);
    if (batchResetBtn) batchResetBtn.addEventListener('click', reset);
    window.addEventListener('beforeunload', () => {
        cleanupBatchItems();
        disableWorkerClient('beforeunload');
    });
}

function reset() {
    cleanupCurrentItem();
    cleanupBatchItems();
    if (singlePreview) singlePreview.style.display = 'none';
    if (multiPreview) multiPreview.style.display = 'none';
    if (modelPresetSection) modelPresetSection.style.display = 'none';
    if (watermarkBoxOverlay) watermarkBoxOverlay.style.display = 'none';
    if (zoomPreviewCard) zoomPreviewCard.style.display = 'none';
    if (fileInput) fileInput.value = '';
    if (imageList) imageList.innerHTML = '';
    updateProgress();
    if (originalImage) originalImage.src = '';
    if (processedImage) processedImage.src = '';
    if (originalInfo) originalInfo.innerHTML = '';
    if (processedInfo) {
        processedInfo.innerHTML = '';
        processedInfo.style.display = 'none';
    }
    if (processedOverlay) processedOverlay.style.display = 'none';
    if (sliderHandle) sliderHandle.style.display = 'none';
    if (copyBtn) copyBtn.style.display = 'none';
    if (downloadBtn) downloadBtn.style.display = 'none';
    setStatusMessage('');
    if (uploadArea) uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function handleFileSelect(e) {
    handleFiles(Array.from(e.target.files));
}

async function handleFiles(files) {
    setStatusMessage('');

    const list = Array.from(files || []).filter(Boolean);
    const videoFile = list.find((file) => getDebugFileKind(file) === 'video');
    if (videoFile) {
        await routeVideoFile(videoFile);
        return;
    }

    const imageFiles = list.filter((file) => getDebugFileKind(file) === 'image');
    if (imageFiles.length === 0) {
        setStatusMessage(TEXT.unsupportedFile, 'warn');
        return;
    }

    const validImageFiles = imageFiles.filter((file) => file.size <= 20 * 1024 * 1024);
    if (validImageFiles.length === 0) {
        setStatusMessage(TEXT.fileTooLarge, 'warn');
        return;
    }

    if (validImageFiles.length < imageFiles.length) {
        setStatusMessage(TEXT.skippedLargeImages, 'warn');
    }

    if (modelPresetSection) modelPresetSection.style.display = 'block';

    if (validImageFiles.length > 1) {
        processBatch(validImageFiles);
        return;
    }

    const validFile = validImageFiles[0];
    cleanupCurrentItem();
    cleanupBatchItems();
    if (multiPreview) multiPreview.style.display = 'none';
    if (imageList) imageList.innerHTML = '';
    currentItem = {
        id: Date.now(),
        file: validFile,
        name: validFile.name,
        originalImg: null,
        processedMeta: null,
        processedBlob: null,
        originalUrl: null,
        processedUrl: null
    };

    imageQueue = [currentItem];
    if (singlePreview) singlePreview.style.display = 'block';
    processSingle(currentItem);
}

function createDebugImageItem(file, index) {
    return {
        id: `${Date.now()}-${index}`,
        file,
        name: file.name,
        status: 'pending',
        originalImg: null,
        processedMeta: null,
        processedBlob: null,
        originalUrl: null,
        processedUrl: null
    };
}

function processBatch(files) {
    cleanupCurrentItem();
    cleanupBatchItems();

    imageQueue = files.map(createDebugImageItem);
    currentItem = imageQueue[0];

    singlePreview.style.display = 'block';
    if (multiPreview) multiPreview.style.display = 'none';

    if (currentItem) {
        processSingle(currentItem);
    }
}

async function processAllItems() {
    const queue = imageQueue.length > 0 ? imageQueue : (currentItem ? [currentItem] : []);
    if (queue.length === 0) return;

    showLoading('Processing all images...');
    activeBatchId++;
    const batchId = activeBatchId;
    processedCount = 0;
    updateProgress();

    // Reset status to pending so processQueue processes them
    queue.forEach(item => {
        item.status = 'pending';
        renderImageCardStatus(item);
    });

    await processQueue(batchId);
    hideLoading();
    setStatusMessage('All images processed successfully with current settings!', 'success');
}

const crc32Table = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
        c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crc32Table[n] = c;
}

function computeCrc32(uint8Array) {
    let crc = 0xffffffff;
    for (let i = 0; i < uint8Array.length; i++) {
        crc = crc32Table[(crc ^ uint8Array[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function createZipBlob(files) {
    const localHeaders = [];
    const cdHeaders = [];
    let offset = 0;
    const encoder = new TextEncoder();

    for (const file of files) {
        const fileNameBytes = encoder.encode(file.name);
        const fileData = file.data;
        const fileCrc = computeCrc32(fileData);
        const dataLength = fileData.length;

        const localHeader = new Uint8Array(30 + fileNameBytes.length);
        const dvLoc = new DataView(localHeader.buffer);
        dvLoc.setUint32(0, 0x04034b50, true);
        dvLoc.setUint16(4, 10, true);
        dvLoc.setUint16(6, 0, true);
        dvLoc.setUint16(8, 0, true);
        dvLoc.setUint16(10, 0, true);
        dvLoc.setUint16(12, 0, true);
        dvLoc.setUint32(14, fileCrc, true);
        dvLoc.setUint32(18, dataLength, true);
        dvLoc.setUint32(22, dataLength, true);
        dvLoc.setUint16(26, fileNameBytes.length, true);
        dvLoc.setUint16(28, 0, true);
        localHeader.set(fileNameBytes, 30);

        const cdHeader = new Uint8Array(46 + fileNameBytes.length);
        const dvCd = new DataView(cdHeader.buffer);
        dvCd.setUint32(0, 0x02014b50, true);
        dvCd.setUint16(4, 20, true);
        dvCd.setUint16(6, 10, true);
        dvCd.setUint16(8, 0, true);
        dvCd.setUint16(10, 0, true);
        dvCd.setUint16(12, 0, true);
        dvCd.setUint16(14, 0, true);
        dvCd.setUint32(16, fileCrc, true);
        dvCd.setUint32(20, dataLength, true);
        dvCd.setUint32(24, dataLength, true);
        dvCd.setUint16(28, fileNameBytes.length, true);
        dvCd.setUint16(30, 0, true);
        dvCd.setUint16(32, 0, true);
        dvCd.setUint16(34, 0, true);
        dvCd.setUint16(36, 0, true);
        dvCd.setUint32(38, 0, true);
        dvCd.setUint32(42, offset, true);
        cdHeader.set(fileNameBytes, 46);

        localHeaders.push(localHeader, fileData);
        cdHeaders.push(cdHeader);

        offset += localHeader.length + fileData.length;
    }

    const cdOffset = offset;
    let cdSize = 0;
    for (const cd of cdHeaders) {
        cdSize += cd.length;
    }

    const eocd = new Uint8Array(22);
    const dvEocd = new DataView(eocd.buffer);
    dvEocd.setUint32(0, 0x06054b50, true);
    dvEocd.setUint16(4, 0, true);
    dvEocd.setUint16(6, 0, true);
    dvEocd.setUint16(8, files.length, true);
    dvEocd.setUint16(10, files.length, true);
    dvEocd.setUint32(12, cdSize, true);
    dvEocd.setUint32(16, cdOffset, true);
    dvEocd.setUint16(20, 0, true);

    return new Blob([...localHeaders, ...cdHeaders, eocd], { type: 'application/zip' });
}

async function downloadAllAsZip() {
    const queue = imageQueue.length > 0 ? imageQueue : (currentItem ? [currentItem] : []);
    if (queue.length === 0) {
        setStatusMessage('No processed images to download', 'warn');
        return;
    }

    showLoading('Packing images into ZIP...');
    try {
        const zipFiles = [];
        const options = getCurrentOptions();

        for (let i = 0; i < queue.length; i++) {
            const item = queue[i];
            const img = item.originalImg || await loadImage(item.file);
            const processed = await processImageWithBestPath(item.file, img, options);
            item.processedBlob = processed.blob;
            const fileName = item.name || `image_${i + 1}.png`;
            const arrayBuffer = await processed.blob.arrayBuffer();
            zipFiles.push({
                name: fileName,
                data: new Uint8Array(arrayBuffer)
            });
        }

        const content = createZipBlob(zipFiles);
        const downloadLink = document.createElement('a');
        downloadLink.href = URL.createObjectURL(content);
        downloadLink.download = 'unwatermarked_images.zip';
        downloadLink.click();

        setTimeout(() => URL.revokeObjectURL(downloadLink.href), 5000);
        hideLoading();
        setStatusMessage('ZIP file downloaded successfully!', 'success');
    } catch (err) {
        hideLoading();
        console.error('ZIP generation error:', err);
        setStatusMessage('Failed to generate ZIP archive', 'warn');
    }
}

async function routeVideoFile(file) {
    try {
        showLoading(TEXT.handoffVideo);
        await saveDebugFileHandoff(file, 'video');
        window.location.assign('./video-preview.html?fileHandoff=1');
    } catch (error) {
        hideLoading();
        console.error(error);
        setStatusMessage(error.message || 'Unable to enter video debug workflow, please open the video page and select the file again.', 'warn');
    }
}

async function consumePendingImageHandoff() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('fileHandoff') !== '1') return;

    try {
        const record = await consumeDebugFileHandoff('image');
        if (!record?.file) return;
        await handleFiles([record.file]);
        window.history.replaceState(null, '', window.location.pathname);
    } catch (error) {
        console.warn('image handoff unavailable:', error);
        setStatusMessage(error.message || 'Failed to read image cache, please select the file again.', 'warn');
    }
}

function renderSingleImageMeta(item) {
    if (!item?.originalImg) return;

    const watermarkInfo = resolveDisplayWatermarkInfo(
        item,
        getEstimatedWatermarkInfo(item)
    );
    if (!watermarkInfo) return;

    originalInfo.innerHTML = `
        <p>${TEXT.size}: ${item.originalImg.width}x${item.originalImg.height}</p>
        <p>${TEXT.watermark}: ${watermarkInfo.size}x${watermarkInfo.size}</p>
        <p>${TEXT.position}: (${watermarkInfo.position.x},${watermarkInfo.position.y})</p>
    `;
}

function getProcessedStatusPresentation(item) {
    const presentation = resolveProcessedStatusPresentation(item);
    return {
        label: TEXT[presentation.messageKey],
        tone: presentation.tone
    };
}

function renderSingleProcessedMeta(item) {
    if (!item?.originalImg) return;

    const watermarkInfo = resolveDisplayWatermarkInfo(
        item,
        getEstimatedWatermarkInfo(item)
    );
    const showWatermarkInfo = watermarkInfo && (isConfirmedWatermarkDecision(item) || presetState.mode === 'fixed-corner');
    const statusPresentation = getProcessedStatusPresentation(item);

    const pos = item.processedMeta?.selectedCandidate?.position || watermarkInfo?.position;
    const posStr = pos ? `(${pos.x},${pos.y})` : '';

    processedInfo.innerHTML = `
        <p>${TEXT.size}: ${item.originalImg.width}x${item.originalImg.height}</p>
        ${showWatermarkInfo ? `<p>${TEXT.watermark}: ${pos?.width || watermarkInfo.size}x${pos?.height || watermarkInfo.size}</p>` : ''}
        ${showWatermarkInfo ? `<p>${TEXT.position}: ${posStr}</p>` : ''}
        <p class="${statusPresentation.tone === 'warning' ? 'text-warning' : ''}">${TEXT.status}: ${presetState.mode === 'fixed-corner' ? 'Classic Corner (Fixed)' : statusPresentation.label}</p>
    `;
}

async function processSingle(item) {
    try {
        const img = await loadImage(item.file);
        item.originalImg = img;
        item.originalUrl = img.src;

        originalImage.src = img.src;
        renderSingleImageMeta(item);

        const options = getCurrentOptions();
        const processed = await processImageWithBestPath(item.file, img, options);
        item.processedMeta = processed.meta;
        item.processedBlob = processed.blob;
        item.processedUrl = URL.createObjectURL(processed.blob);

        processedImage.src = item.processedUrl;
        processedOverlay.style.display = 'block';
        sliderHandle.style.display = 'flex';
        processedInfo.style.display = 'block';

        copyBtn.style.display = 'flex';
        copyBtn.onclick = () => copyImage(item);

        downloadBtn.style.display = 'flex';
        downloadBtn.onclick = () => downloadImage(item);

        renderSingleProcessedMeta(item);
        updateWatermarkBoxOverlay(item);
        document.getElementById('comparisonContainer').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error(error);
    }
}

function createImageCard(item) {
    const card = document.createElement('div');
    card.id = `card-${item.id}`;
    card.className = 'batch-card';
    card.innerHTML = `
        <div class="batch-comparison">
            <div class="batch-pane original">
                <span class="batch-pane-label">Original</span>
                <img id="original-${item.id}" class="batch-image" draggable="false" alt="" />
            </div>
            <div class="batch-pane processed">
                <span class="batch-pane-label">Processed</span>
                <img id="processed-${item.id}" class="batch-image" draggable="false" alt="" />
            </div>
        </div>
        <div class="batch-main">
            <h4 class="batch-title"></h4>
            <div class="batch-status" id="status-${item.id}"></div>
        </div>
        <div class="batch-actions">
            <button id="copy-${item.id}" class="batch-button primary" style="display: none;">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m-1 10H8m4-3H8m1.5 6H8"></path></svg>
                <span>${TEXT.copy}</span>
            </button>
            <button id="download-${item.id}" class="batch-button secondary" style="display: none;">
                Download Result
            </button>
        </div>
    `;
    imageList.appendChild(card);

    const title = card.querySelector('.batch-title');
    title.textContent = typeof item.name === 'string' ? item.name : '';
    title.title = title.textContent;
    renderImageCardStatus(item);
}

function renderImageCardStatus(item) {
    const statusEl = document.getElementById(`status-${item.id}`);
    if (!statusEl) return;

    statusEl.classList.remove('text-primary', 'text-warning');
    if (item.status === 'completed') {
        const presentation = getProcessedStatusPresentation(item);
        statusEl.textContent = presetState.mode === 'fixed-corner' ? 'Classic Corner (Fixed)' : presentation.label;
        statusEl.classList.add(presentation.tone === 'warning' ? 'text-warning' : 'text-primary');
        return;
    }

    const labels = {
        pending: TEXT.pending,
        loading: TEXT.loadingImage,
        processing: TEXT.processing,
        error: TEXT.processFailed
    };
    statusEl.textContent = labels[item.status] || TEXT.pending;
}

function updateProgress() {
    if (progressText) {
        progressText.textContent = `${TEXT.progress}: ${processedCount}/${imageQueue.length}`;
    }
}

async function processQueue(batchId) {
    const concurrency = 3;
    const options = getCurrentOptions();
    for (let i = 0; i < imageQueue.length; i += concurrency) {
        if (batchId !== activeBatchId) return;

        await Promise.all(imageQueue.slice(i, i + concurrency).map(async (item) => {
            if (batchId !== activeBatchId || item.status !== 'pending') return;

            item.status = 'loading';
            renderImageCardStatus(item);

            try {
                const img = await loadImage(item.file);
                if (batchId !== activeBatchId) return;

                item.originalImg = img;
                item.originalUrl = img.src;
                const originalPreview = document.getElementById(`original-${item.id}`);
                const processedPreview = document.getElementById(`processed-${item.id}`);
                if (originalPreview) originalPreview.src = img.src;

                item.status = 'processing';
                renderImageCardStatus(item);

                const processed = await processImageWithBestPath(item.file, img, options);
                if (batchId !== activeBatchId) return;

                item.processedMeta = processed.meta;
                item.processedBlob = processed.blob;
                item.processedUrl = URL.createObjectURL(processed.blob);
                if (processedPreview) processedPreview.src = item.processedUrl;

                item.status = 'completed';
                processedCount++;
                renderImageCardStatus(item);
                updateProgress();

                const itemCopyBtn = document.getElementById(`copy-${item.id}`);
                if (itemCopyBtn) {
                    itemCopyBtn.style.display = 'inline-flex';
                    itemCopyBtn.onclick = () => copyImage(item, itemCopyBtn);
                }

                const itemDownloadBtn = document.getElementById(`download-${item.id}`);
                if (itemDownloadBtn) {
                    itemDownloadBtn.style.display = 'inline-flex';
                    itemDownloadBtn.onclick = () => downloadImage(item);
                }
            } catch (error) {
                if (batchId !== activeBatchId) return;
                item.status = 'error';
                renderImageCardStatus(item);
                console.error(error);
            }
        }));
    }
}

async function processImageWithBestPath(file, fallbackImage, options = {}) {
    if (workerClient) {
        try {
            return await workerClient.processBlob(file, options);
        } catch (error) {
            console.warn('worker process failed, fallback to main thread:', error);
            disableWorkerClient(error);
        }
    }

    const engine = await getEngine();
    const canvas = await engine.removeWatermarkFromImage(fallbackImage, options);
    const blob = await canvasToBlob(canvas);
    return {
        blob,
        meta: canvas.__watermarkMeta || null
    };
}

async function copyImage(item, targetBtn = copyBtn) {
    if (!navigator.clipboard || !window.ClipboardItem) {
        setStatusMessage(TEXT.unsupported, 'warn');
        return;
    }

    try {
        if (!item.processedBlob) return;
        const data = [new ClipboardItem({ [item.processedBlob.type]: item.processedBlob })];
        await navigator.clipboard.write(data);

        const span = targetBtn.querySelector('span');
        const svg = targetBtn.querySelector('svg');
        const originalSvgPath = svg.innerHTML;

        span.textContent = TEXT.copied;
        svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>';

        setTimeout(() => {
            span.textContent = TEXT.copy;
            svg.innerHTML = originalSvgPath;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy image: ', err);
        setStatusMessage(TEXT.copyFailed, 'warn');
    }
}

function downloadImage(item) {
    const a = document.createElement('a');
    a.href = item.processedUrl;
    a.download = item.name || 'processed.png';
    a.click();
}

function setupSlider() {
    const container = document.getElementById('comparisonContainer');
    let isDown = false;

    function move(e) {
        if (!isDown) return;
        const rect = container.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        if (!clientX) return;

        const x = clientX - rect.left;
        const percent = Math.min(Math.max(x / rect.width, 0), 1) * 100;

        processedOverlay.style.width = `${percent}%`;
        sliderHandle.style.left = `${percent}%`;
    }

    container.addEventListener('mousedown', (e) => {
        isDown = true;
        move(e);
    });
    window.addEventListener('mouseup', () => { isDown = false; });
    window.addEventListener('mousemove', move);

    container.addEventListener('touchstart', (e) => {
        isDown = true;
        move(e);
    });
    window.addEventListener('touchend', () => { isDown = false; });
    window.addEventListener('touchmove', move);
}

init();
