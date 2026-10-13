import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
    getFixedCornerVideoPosition
} from '../../src/video/videoWatermarkCatalog.js';
import {
    processVideoWatermarkFrame
} from '../../src/video/videoExport.js';
import { getVideoAlphaMap } from '../../src/video/videoWatermarkDetector.js';

test('getFixedCornerVideoPosition should compute default 1080p corner watermark position', () => {
    const pos = getFixedCornerVideoPosition(1920, 1080);
    assert.equal(pos.width, 72);
    assert.equal(pos.height, 72);
    // 1920 - 108 - 72 = 1740
    assert.equal(pos.x, 1740);
    // 1080 - 108 - 72 = 900
    assert.equal(pos.y, 900);
});

test('getFixedCornerVideoPosition should adjust position and size based on options', () => {
    const pos = getFixedCornerVideoPosition(1920, 1080, {
        sizeScale: 1.2,
        positionX: 15,
        positionY: -10
    });
    const expectedSize = Math.round(72 * 1.2); // 86
    assert.equal(pos.width, expectedSize);
    assert.equal(pos.height, expectedSize);
    const expectedBaseX = 1920 - 108 - expectedSize;
    const expectedBaseY = 1080 - 108 - expectedSize;
    assert.equal(pos.x, expectedBaseX + 15);
    assert.equal(pos.y, expectedBaseY - 10);
});

test('getFixedCornerVideoPosition should clamp coordinates within video bounds', () => {
    const posMax = getFixedCornerVideoPosition(1920, 1080, {
        positionX: 500,
        positionY: 500
    });
    assert.ok(posMax.x + posMax.width <= 1920);
    assert.ok(posMax.y + posMax.height <= 1080);

    const posMin = getFixedCornerVideoPosition(1920, 1080, {
        positionX: -3000,
        positionY: -3000
    });
    assert.equal(posMin.x, 0);
    assert.equal(posMin.y, 0);
});

test('processVideoWatermarkFrame should process frame without skipping in fixed-corner mode', async () => {
    const width = 100;
    const height = 100;
    const canvas = {
        width,
        height
    };
    const ctx = {
        canvas,
        getImageData(x, y, w, h) {
            const data = new Uint8ClampedArray(w * h * 4);
            // Fill with uniform gray
            data.fill(128);
            return { width: w, height: h, data };
        },
        putImageData() {}
    };

    const position = { x: 10, y: 10, width: 32, height: 32 };
    const alphaMap = getVideoAlphaMap(32);
    const detection = {
        isFixedCorner: true,
        position,
        alphaMap,
        candidate: { id: 'fixed-corner', ...position }
    };

    const result = await processVideoWatermarkFrame(ctx, detection, new Map(), {
        presetMode: 'fixed-corner',
        seedAlphaGain: 1.0,
        lowConfidenceThreshold: 0.99 // High threshold that would normally skip
    });

    assert.equal(result.skipped, false);
    assert.equal(result.alphaGain, 1.0);
});

test('video-preview.html includes Classic Corner (Fixed) settings markup and preview elements', async () => {
    const html = await readFile(new URL('../../public/video-preview.html', import.meta.url), 'utf8');

    assert.match(html, /Classic Corner \(Fixed\) Settings/);
    assert.match(html, /strengthGainInput/);
    assert.match(html, /sizeScaleInput/);
    assert.match(html, /positionXInput/);
    assert.match(html, /positionYInput/);
    assert.match(html, /zoomPreviewCard/);
    assert.match(html, /zoomCanvas/);
    assert.match(html, /watermarkBoxOverlay/);
    assert.match(html, /previewCanvas/);
    assert.match(html, /fixedExportBtn/);
});
