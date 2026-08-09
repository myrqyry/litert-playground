import { Tensor } from '@litertjs/core';

// Resizes ImageData to target dimensions
export function resizeImageData(imageData: ImageData, targetWidth: number, targetHeight: number): ImageData {
  const srcCanvas = new OffscreenCanvas(imageData.width, imageData.height)
  const srcCtx = srcCanvas.getContext('2d')!
  srcCtx.putImageData(imageData, 0, 0)

  const dstCanvas = new OffscreenCanvas(targetWidth, targetHeight)
  const dstCtx = dstCanvas.getContext('2d')!
  dstCtx.drawImage(srcCanvas, 0, 0, targetWidth, targetHeight)
  return dstCtx.getImageData(0, 0, targetWidth, targetHeight)
}

interface ImageProcessingOptions {
  colorOrder?: 'RGB' | 'BGR';
  dataFormat?: 'NCHW' | 'NHWC';
  normalization?: 'imagenet' | '0-1' | '-1-1' | 'none';
  mean?: number[];
  std?: number[];
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// Converts ImageData to a Tensor with flexible formatting and normalization
export function normalizeAndFormatImageData(
  imageData: ImageData,
  shape: number[], // Expected model input shape [N, C, H, W] or [N, H, W, C]
  options: ImageProcessingOptions = {}
): Tensor {
  const {
    colorOrder = 'RGB',
    dataFormat = 'NCHW',
    normalization = '0-1',
    mean = IMAGENET_MEAN,
    std = IMAGENET_STD,
  } = options;

  const [N, D1, D2, D3] = shape; // D1, D2, D3 can be C, H, W or H, W, C
  const C = dataFormat === 'NCHW' ? D1 : D3;
  const H = dataFormat === 'NCHW' ? D2 : D1;
  const W = dataFormat === 'NCHW' ? D3 : D2;

  const pixelData = new Float32Array(N * C * H * W);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const imgDataIdx = (y * imageData.width + x) * 4; // RGBA input
      let r = imageData.data[imgDataIdx];
      let g = imageData.data[imgDataIdx + 1];
      let b = imageData.data[imgDataIdx + 2];

      if (colorOrder === 'BGR') {
        [b, r] = [r, b]; // Swap R and B
      }

      const pixelValues = [r, g, b];

      for (let c = 0; c < C; c++) {
        let value = pixelValues[c];

        // Apply normalization
        if (normalization === '0-1') {
          value /= 255.0;
        } else if (normalization === '-1-1') {
          value = (value / 127.5) - 1.0;
        } else if (normalization === 'imagenet') {
          value = (value / 255.0 - mean[c]) / std[c];
        } else { // 'none'
          value /= 255.0; // Still divide by 255 to get float values
        }

        let outputIndex;
        if (dataFormat === 'NCHW') {
          outputIndex = c * H * W + y * W + x;
        } else { // NHWC
          outputIndex = y * W * C + x * C + c;
        }
        pixelData[outputIndex] = value;
      }
    }
  }

  return new Tensor(pixelData, shape);
}

// Converts a Tensor (grayscale, 1 channel) to ImageData, specifically for depth maps
export async function tensorToImageData(tensor: Tensor, width: number, height: number): Promise<ImageData> {
  const pixelData = await tensor.data() as Float32Array
  const imageData = new ImageData(width, height);

  // Normalize depth values to 0-255 for display
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < pixelData.length; i++) {
    if (pixelData[i] < min) min = pixelData[i];
    if (pixelData[i] > max) max = pixelData[i];
  }

  const range = max - min;

  for (let i = 0; i < pixelData.length; i++) {
    const value = range === 0 ? 0 : (pixelData[i] - min) / range; // Normalize to [0, 1]
    const byteValue = Math.round(value * 255); // Scale to [0, 255]
    imageData.data[i * 4 + 0] = byteValue; // R
    imageData.data[i * 4 + 1] = byteValue; // G
    imageData.data[i * 4 + 2] = byteValue; // B
    imageData.data[i * 4 + 3] = 255;      // A
  }

  return imageData;
}

// Converts a Tensor (multi-channel or single-channel) to raw ImageData without specific normalization
export async function tensorToRawImageData(tensor: Tensor, width: number, height: number, channels: number = 1): Promise<ImageData> {
  const pixelData = await tensor.data() as Float32Array
  const imageData = new ImageData(width, height)
  const C = channels

  let maxVal = -Infinity;
  for (let i = 0; i < pixelData.length; i++) {
    if (pixelData[i] > maxVal) maxVal = pixelData[i];
  }

  for (let i = 0; i < pixelData.length / C; i++) {
    for (let c = 0; c < C; c++) {
      const value = (pixelData[i * C + c] / maxVal) * 255; // Simple scaling for display
      imageData.data[i * 4 + c] = Math.round(value);
    }
    imageData.data[i * 4 + 3] = 255; // Alpha channel
  }

  return imageData;
}