import { Image } from 'react-native';
import RNFS from 'react-native-fs';

export interface ProcessedImage {
  data: Uint8Array;
  width: number;
  height: number;
}

/**
 * Load and process an image from a file path
 * @param imagePath - Local file path to the image
 * @returns Processed image data ready for model input
 */
export async function processImageFromPath(imagePath: string): Promise<ProcessedImage> {
  try {
    // Read the image file as base64
    const imageBase64 = await RNFS.readFile(imagePath, 'base64');
    
    // Convert base64 to Uint8Array
    const imageData = base64ToUint8Array(imageBase64);
    
    // Get image dimensions
    const dimensions = await getImageDimensions(imagePath);
    
    return {
      data: imageData,
      width: dimensions.width,
      height: dimensions.height
    };
  } catch (error) {
    console.error('Error processing image:', error);
    throw new Error('Failed to process image for analysis');
  }
}

/**
 * Convert base64 string to Uint8Array
 */
function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  return bytes;
}

/**
 * Get image dimensions from file path
 */
function getImageDimensions(imagePath: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    Image.getSize(
      imagePath,
      (width, height) => resolve({ width, height }),
      (error) => reject(error)
    );
  });
}

/**
 * Resize image data to target dimensions using bilinear interpolation
 * This is a simplified version - in production, use a proper image processing library
 */
export function resizeImage(
  sourceData: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
): Uint8Array {
  const output = new Uint8Array(targetWidth * targetHeight * 3);
  const xRatio = sourceWidth / targetWidth;
  const yRatio = sourceHeight / targetHeight;
  
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const px = Math.floor(x * xRatio);
      const py = Math.floor(y * yRatio);
      
      const sourceIdx = (py * sourceWidth + px) * 4; // RGBA
      const targetIdx = (y * targetWidth + x) * 3; // RGB
      
      // Copy RGB values, skip alpha
      output[targetIdx] = sourceData[sourceIdx];
      output[targetIdx + 1] = sourceData[sourceIdx + 1];
      output[targetIdx + 2] = sourceData[sourceIdx + 2];
    }
  }
  
  return output;
}

/**
 * Normalize pixel values to [0, 1] range
 */
export function normalizePixels(data: Uint8Array): Float32Array {
  const normalized = new Float32Array(data.length);
  
  for (let i = 0; i < data.length; i++) {
    normalized[i] = data[i] / 255.0;
  }
  
  return normalized;
}

/**
 * Apply image augmentation for better model performance
 */
export function augmentImage(
  data: Uint8Array,
  width: number,
  height: number,
  options: {
    brightness?: number; // -1 to 1
    contrast?: number; // 0 to 2
    saturation?: number; // 0 to 2
  } = {}
): Uint8Array {
  const output = new Uint8Array(data.length);
  const { brightness = 0, contrast = 1, saturation = 1 } = options;
  
  for (let i = 0; i < data.length; i += 3) {
    // Get RGB values
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];
    
    // Apply brightness
    r = Math.min(255, Math.max(0, r + brightness * 255));
    g = Math.min(255, Math.max(0, g + brightness * 255));
    b = Math.min(255, Math.max(0, b + brightness * 255));
    
    // Apply contrast
    r = Math.min(255, Math.max(0, ((r - 128) * contrast) + 128));
    g = Math.min(255, Math.max(0, ((g - 128) * contrast) + 128));
    b = Math.min(255, Math.max(0, ((b - 128) * contrast) + 128));
    
    // Apply saturation (simplified)
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    r = Math.min(255, Math.max(0, gray + saturation * (r - gray)));
    g = Math.min(255, Math.max(0, gray + saturation * (g - gray)));
    b = Math.min(255, Math.max(0, gray + saturation * (b - gray)));
    
    output[i] = r;
    output[i + 1] = g;
    output[i + 2] = b;
  }
  
  return output;
}

/**
 * Crop center square from image
 */
export function cropCenterSquare(
  data: Uint8Array,
  width: number,
  height: number
): { data: Uint8Array; size: number } {
  const size = Math.min(width, height);
  const xOffset = Math.floor((width - size) / 2);
  const yOffset = Math.floor((height - size) / 2);
  
  const output = new Uint8Array(size * size * 3);
  
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sourceIdx = ((y + yOffset) * width + (x + xOffset)) * 3;
      const targetIdx = (y * size + x) * 3;
      
      output[targetIdx] = data[sourceIdx];
      output[targetIdx + 1] = data[sourceIdx + 1];
      output[targetIdx + 2] = data[sourceIdx + 2];
    }
  }
  
  return { data: output, size };
}

/**
 * Calculate image quality metrics
 */
export function calculateImageQuality(data: Uint8Array): {
  brightness: number;
  contrast: number;
  sharpness: number;
  isBlurry: boolean;
} {
  let sumBrightness = 0;
  let minBrightness = 255;
  let maxBrightness = 0;
  
  // Calculate brightness and contrast
  for (let i = 0; i < data.length; i += 3) {
    const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
    sumBrightness += brightness;
    minBrightness = Math.min(minBrightness, brightness);
    maxBrightness = Math.max(maxBrightness, brightness);
  }
  
  const avgBrightness = sumBrightness / (data.length / 3);
  const contrast = maxBrightness - minBrightness;
  
  // Simple sharpness estimation (Laplacian variance)
  let sharpness = 0;
  const pixelCount = data.length / 3;
  const width = Math.sqrt(pixelCount);
  
  for (let i = width; i < pixelCount - width; i++) {
    if (i % width === 0 || i % width === width - 1) continue;
    
    const idx = i * 3;
    const center = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
    
    const left = (data[idx - 3] + data[idx - 2] + data[idx - 1]) / 3;
    const right = (data[idx + 3] + data[idx + 4] + data[idx + 5]) / 3;
    const top = (data[idx - width * 3] + data[idx - width * 3 + 1] + data[idx - width * 3 + 2]) / 3;
    const bottom = (data[idx + width * 3] + data[idx + width * 3 + 1] + data[idx + width * 3 + 2]) / 3;
    
    const laplacian = Math.abs(4 * center - left - right - top - bottom);
    sharpness += laplacian * laplacian;
  }
  
  sharpness = Math.sqrt(sharpness / pixelCount);
  
  return {
    brightness: avgBrightness / 255,
    contrast: contrast / 255,
    sharpness: sharpness / 255,
    isBlurry: sharpness < 0.1
  };
}

/**
 * Validate image before processing
 */
export function validateImage(imagePath: string): Promise<{
  isValid: boolean;
  error?: string;
}> {
  return new Promise(async (resolve) => {
    try {
      // Check if file exists
      const exists = await RNFS.exists(imagePath);
      if (!exists) {
        resolve({ isValid: false, error: 'Image file not found' });
        return;
      }
      
      // Check file size (max 10MB)
      const stat = await RNFS.stat(imagePath);
      if (stat.size > 10 * 1024 * 1024) {
        resolve({ isValid: false, error: 'Image file too large (max 10MB)' });
        return;
      }
      
      // Check dimensions
      const dimensions = await getImageDimensions(imagePath);
      if (dimensions.width < 224 || dimensions.height < 224) {
        resolve({ isValid: false, error: 'Image resolution too low (min 224x224)' });
        return;
      }
      
      resolve({ isValid: true });
    } catch (error) {
      resolve({ isValid: false, error: 'Failed to validate image' });
    }
  });
}