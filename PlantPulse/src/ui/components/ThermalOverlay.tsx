import React, { useMemo } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  ImageBackground,
} from 'react-native';
import Svg, { Rect, Circle, Line } from 'react-native-svg';
import { ThermalFrame } from '../../types/thermal';

interface ThermalOverlayProps {
  thermalFrame: ThermalFrame;
  width?: number;
  height?: number;
  colorMap?: 'iron' | 'rainbow' | 'grayscale' | 'hot';
  opacity?: number;
  showHotspots?: boolean;
  showGrid?: boolean;
}

const ThermalOverlay: React.FC<ThermalOverlayProps> = ({
  thermalFrame,
  width = Dimensions.get('window').width,
  height = (Dimensions.get('window').width * 192) / 256,
  colorMap = 'iron',
  opacity = 1,
  showHotspots = true,
  showGrid = false,
}) => {

  const getColorForTemperature = (
    temp: number,
    min: number,
    max: number
  ): string => {
    const normalized = (temp - min) / (max - min);
    
    switch (colorMap) {
      case 'iron':
        return getIronColor(normalized);
      case 'rainbow':
        return getRainbowColor(normalized);
      case 'hot':
        return getHotColor(normalized);
      case 'grayscale':
      default:
        const gray = Math.floor(normalized * 255);
        return `rgb(${gray}, ${gray}, ${gray})`;
    }
  };

  const getIronColor = (value: number): string => {
    // Iron color palette (black -> purple -> red -> yellow -> white)
    const r = Math.min(255, Math.floor(value * 765));
    const g = Math.min(255, Math.max(0, Math.floor(value * 765 - 255)));
    const b = Math.min(255, Math.max(0, Math.floor(value * 765 - 510)));
    
    if (value < 0.33) {
      // Black to purple
      return `rgb(${Math.floor(value * 3 * 128)}, 0, ${Math.floor(value * 3 * 255)})`;
    } else if (value < 0.66) {
      // Purple to red
      const t = (value - 0.33) * 3;
      return `rgb(${Math.floor(128 + t * 127)}, 0, ${Math.floor(255 * (1 - t))})`;
    } else {
      // Red to yellow to white
      const t = (value - 0.66) * 3;
      return `rgb(255, ${Math.floor(t * 255)}, ${Math.floor(t * 255)})`;
    }
  };

  const getRainbowColor = (value: number): string => {
    // HSL color space for smooth rainbow
    const hue = (1 - value) * 240; // 240 (blue) to 0 (red)
    return `hsl(${hue}, 100%, 50%)`;
  };

  const getHotColor = (value: number): string => {
    // Hot color palette (black -> red -> yellow -> white)
    if (value < 0.33) {
      const r = Math.floor(value * 3 * 255);
      return `rgb(${r}, 0, 0)`;
    } else if (value < 0.66) {
      const g = Math.floor((value - 0.33) * 3 * 255);
      return `rgb(255, ${g}, 0)`;
    } else {
      const b = Math.floor((value - 0.66) * 3 * 255);
      return `rgb(255, 255, ${b})`;
    }
  };

  const thermalPixels = useMemo(() => {
    const { temperatureData } = thermalFrame;
    const dataWidth = 256;
    const dataHeight = 192;
    
    // Find min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    
    for (let i = 0; i < temperatureData.length; i++) {
      if (!isNaN(temperatureData[i]) && isFinite(temperatureData[i])) {
        min = Math.min(min, temperatureData[i]);
        max = Math.max(max, temperatureData[i]);
      }
    }
    
    // Create downsampled pixel data for performance
    const sampleRate = 4; // Sample every 4th pixel
    const pixels: { x: number; y: number; width: number; height: number; color: string }[] = [];
    const pixelWidth = (width / dataWidth) * sampleRate;
    const pixelHeight = (height / dataHeight) * sampleRate;
    
    for (let y = 0; y < dataHeight; y += sampleRate) {
      for (let x = 0; x < dataWidth; x += sampleRate) {
        const index = y * dataWidth + x;
        const temp = temperatureData[index];
        const color = getColorForTemperature(temp, min, max);
        
        pixels.push({
          x: (x / dataWidth) * width,
          y: (y / dataHeight) * height,
          width: pixelWidth,
          height: pixelHeight,
          color,
        });
      }
    }
    
    return { pixels, min, max };
  }, [thermalFrame, width, height, colorMap]);

  const findHotspots = () => {
    const { temperatureData } = thermalFrame;
    const dataWidth = 256;
    const dataHeight = 192;
    const hotspots: { x: number; y: number; temp: number }[] = [];
    
    // Calculate average temperature
    let sum = 0;
    let count = 0;
    for (let i = 0; i < temperatureData.length; i++) {
      if (!isNaN(temperatureData[i]) && isFinite(temperatureData[i])) {
        sum += temperatureData[i];
        count++;
      }
    }
    const avg = sum / count;
    const threshold = avg + 5; // 5°C above average
    
    // Find hotspots
    for (let y = 0; y < dataHeight; y++) {
      for (let x = 0; x < dataWidth; x++) {
        const index = y * dataWidth + x;
        const temp = temperatureData[index];
        
        if (temp > threshold) {
          hotspots.push({
            x: (x / dataWidth) * width,
            y: (y / dataHeight) * height,
            temp,
          });
        }
      }
    }
    
    return hotspots;
  };

  if (!thermalFrame || !thermalFrame.temperatureData) {
    return null;
  }

  const hotspots = showHotspots ? findHotspots() : [];

  return (
    <View style={[styles.container, { width, height }]}>
      <Svg width={width} height={height} style={StyleSheet.absoluteFillObject}>
        {/* Render thermal pixels */}
        {thermalPixels.pixels.map((pixel, index) => (
          <Rect
            key={index}
            x={pixel.x}
            y={pixel.y}
            width={pixel.width}
            height={pixel.height}
            fill={pixel.color}
            opacity={opacity}
          />
        ))}
        
        {/* Render hotspot indicators */}
        {hotspots.map((hotspot, index) => (
          <Circle
            key={`hotspot-${index}`}
            cx={hotspot.x}
            cy={hotspot.y}
            r={10}
            stroke="red"
            strokeWidth={2}
            fill="none"
            opacity={0.7}
          />
        ))}
        
        {/* Render grid overlay */}
        {showGrid && (
          <>
            {Array.from({ length: 10 }).map((_, i) => {
              const x = (i + 1) * (width / 10);
              return (
                <Line
                  key={`grid-v-${i}`}
                  x1={x}
                  y1={0}
                  x2={x}
                  y2={height}
                  stroke="rgba(255, 255, 255, 0.2)"
                  strokeWidth={1}
                />
              );
            })}
            {Array.from({ length: 8 }).map((_, i) => {
              const y = (i + 1) * (height / 8);
              return (
                <Line
                  key={`grid-h-${i}`}
                  x1={0}
                  y1={y}
                  x2={width}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.2)"
                  strokeWidth={1}
                />
              );
            })}
          </>
        )}
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'transparent',
  },
});

export default ThermalOverlay;