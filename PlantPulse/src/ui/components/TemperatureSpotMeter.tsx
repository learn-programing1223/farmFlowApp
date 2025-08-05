import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  PanResponder,
  Animated,
  Dimensions,
} from 'react-native';
import Svg, { Circle, Line, Text as SvgText } from 'react-native-svg';
import { ThermalFrame } from '../../types/thermal';
import { TemperatureParser } from '../../camera/TemperatureParser';

interface TemperatureSpotMeterProps {
  thermalFrame: ThermalFrame;
  width?: number;
  height?: number;
  onTemperatureRead?: (temp: number, x: number, y: number) => void;
}

const TemperatureSpotMeter: React.FC<TemperatureSpotMeterProps> = ({
  thermalFrame,
  width = Dimensions.get('window').width,
  height = (Dimensions.get('window').width * 192) / 256,
  onTemperatureRead,
}) => {
  const [spotPosition, setSpotPosition] = useState({ x: width / 2, y: height / 2 });
  const [spotTemperature, setSpotTemperature] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const pan = React.useRef(new Animated.ValueXY({ x: width / 2, y: height / 2 })).current;
  const temperatureParser = new TemperatureParser();

  const panResponder = React.useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        setIsDragging(true);
      },
      onPanResponderMove: (evt, gestureState) => {
        const newX = Math.max(0, Math.min(width, gestureState.moveX));
        const newY = Math.max(0, Math.min(height, gestureState.moveY));
        
        pan.setValue({ x: newX, y: newY });
        setSpotPosition({ x: newX, y: newY });
        
        // Calculate temperature at this position
        const temp = getTemperatureAtPosition(newX, newY);
        setSpotTemperature(temp);
        
        if (onTemperatureRead && temp !== null) {
          onTemperatureRead(temp, newX, newY);
        }
      },
      onPanResponderRelease: () => {
        setIsDragging(false);
      },
    })
  ).current;

  const getTemperatureAtPosition = (x: number, y: number): number | null => {
    if (!thermalFrame || !thermalFrame.temperatureData) return null;
    
    // Convert screen coordinates to data coordinates
    const dataX = Math.floor((x / width) * 256);
    const dataY = Math.floor((y / height) * 192);
    
    // Get temperature using the parser's spot temperature method
    const temp = temperatureParser.extractSpotTemperature(
      thermalFrame.temperatureData,
      dataX,
      dataY,
      3 // radius
    );
    
    return temp;
  };

  React.useEffect(() => {
    // Update temperature when frame changes
    const temp = getTemperatureAtPosition(spotPosition.x, spotPosition.y);
    setSpotTemperature(temp);
  }, [thermalFrame]);

  return (
    <View style={[styles.container, { width, height }]} pointerEvents="box-none">
      <Animated.View
        style={[
          styles.spotContainer,
          {
            transform: [
              { translateX: pan.x },
              { translateY: pan.y },
              { translateX: -25 }, // Center the spot
              { translateY: -25 },
            ],
          },
        ]}
        {...panResponder.panHandlers}
      >
        <Svg width={50} height={50}>
          {/* Crosshair */}
          <Line
            x1={25}
            y1={0}
            x2={25}
            y2={20}
            stroke="#fff"
            strokeWidth={2}
          />
          <Line
            x1={25}
            y1={30}
            x2={25}
            y2={50}
            stroke="#fff"
            strokeWidth={2}
          />
          <Line
            x1={0}
            y1={25}
            x2={20}
            y2={25}
            stroke="#fff"
            strokeWidth={2}
          />
          <Line
            x1={30}
            y1={25}
            x2={50}
            y2={25}
            stroke="#fff"
            strokeWidth={2}
          />
          
          {/* Center circle */}
          <Circle
            cx={25}
            cy={25}
            r={8}
            stroke="#fff"
            strokeWidth={2}
            fill="none"
          />
          <Circle
            cx={25}
            cy={25}
            r={3}
            fill="#ff0000"
          />
        </Svg>
        
        {/* Temperature display */}
        {spotTemperature !== null && (
          <View style={[
            styles.temperatureDisplay,
            isDragging && styles.temperatureDisplayActive
          ]}>
            <Text style={styles.temperatureText}>
              {spotTemperature.toFixed(1)}°C
            </Text>
          </View>
        )}
      </Animated.View>
      
      {/* Instructions */}
      {!isDragging && (
        <View style={styles.instructions}>
          <Text style={styles.instructionText}>
            Drag crosshair to measure temperature
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  spotContainer: {
    position: 'absolute',
    width: 50,
    height: 50,
  },
  temperatureDisplay: {
    position: 'absolute',
    top: -35,
    left: -15,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    minWidth: 80,
    alignItems: 'center',
  },
  temperatureDisplayActive: {
    backgroundColor: 'rgba(255, 0, 0, 0.9)',
  },
  temperatureText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  instructions: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  instructionText: {
    color: '#fff',
    fontSize: 14,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
});

export default TemperatureSpotMeter;