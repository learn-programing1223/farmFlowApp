// Temperature thresholds based on research
export const THERMAL_CONSTANTS = {
  // Water stress thresholds
  WATER_STRESS: {
    CWSI_THRESHOLD: 0.36, // Above this indicates stress
    LEAF_TEMP_ELEVATION_MIN: 1.5, // °C above baseline
    LEAF_TEMP_ELEVATION_MAX: 6.0, // °C above baseline
  },
  
  // Healthy plant indicators
  HEALTHY_PLANT: {
    TEMP_BELOW_AMBIENT_MIN: 6, // °C below ambient
    TEMP_BELOW_AMBIENT_MAX: 15, // °C below ambient
  },
  
  // Disease signatures
  DISEASE_SIGNATURES: {
    BIOTROPHIC_FUNGI: {
      MIN: -2.5, // K temperature change
      MAX: 2.0,
    },
    NECROTROPHIC_FUNGI: {
      MIN: -5.0, // K temperature change
      MAX: 9.0,
    },
    BACTERIAL: {
      INITIAL_COOLING: -2.0, // K
      LATER_HEATING: 3.0, // K
    },
  },
  
  // Camera specifications
  CAMERA_SPECS: {
    INFIRAY_P2_PRO: {
      RESOLUTION: { WIDTH: 256, HEIGHT: 192 },
      SENSITIVITY: 0.04, // 40mK
      TEMP_RANGE: { MIN: -20, MAX: 600 },
    },
    TOPDON_TC002C: {
      RESOLUTION: { WIDTH: 256, HEIGHT: 192 },
      ENHANCED_RESOLUTION: { WIDTH: 512, HEIGHT: 384 },
      TEMP_RANGE: { MIN: -20, MAX: 600 },
    },
  },
  
  // Processing parameters
  PROCESSING: {
    STABLE_FPS: 5,
    ACTIVE_FPS: 25,
    TEMP_CHANGE_THRESHOLD: 2.0, // °C to trigger active mode
    GPU_INFERENCE_TARGET: 100, // ms
  },
  
  // Default calibration values
  DEFAULT_CALIBRATION: {
    EMISSIVITY: 0.95, // For plant leaves
    REFLECTED_TEMP: 20.0, // °C
    DISTANCE: 0.5, // meters
    HUMIDITY: 50, // %
  },
};