import { NativeModules, NativeEventEmitter, Platform } from 'react-native';
import { ThermalDevice, ThermalFrame, ThermalCalibration } from '../types/thermal';
import { TemperatureParser } from './TemperatureParser';
import { USBDeviceHandler } from './USBDeviceHandler';

const { UVCCameraModule } = NativeModules;

export class ThermalCameraManager {
  private static instance: ThermalCameraManager;
  private eventEmitter: NativeEventEmitter;
  private temperatureParser: TemperatureParser;
  private usbHandler: USBDeviceHandler;
  private currentDevice: ThermalDevice | null = null;
  private frameCallback: ((frame: ThermalFrame) => void) | null = null;
  private calibration: ThermalCalibration = {
    emissivity: 0.95,
    reflectedTemperature: 20,
    distance: 0.5,
    humidity: 50,
    ambientTemperature: 22,
  };

  private constructor() {
    this.eventEmitter = new NativeEventEmitter(UVCCameraModule);
    this.temperatureParser = new TemperatureParser();
    this.usbHandler = new USBDeviceHandler();
    this.setupEventListeners();
  }

  static getInstance(): ThermalCameraManager {
    if (!ThermalCameraManager.instance) {
      ThermalCameraManager.instance = new ThermalCameraManager();
    }
    return ThermalCameraManager.instance;
  }

  private setupEventListeners() {
    this.eventEmitter.addListener('onDeviceConnected', this.handleDeviceConnected);
    this.eventEmitter.addListener('onDeviceDisconnected', this.handleDeviceDisconnected);
    this.eventEmitter.addListener('onFrameAvailable', this.handleFrameAvailable);
    this.eventEmitter.addListener('onError', this.handleError);
  }

  private handleDeviceConnected = (device: any) => {
    console.log('Thermal camera connected:', device);
    this.currentDevice = this.identifyDevice(device);
  };

  private handleDeviceDisconnected = () => {
    console.log('Thermal camera disconnected');
    this.currentDevice = null;
  };

  private handleFrameAvailable = (frameData: any) => {
    if (!this.frameCallback) return;

    const thermalFrame = this.processFrame(frameData);
    if (thermalFrame) {
      this.frameCallback(thermalFrame);
    }
  };

  private handleError = (error: any) => {
    console.error('Thermal camera error:', error);
  };

  private identifyDevice(deviceInfo: any): ThermalDevice {
    let model: ThermalDevice['model'] = 'Unknown';
    
    if (deviceInfo.vendorId === 0x0BDA && deviceInfo.productId === 0x5830) {
      model = 'InfiRay_P2_Pro';
    } else if (deviceInfo.vendorId === 0x0BDA && deviceInfo.productId === 0x5832) {
      model = 'TOPDON_TC002C';
    }

    return {
      id: deviceInfo.deviceId || 'thermal_camera_1',
      name: deviceInfo.name || model,
      model,
      resolution: {
        width: 256,
        height: 192,
      },
      temperatureRange: {
        min: -20,
        max: 550,
      },
      isConnected: true,
    };
  }

  private processFrame(frameData: any): ThermalFrame | null {
    try {
      const temperatureData = this.temperatureParser.parseFrame(
        frameData.data,
        this.currentDevice?.model || 'Unknown',
        this.calibration
      );

      return {
        temperatureData,
        timestamp: Date.now(),
        deviceId: this.currentDevice?.id || 'unknown',
        calibrationOffset: this.calibration.reflectedTemperature,
      };
    } catch (error) {
      console.error('Failed to process thermal frame:', error);
      return null;
    }
  }

  async initialize(): Promise<boolean> {
    try {
      if (Platform.OS === 'ios') {
        return await this.usbHandler.requestUSBPermission();
      }
      return true;
    } catch (error) {
      console.error('Failed to initialize thermal camera:', error);
      return false;
    }
  }

  async startCapture(callback: (frame: ThermalFrame) => void): Promise<void> {
    this.frameCallback = callback;
    
    if (!this.currentDevice) {
      const devices = await this.scanForDevices();
      if (devices.length === 0) {
        throw new Error('No thermal camera detected');
      }
      await this.connectToDevice(devices[0]);
    }

    await UVCCameraModule.startCapture({
      width: 256,
      height: 384, // Full frame including temperature data
      fps: 25,
    });
  }

  async stopCapture(): Promise<void> {
    this.frameCallback = null;
    await UVCCameraModule.stopCapture();
  }

  async scanForDevices(): Promise<ThermalDevice[]> {
    const devices = await this.usbHandler.listUSBDevices();
    return devices
      .filter((device: any) => this.isThermalCamera(device))
      .map((device: any) => this.identifyDevice(device));
  }

  private isThermalCamera(device: any): boolean {
    const thermalVendorIds = [0x0BDA]; // Realtek bridge chip
    const thermalProductIds = [0x5830, 0x5832];
    
    return thermalVendorIds.includes(device.vendorId) &&
           thermalProductIds.includes(device.productId);
  }

  async connectToDevice(device: ThermalDevice): Promise<void> {
    await UVCCameraModule.connectDevice(device.id);
    this.currentDevice = device;
  }

  async disconnectDevice(): Promise<void> {
    await UVCCameraModule.disconnectDevice();
    this.currentDevice = null;
  }

  setCalibration(calibration: Partial<ThermalCalibration>): void {
    this.calibration = { ...this.calibration, ...calibration };
  }

  getCalibration(): ThermalCalibration {
    return { ...this.calibration };
  }

  getCurrentDevice(): ThermalDevice | null {
    return this.currentDevice;
  }

  dispose(): void {
    this.eventEmitter.removeAllListeners();
    this.stopCapture();
  }
}