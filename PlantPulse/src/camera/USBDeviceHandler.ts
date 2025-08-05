import { NativeModules, Platform } from 'react-native';

const { USBModule } = NativeModules;

export class USBDeviceHandler {
  async requestUSBPermission(): Promise<boolean> {
    if (Platform.OS === 'android') {
      try {
        return await USBModule.requestPermission();
      } catch (error) {
        console.error('Failed to request USB permission:', error);
        return false;
      }
    }
    
    // iOS handles USB permissions differently
    if (Platform.OS === 'ios') {
      // iOS requires External Accessory framework configuration
      // This is handled in Info.plist
      return true;
    }
    
    return false;
  }

  async listUSBDevices(): Promise<any[]> {
    try {
      if (Platform.OS === 'android') {
        return await USBModule.getDeviceList();
      }
      
      if (Platform.OS === 'ios') {
        // iOS USB device enumeration
        return await USBModule.getConnectedAccessories();
      }
      
      return [];
    } catch (error) {
      console.error('Failed to list USB devices:', error);
      return [];
    }
  }

  async openDevice(deviceId: string): Promise<boolean> {
    try {
      return await USBModule.openDevice(deviceId);
    } catch (error) {
      console.error('Failed to open USB device:', error);
      return false;
    }
  }

  async closeDevice(deviceId: string): Promise<void> {
    try {
      await USBModule.closeDevice(deviceId);
    } catch (error) {
      console.error('Failed to close USB device:', error);
    }
  }

  async readData(
    deviceId: string,
    endpoint: number,
    length: number,
    timeout: number = 1000
  ): Promise<Uint8Array> {
    try {
      const data = await USBModule.bulkTransfer(
        deviceId,
        endpoint,
        length,
        timeout
      );
      return new Uint8Array(data);
    } catch (error) {
      console.error('Failed to read USB data:', error);
      throw error;
    }
  }

  async writeData(
    deviceId: string,
    endpoint: number,
    data: Uint8Array,
    timeout: number = 1000
  ): Promise<number> {
    try {
      return await USBModule.bulkTransfer(
        deviceId,
        endpoint,
        Array.from(data),
        timeout
      );
    } catch (error) {
      console.error('Failed to write USB data:', error);
      throw error;
    }
  }

  async claimInterface(deviceId: string, interfaceNumber: number): Promise<boolean> {
    try {
      return await USBModule.claimInterface(deviceId, interfaceNumber);
    } catch (error) {
      console.error('Failed to claim USB interface:', error);
      return false;
    }
  }

  async releaseInterface(deviceId: string, interfaceNumber: number): Promise<void> {
    try {
      await USBModule.releaseInterface(deviceId, interfaceNumber);
    } catch (error) {
      console.error('Failed to release USB interface:', error);
    }
  }

  async setConfiguration(deviceId: string, configuration: number): Promise<boolean> {
    try {
      return await USBModule.setConfiguration(deviceId, configuration);
    } catch (error) {
      console.error('Failed to set USB configuration:', error);
      return false;
    }
  }

  subscribeToDeviceEvents(
    onConnect: (device: any) => void,
    onDisconnect: (deviceId: string) => void
  ): () => void {
    // This would be implemented with native event emitters
    // For now, return a dummy unsubscribe function
    return () => {};
  }
}