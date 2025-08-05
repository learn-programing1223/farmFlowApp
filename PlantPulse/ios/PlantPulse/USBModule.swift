import Foundation
import ExternalAccessory

@objc(USBModule)
class USBModule: NSObject {
  
  @objc
  func getConnectedAccessories(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    let accessories = EAAccessoryManager.shared().connectedAccessories
    
    let deviceList = accessories.map { accessory in
      return [
        "deviceId": accessory.serialNumber,
        "name": accessory.name,
        "manufacturer": accessory.manufacturer,
        "modelNumber": accessory.modelNumber,
        "firmwareRevision": accessory.firmwareRevision,
        "hardwareRevision": accessory.hardwareRevision,
        "protocolStrings": accessory.protocolStrings,
        "vendorId": extractVendorId(from: accessory),
        "productId": extractProductId(from: accessory)
      ]
    }
    
    resolve(deviceList)
  }
  
  @objc
  func requestPermission(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // iOS doesn't require explicit USB permissions like Android
    // But we can check if we have any thermal cameras connected
    let accessories = EAAccessoryManager.shared().connectedAccessories
    let thermalProtocols = ["com.infiray.p2pro", "com.topdon.tc002c", "com.flir.one"]
    
    let hasThermalCamera = accessories.contains { accessory in
      accessory.protocolStrings.contains { protocol in
        thermalProtocols.contains(protocol.lowercased())
      }
    }
    
    resolve(hasThermalCamera)
  }
  
  @objc
  func openDevice(_ deviceId: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    let accessories = EAAccessoryManager.shared().connectedAccessories
    
    if let accessory = accessories.first(where: { $0.serialNumber == deviceId }) {
      // Check if it's a supported thermal camera
      let supportedProtocols = ["com.infiray.p2pro", "com.topdon.tc002c"]
      
      if let protocol = accessory.protocolStrings.first(where: { supportedProtocols.contains($0) }) {
        let session = EASession(accessory: accessory, forProtocol: protocol)
        
        if session != nil {
          resolve(true)
        } else {
          reject("OPEN_FAILED", "Failed to open session with device", nil)
        }
      } else {
        reject("UNSUPPORTED_DEVICE", "Device protocol not supported", nil)
      }
    } else {
      reject("DEVICE_NOT_FOUND", "Device not found", nil)
    }
  }
  
  @objc
  func closeDevice(_ deviceId: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // Close any sessions associated with this device
    resolve(true)
  }
  
  @objc
  func bulkTransfer(_ deviceId: String, endpoint: NSNumber, data: NSArray, timeout: NSNumber, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // This would need actual implementation based on the thermal camera SDK
    // For now, return mock data
    resolve(data.count)
  }
  
  @objc
  func claimInterface(_ deviceId: String, interfaceNumber: NSNumber, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // iOS handles this automatically through External Accessory framework
    resolve(true)
  }
  
  @objc
  func releaseInterface(_ deviceId: String, interfaceNumber: NSNumber, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    resolve(true)
  }
  
  @objc
  func setConfiguration(_ deviceId: String, configuration: NSNumber, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    resolve(true)
  }
  
  private func extractVendorId(from accessory: EAAccessory) -> String {
    // Try to extract vendor ID from the accessory info
    // InfiRay and TOPDON use Realtek chips with vendor ID 0x0BDA
    if accessory.manufacturer.lowercased().contains("infiray") ||
       accessory.name.lowercased().contains("p2 pro") {
      return "0x0BDA"
    } else if accessory.manufacturer.lowercased().contains("topdon") ||
              accessory.name.lowercased().contains("tc002") {
      return "0x0BDA"
    }
    return "0x0000"
  }
  
  private func extractProductId(from accessory: EAAccessory) -> String {
    // Extract product ID based on known models
    if accessory.name.lowercased().contains("p2 pro") {
      return "0x5830"
    } else if accessory.name.lowercased().contains("tc002c") {
      return "0x5832"
    }
    return "0x0000"
  }
  
  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}