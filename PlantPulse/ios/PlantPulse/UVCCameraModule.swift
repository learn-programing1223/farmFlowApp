import Foundation
import UIKit
import AVFoundation
import ExternalAccessory

@objc(UVCCameraModule)
class UVCCameraModule: RCTEventEmitter {
  
  private var captureSession: AVCaptureSession?
  private var videoOutput: AVCaptureVideoDataOutput?
  private var isCapturing = false
  private var connectedDevice: EAAccessory?
  
  override init() {
    super.init()
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(accessoryDidConnect),
      name: .EAAccessoryDidConnect,
      object: nil
    )
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(accessoryDidDisconnect),
      name: .EAAccessoryDidDisconnect,
      object: nil
    )
  }
  
  deinit {
    NotificationCenter.default.removeObserver(self)
  }
  
  override func supportedEvents() -> [String]! {
    return ["onDeviceConnected", "onDeviceDisconnected", "onFrameAvailable", "onError"]
  }
  
  override static func requiresMainQueueSetup() -> Bool {
    return true
  }
  
  @objc
  func initialize(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async {
      self.setupCaptureSession()
      resolve(true)
    }
  }
  
  @objc
  func startCapture(_ config: NSDictionary, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard let session = captureSession else {
      reject("NO_SESSION", "Capture session not initialized", nil)
      return
    }
    
    DispatchQueue.main.async {
      if !session.isRunning {
        session.startRunning()
        self.isCapturing = true
        resolve(true)
      } else {
        resolve(false)
      }
    }
  }
  
  @objc
  func stopCapture(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard let session = captureSession else {
      reject("NO_SESSION", "Capture session not initialized", nil)
      return
    }
    
    DispatchQueue.main.async {
      if session.isRunning {
        session.stopRunning()
        self.isCapturing = false
      }
      resolve(true)
    }
  }
  
  @objc
  func connectDevice(_ deviceId: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // Find the accessory with matching identifier
    let accessories = EAAccessoryManager.shared().connectedAccessories
    
    if let accessory = accessories.first(where: { $0.serialNumber == deviceId }) {
      self.connectedDevice = accessory
      
      // Open session for thermal data protocol
      let protocolString = accessory.protocolStrings.first ?? ""
      let session = EASession(accessory: accessory, forProtocol: protocolString)
      
      if let session = session {
        session.inputStream?.delegate = self
        session.outputStream?.delegate = self
        
        session.inputStream?.schedule(in: .current, forMode: .default)
        session.outputStream?.schedule(in: .current, forMode: .default)
        
        session.inputStream?.open()
        session.outputStream?.open()
        
        resolve(true)
      } else {
        reject("CONNECTION_FAILED", "Failed to open session with device", nil)
      }
    } else {
      reject("DEVICE_NOT_FOUND", "Device with ID \(deviceId) not found", nil)
    }
  }
  
  @objc
  func disconnectDevice(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // Close any open sessions
    if let device = connectedDevice {
      // Clean up sessions
      self.connectedDevice = nil
      resolve(true)
    } else {
      resolve(false)
    }
  }
  
  private func setupCaptureSession() {
    captureSession = AVCaptureSession()
    captureSession?.sessionPreset = .medium
    
    // Setup video output
    videoOutput = AVCaptureVideoDataOutput()
    videoOutput?.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]
    
    let queue = DispatchQueue(label: "thermal.camera.queue")
    videoOutput?.setSampleBufferDelegate(self, queue: queue)
    
    if let output = videoOutput {
      captureSession?.addOutput(output)
    }
  }
  
  @objc private func accessoryDidConnect(_ notification: Notification) {
    guard let accessory = notification.userInfo?[EAAccessoryKey] as? EAAccessory else { return }
    
    // Check if it's a thermal camera
    let thermalProtocols = ["com.infiray.p2pro", "com.topdon.tc002c", "com.flir.one"]
    let isThermalCamera = accessory.protocolStrings.contains { protocol in
      thermalProtocols.contains(protocol.lowercased())
    }
    
    if isThermalCamera {
      let deviceInfo: [String: Any] = [
        "deviceId": accessory.serialNumber,
        "name": accessory.name,
        "vendorId": accessory.manufacturer,
        "productId": accessory.modelNumber
      ]
      
      sendEvent(withName: "onDeviceConnected", body: deviceInfo)
    }
  }
  
  @objc private func accessoryDidDisconnect(_ notification: Notification) {
    sendEvent(withName: "onDeviceDisconnected", body: nil)
  }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension UVCCameraModule: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard isCapturing else { return }
    
    // Extract pixel buffer
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    
    // Process thermal data (this would need actual thermal camera SDK integration)
    let frameData = processThermalFrame(pixelBuffer)
    
    sendEvent(withName: "onFrameAvailable", body: frameData)
  }
  
  private func processThermalFrame(_ pixelBuffer: CVPixelBuffer) -> [String: Any] {
    // This is a placeholder - actual implementation would:
    // 1. Extract temperature data from the pixel buffer
    // 2. Parse according to camera model format
    // 3. Return temperature array
    
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    
    // Generate mock thermal data for testing
    var temperatureData: [Float] = []
    for _ in 0..<(256 * 192) {
      // Simulate temperature readings between 20-35°C
      let temp = Float.random(in: 20...35)
      temperatureData.append(temp)
    }
    
    return [
      "data": temperatureData,
      "width": 256,
      "height": 192,
      "timestamp": Date().timeIntervalSince1970 * 1000
    ]
  }
}

// MARK: - StreamDelegate for External Accessory
extension UVCCameraModule: StreamDelegate {
  func stream(_ aStream: Stream, handle eventCode: Stream.Event) {
    switch eventCode {
    case .hasBytesAvailable:
      handleIncomingData(from: aStream as! InputStream)
    case .hasSpaceAvailable:
      // Handle output if needed
      break
    case .errorOccurred:
      sendEvent(withName: "onError", body: ["error": "Stream error occurred"])
    case .endEncountered:
      sendEvent(withName: "onError", body: ["error": "Stream ended"])
    default:
      break
    }
  }
  
  private func handleIncomingData(from inputStream: InputStream) {
    let bufferSize = 1024
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
    defer { buffer.deallocate() }
    
    while inputStream.hasBytesAvailable {
      let bytesRead = inputStream.read(buffer, maxLength: bufferSize)
      if bytesRead > 0 {
        // Process thermal camera data
        // This would need to be implemented based on specific camera protocols
      }
    }
  }
}