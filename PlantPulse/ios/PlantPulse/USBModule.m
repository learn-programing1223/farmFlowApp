#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(USBModule, NSObject)

RCT_EXTERN_METHOD(getConnectedAccessories:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(requestPermission:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(openDevice:(NSString *)deviceId
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(closeDevice:(NSString *)deviceId
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(bulkTransfer:(NSString *)deviceId
                  endpoint:(nonnull NSNumber *)endpoint
                  data:(NSArray *)data
                  timeout:(nonnull NSNumber *)timeout
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(claimInterface:(NSString *)deviceId
                  interfaceNumber:(nonnull NSNumber *)interfaceNumber
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(releaseInterface:(NSString *)deviceId
                  interfaceNumber:(nonnull NSNumber *)interfaceNumber
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(setConfiguration:(NSString *)deviceId
                  configuration:(nonnull NSNumber *)configuration
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end