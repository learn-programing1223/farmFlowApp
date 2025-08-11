import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  Text,
  Alert,
  ActivityIndicator,
  Dimensions,
  Modal,
  ScrollView,
  Image,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  PhotoFile,
} from 'react-native-vision-camera';
import { rgbDiseaseModel, DiseaseDetectionResult } from '../../ml/RGBDiseaseModel';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export const RGBCameraScreen: React.FC = () => {
  const camera = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
  const [results, setResults] = useState<DiseaseDetectionResult | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  // Load model on mount
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      await rgbDiseaseModel.loadModel();
      setModelLoaded(true);
    } catch (error) {
      Alert.alert('Error', 'Failed to load disease detection model');
      console.error(error);
    }
  };

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const takePicture = useCallback(async () => {
    if (!camera.current || isProcessing) return;

    setIsProcessing(true);
    try {
      const photo = await camera.current.takePhoto({
        qualityPrioritization: 'balanced',
        enableAutoStabilization: true,
      });
      
      setCapturedPhoto(`file://${photo.path}`);
      await analyzePhoto(photo);
    } catch (error) {
      console.error('Failed to take photo:', error);
      Alert.alert('Error', 'Failed to capture photo');
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  const analyzePhoto = async (photo: PhotoFile) => {
    try {
      // In a real implementation, you would:
      // 1. Load the image file
      // 2. Convert to Uint8Array
      // 3. Pass to the model
      
      // For now, we'll simulate the analysis
      // In production, you'd use react-native-fs or similar to read the image
      
      // Simulated results for demonstration
      const mockImageData = new Uint8Array(224 * 224 * 4);
      const result = await rgbDiseaseModel.detectDisease(
        mockImageData,
        224,
        224
      );
      
      setResults(result);
      setShowResults(true);
    } catch (error) {
      console.error('Analysis failed:', error);
      Alert.alert('Error', 'Failed to analyze plant health');
    }
  };

  const retakePhoto = () => {
    setCapturedPhoto(null);
    setResults(null);
    setShowResults(false);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'mild': return '#4CAF50';
      case 'moderate': return '#FF9800';
      case 'severe': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  if (!device) {
    return (
      <View style={styles.container}>
        <Text>No camera device found</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          Camera permission is required to detect plant diseases
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {!capturedPhoto ? (
        <>
          <Camera
            ref={camera}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
            photo={true}
            enableZoomGesture={true}
          />
          
          <View style={styles.overlay}>
            <View style={styles.header}>
              <Text style={styles.headerText}>Plant Disease Detection</Text>
              {modelLoaded && (
                <View style={styles.modelStatus}>
                  <Icon name="check-circle" size={20} color="#4CAF50" />
                  <Text style={styles.statusText}>Model Ready</Text>
                </View>
              )}
            </View>

            <View style={styles.guidanceFrame}>
              <View style={[styles.corner, styles.topLeft]} />
              <View style={[styles.corner, styles.topRight]} />
              <View style={[styles.corner, styles.bottomLeft]} />
              <View style={[styles.corner, styles.bottomRight]} />
              <Text style={styles.guidanceText}>
                Position the affected plant part within the frame
              </Text>
            </View>

            <TouchableOpacity
              style={[styles.captureButton, isProcessing && styles.buttonDisabled]}
              onPress={takePicture}
              disabled={isProcessing || !modelLoaded}
            >
              {isProcessing ? (
                <ActivityIndicator color="#fff" size="large" />
              ) : (
                <Icon name="camera" size={40} color="#fff" />
              )}
            </TouchableOpacity>
          </View>
        </>
      ) : (
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedPhoto }} style={styles.preview} />
          
          <TouchableOpacity style={styles.retakeButton} onPress={retakePhoto}>
            <Icon name="refresh" size={24} color="#fff" />
            <Text style={styles.retakeText}>Retake</Text>
          </TouchableOpacity>
        </View>
      )}

      <Modal
        visible={showResults}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowResults(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <ScrollView showsVerticalScrollIndicator={false}>
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => setShowResults(false)}
              >
                <Icon name="close" size={24} color="#333" />
              </TouchableOpacity>

              {results && (
                <>
                  <View style={styles.resultHeader}>
                    <Icon 
                      name={results.disease === 'Healthy' ? 'check-circle' : 'warning'}
                      size={60}
                      color={results.disease === 'Healthy' ? '#4CAF50' : '#FF9800'}
                    />
                    <Text style={styles.diseaseTitle}>{results.disease.replace('_', ' ')}</Text>
                    <View style={styles.confidenceContainer}>
                      <Text style={styles.confidenceLabel}>Confidence:</Text>
                      <Text style={styles.confidenceValue}>
                        {(results.confidence * 100).toFixed(1)}%
                      </Text>
                    </View>
                  </View>

                  {results.disease !== 'Healthy' && (
                    <>
                      <View style={[
                        styles.severityBadge,
                        { backgroundColor: getSeverityColor(results.severity) }
                      ]}>
                        <Text style={styles.severityText}>
                          {results.severity.toUpperCase()} SEVERITY
                        </Text>
                      </View>

                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>Description</Text>
                        <Text style={styles.description}>{results.additionalInfo}</Text>
                      </View>
                    </>
                  )}

                  <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Recommendations</Text>
                    {results.recommendations.map((rec, index) => (
                      <View key={index} style={styles.recommendation}>
                        <Icon name="check" size={20} color="#4CAF50" />
                        <Text style={styles.recommendationText}>{rec}</Text>
                      </View>
                    ))}
                  </View>

                  <TouchableOpacity
                    style={styles.actionButton}
                    onPress={() => {
                      setShowResults(false);
                      retakePhoto();
                    }}
                  >
                    <Text style={styles.actionButtonText}>Scan Another Plant</Text>
                  </TouchableOpacity>
                </>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#4CAF50',
    padding: 15,
    borderRadius: 8,
    marginHorizontal: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
  },
  header: {
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 20,
    paddingTop: 50,
  },
  headerText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  modelStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  statusText: {
    color: '#fff',
    marginLeft: 5,
  },
  guidanceFrame: {
    width: screenWidth * 0.8,
    height: screenWidth * 0.8,
    alignSelf: 'center',
    justifyContent: 'center',
  },
  corner: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderColor: '#4CAF50',
    borderWidth: 3,
  },
  topLeft: {
    top: 0,
    left: 0,
    borderRightWidth: 0,
    borderBottomWidth: 0,
  },
  topRight: {
    top: 0,
    right: 0,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderRightWidth: 0,
    borderTopWidth: 0,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderLeftWidth: 0,
    borderTopWidth: 0,
  },
  guidanceText: {
    color: '#fff',
    textAlign: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 8,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    marginBottom: 40,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  previewContainer: {
    flex: 1,
  },
  preview: {
    flex: 1,
    resizeMode: 'contain',
  },
  retakeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  retakeText: {
    color: '#fff',
    marginLeft: 5,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    maxHeight: screenHeight * 0.8,
  },
  closeButton: {
    alignSelf: 'flex-end',
    padding: 5,
  },
  resultHeader: {
    alignItems: 'center',
    marginVertical: 20,
  },
  diseaseTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 10,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
  },
  confidenceLabel: {
    fontSize: 16,
    color: '#666',
    marginRight: 5,
  },
  confidenceValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  severityBadge: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    alignSelf: 'center',
    marginBottom: 20,
  },
  severityText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  recommendation: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  recommendationText: {
    flex: 1,
    marginLeft: 10,
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  actionButton: {
    backgroundColor: '#4CAF50',
    padding: 15,
    borderRadius: 8,
    marginTop: 20,
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});