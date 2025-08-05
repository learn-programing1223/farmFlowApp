import { StackNavigationProp } from '@react-navigation/stack';
import { RouteProp } from '@react-navigation/native';
import { PlantProfile } from './plant';

export type RootStackParamList = {
  Main: undefined;
  PlantDetail: { plantId: string };
  Analysis: { plantId: string; analysisId: string };
  AddPlant: undefined;
  Settings: undefined;
  Calibration: undefined;
};

export type MainTabParamList = {
  Camera: undefined;
  Plants: undefined;
  History: undefined;
  Profile: undefined;
};

export type CameraScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  'Main'
>;

export type PlantDetailScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  'PlantDetail'
>;

export type PlantDetailScreenRouteProp = RouteProp<
  RootStackParamList,
  'PlantDetail'
>;

export type AnalysisScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  'Analysis'
>;

export type AnalysisScreenRouteProp = RouteProp<
  RootStackParamList,
  'Analysis'
>;