import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { RootStackParamList } from '../types';
import MainTabNavigator from './MainTabNavigator';
import PlantDetailScreen from '../ui/screens/PlantDetailScreen';
import AnalysisScreen from '../ui/screens/AnalysisScreen';
import AddPlantScreen from '../ui/screens/AddPlantScreen';
import SettingsScreen from '../ui/screens/SettingsScreen';
import CalibrationScreen from '../ui/screens/CalibrationScreen';

const Stack = createStackNavigator<RootStackParamList>();

const RootNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Main"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#4CAF50',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}>
        <Stack.Screen
          name="Main"
          component={MainTabNavigator}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="PlantDetail"
          component={PlantDetailScreen}
          options={{ title: 'Plant Details' }}
        />
        <Stack.Screen
          name="Analysis"
          component={AnalysisScreen}
          options={{ title: 'Health Analysis' }}
        />
        <Stack.Screen
          name="AddPlant"
          component={AddPlantScreen}
          options={{ title: 'Add New Plant' }}
        />
        <Stack.Screen
          name="Settings"
          component={SettingsScreen}
          options={{ title: 'Settings' }}
        />
        <Stack.Screen
          name="Calibration"
          component={CalibrationScreen}
          options={{ title: 'Camera Calibration' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default RootNavigator;