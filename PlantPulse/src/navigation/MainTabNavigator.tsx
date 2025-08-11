import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { MainTabParamList } from '../types';
import UnifiedCameraScreen from '../ui/screens/UnifiedCameraScreen';
import PlantsScreen from '../ui/screens/PlantsScreen';
import HistoryScreen from '../ui/screens/HistoryScreen';
import ProfileScreen from '../ui/screens/ProfileScreen';

const Tab = createBottomTabNavigator<MainTabParamList>();

const MainTabNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Camera':
              iconName = focused ? 'camera' : 'camera-outline';
              break;
            case 'Plants':
              iconName = focused ? 'flower' : 'flower-outline';
              break;
            case 'History':
              iconName = focused ? 'history' : 'history';
              break;
            case 'Profile':
              iconName = focused ? 'account' : 'account-outline';
              break;
            default:
              iconName = 'circle';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#4CAF50',
        tabBarInactiveTintColor: 'gray',
        headerStyle: {
          backgroundColor: '#4CAF50',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      })}>
      <Tab.Screen
        name="Camera"
        component={UnifiedCameraScreen}
        options={{ title: 'Scan' }}
      />
      <Tab.Screen
        name="Plants"
        component={PlantsScreen}
        options={{ title: 'My Plants' }}
      />
      <Tab.Screen
        name="History"
        component={HistoryScreen}
        options={{ title: 'History' }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{ title: 'Profile' }}
      />
    </Tab.Navigator>
  );
};

export default MainTabNavigator;