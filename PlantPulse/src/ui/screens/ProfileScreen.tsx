import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { CameraScreenNavigationProp } from '../../types';

const ProfileScreen: React.FC = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>();

  const menuItems = [
    {
      icon: 'cog',
      title: 'Settings',
      subtitle: 'App preferences and camera settings',
      onPress: () => navigation.navigate('Settings'),
    },
    {
      icon: 'information',
      title: 'About PlantPulse',
      subtitle: 'Version 1.0.0',
      onPress: () => {},
    },
    {
      icon: 'help-circle',
      title: 'Help & Support',
      subtitle: 'FAQs and troubleshooting',
      onPress: () => {},
    },
    {
      icon: 'share-variant',
      title: 'Share App',
      subtitle: 'Recommend PlantPulse to friends',
      onPress: () => {},
    },
  ];

  const stats = [
    { label: 'Plants Monitored', value: '0' },
    { label: 'Scans Completed', value: '0' },
    { label: 'Issues Detected', value: '0' },
    { label: 'Days Active', value: '0' },
  ];

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.avatar}>
          <Icon name="account-circle" size={80} color="#4CAF50" />
        </View>
        <Text style={styles.userName}>Plant Parent</Text>
        <Text style={styles.userEmail}>Keeping plants healthy with thermal imaging</Text>
      </View>

      <View style={styles.statsContainer}>
        {stats.map((stat, index) => (
          <View key={index} style={styles.statItem}>
            <Text style={styles.statValue}>{stat.value}</Text>
            <Text style={styles.statLabel}>{stat.label}</Text>
          </View>
        ))}
      </View>

      <View style={styles.menuContainer}>
        {menuItems.map((item, index) => (
          <TouchableOpacity
            key={index}
            style={styles.menuItem}
            onPress={item.onPress}>
            <View style={styles.menuIcon}>
              <Icon name={item.icon} size={24} color="#4CAF50" />
            </View>
            <View style={styles.menuContent}>
              <Text style={styles.menuTitle}>{item.title}</Text>
              <Text style={styles.menuSubtitle}>{item.subtitle}</Text>
            </View>
            <Icon name="chevron-right" size={24} color="#ccc" />
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 32,
    backgroundColor: '#fff',
  },
  avatar: {
    marginBottom: 16,
  },
  userName: {
    fontSize: 24,
    fontWeight: '600',
    color: '#333',
  },
  userEmail: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    marginTop: 16,
    paddingVertical: 20,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#e0e0e0',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '600',
    color: '#4CAF50',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  menuContainer: {
    marginTop: 16,
    backgroundColor: '#fff',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  menuIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#E8F5E9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  menuContent: {
    flex: 1,
  },
  menuTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  menuSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
});

export default ProfileScreen;