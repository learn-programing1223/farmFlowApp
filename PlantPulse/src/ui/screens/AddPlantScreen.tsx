import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { CameraScreenNavigationProp } from '../../types';
import { COMMON_PLANTS } from '../../constants';

const AddPlantScreen: React.FC = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>();
  const [nickname, setNickname] = useState('');
  const [selectedSpecies, setSelectedSpecies] = useState('');

  const handleSave = () => {
    // TODO: Implement plant saving logic
    navigation.goBack();
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.form}>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Plant Nickname</Text>
          <TextInput
            style={styles.input}
            value={nickname}
            onChangeText={setNickname}
            placeholder="E.g., Living Room Monstera"
            placeholderTextColor="#999"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Plant Species</Text>
          <View style={styles.speciesList}>
            {COMMON_PLANTS.map((plant) => (
              <TouchableOpacity
                key={plant.id}
                style={[
                  styles.speciesItem,
                  selectedSpecies === plant.id && styles.speciesItemSelected,
                ]}
                onPress={() => setSelectedSpecies(plant.id)}>
                <Icon
                  name="flower"
                  size={24}
                  color={selectedSpecies === plant.id ? '#4CAF50' : '#666'}
                />
                <View style={styles.speciesInfo}>
                  <Text
                    style={[
                      styles.speciesName,
                      selectedSpecies === plant.id && styles.speciesNameSelected,
                    ]}>
                    {plant.commonName}
                  </Text>
                  <Text style={styles.speciesScientific}>
                    {plant.scientificName}
                  </Text>
                </View>
                {selectedSpecies === plant.id && (
                  <Icon name="check-circle" size={20} color="#4CAF50" />
                )}
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <TouchableOpacity
          style={[
            styles.saveButton,
            (!nickname || !selectedSpecies) && styles.saveButtonDisabled,
          ]}
          onPress={handleSave}
          disabled={!nickname || !selectedSpecies}>
          <Text style={styles.saveButtonText}>Add Plant</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  form: {
    padding: 16,
  },
  inputGroup: {
    marginBottom: 24,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#333',
  },
  speciesList: {
    gap: 8,
  },
  speciesItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 8,
    padding: 12,
    gap: 12,
  },
  speciesItemSelected: {
    borderColor: '#4CAF50',
    backgroundColor: '#E8F5E9',
  },
  speciesInfo: {
    flex: 1,
  },
  speciesName: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  speciesNameSelected: {
    color: '#4CAF50',
  },
  speciesScientific: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 24,
  },
  saveButtonDisabled: {
    backgroundColor: '#ccc',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default AddPlantScreen;