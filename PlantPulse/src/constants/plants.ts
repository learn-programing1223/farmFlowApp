import { PlantSpecies } from '../types';

export const COMMON_PLANTS: PlantSpecies[] = [
  {
    id: 'monstera_deliciosa',
    scientificName: 'Monstera deliciosa',
    commonName: 'Swiss Cheese Plant',
    optimalTemperature: { min: 18, max: 27 },
    optimalHumidity: { min: 60, max: 80 },
    waterStressThreshold: 0.4,
    diseaseResistance: {
      bacterial: 0.7,
      fungal: 0.6,
      viral: 0.8,
    },
  },
  {
    id: 'ficus_lyrata',
    scientificName: 'Ficus lyrata',
    commonName: 'Fiddle Leaf Fig',
    optimalTemperature: { min: 16, max: 24 },
    optimalHumidity: { min: 30, max: 65 },
    waterStressThreshold: 0.35,
    diseaseResistance: {
      bacterial: 0.6,
      fungal: 0.5,
      viral: 0.7,
    },
  },
  {
    id: 'epipremnum_aureum',
    scientificName: 'Epipremnum aureum',
    commonName: 'Pothos',
    optimalTemperature: { min: 15, max: 30 },
    optimalHumidity: { min: 40, max: 70 },
    waterStressThreshold: 0.45,
    diseaseResistance: {
      bacterial: 0.8,
      fungal: 0.7,
      viral: 0.8,
    },
  },
  {
    id: 'sansevieria_trifasciata',
    scientificName: 'Sansevieria trifasciata',
    commonName: 'Snake Plant',
    optimalTemperature: { min: 16, max: 27 },
    optimalHumidity: { min: 30, max: 50 },
    waterStressThreshold: 0.5,
    diseaseResistance: {
      bacterial: 0.9,
      fungal: 0.8,
      viral: 0.9,
    },
  },
  {
    id: 'spathiphyllum',
    scientificName: 'Spathiphyllum wallisii',
    commonName: 'Peace Lily',
    optimalTemperature: { min: 18, max: 26 },
    optimalHumidity: { min: 50, max: 80 },
    waterStressThreshold: 0.32,
    diseaseResistance: {
      bacterial: 0.6,
      fungal: 0.5,
      viral: 0.7,
    },
  },
  {
    id: 'petunia',
    scientificName: 'Petunia × atkinsiana',
    commonName: 'Petunia',
    optimalTemperature: { min: 13, max: 24 },
    optimalHumidity: { min: 40, max: 60 },
    waterStressThreshold: 0.38,
    diseaseResistance: {
      bacterial: 0.5,
      fungal: 0.4,
      viral: 0.6,
    },
  },
  {
    id: 'viola_wittrockiana',
    scientificName: 'Viola × wittrockiana',
    commonName: 'Pansy',
    optimalTemperature: { min: 7, max: 18 },
    optimalHumidity: { min: 45, max: 65 },
    waterStressThreshold: 0.36,
    diseaseResistance: {
      bacterial: 0.6,
      fungal: 0.5,
      viral: 0.7,
    },
  },
  {
    id: 'calendula_officinalis',
    scientificName: 'Calendula officinalis',
    commonName: 'Calendula',
    optimalTemperature: { min: 15, max: 25 },
    optimalHumidity: { min: 40, max: 60 },
    waterStressThreshold: 0.37,
    diseaseResistance: {
      bacterial: 0.7,
      fungal: 0.6,
      viral: 0.7,
    },
  },
];

export const STRESS_LEVEL_THRESHOLDS = {
  none: { min: 0, max: 0.2 },
  mild: { min: 0.2, max: 0.36 },
  moderate: { min: 0.36, max: 0.6 },
  severe: { min: 0.6, max: 1.0 },
};

export const RECOMMENDATION_TEMPLATES = {
  water_stress: {
    mild: 'Consider watering your plant soon. The leaves are showing early signs of water stress.',
    moderate: 'Your plant needs water. Water thoroughly until excess drains from the bottom.',
    severe: 'Critical: Water immediately! Your plant is experiencing severe dehydration.',
  },
  disease: {
    bacterial: 'Bacterial infection detected. Isolate the plant, remove affected leaves, and improve air circulation.',
    fungal: 'Fungal infection detected. Reduce humidity, increase air flow, and consider fungicide treatment.',
    viral: 'Possible viral infection. Isolate immediately and monitor closely. Remove affected parts with sterilized tools.',
  },
  nutrient: {
    nitrogen_deficient: 'Nitrogen deficiency detected. Use a balanced fertilizer with higher nitrogen content.',
    phosphorus_deficient: 'Phosphorus deficiency detected. Apply phosphorus-rich fertilizer for better flowering.',
    potassium_deficient: 'Potassium deficiency detected. Use potassium-rich fertilizer to improve overall plant health.',
  },
};