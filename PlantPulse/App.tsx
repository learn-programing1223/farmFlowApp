import React from 'react';
import { StatusBar, useColorScheme } from 'react-native';
import { Provider as PaperProvider } from 'react-native-paper';
import RootNavigator from './src/navigation/RootNavigator';

function App(): React.JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <PaperProvider>
      <StatusBar 
        barStyle="light-content" 
        backgroundColor="#4CAF50"
      />
      <RootNavigator />
    </PaperProvider>
  );
}

export default App;
