import React from 'react'
import { Container, AppBar, Toolbar, Typography, Box, Paper } from '@mui/material'
import AudioUploader from './components/AudioUploader'

export default function App(){
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Typography variant="h6">Purple Audio Explain</Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>Upload audio and get predictions + SHAP</Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Upload a WAV/MP3 audio file. The backend will return a prediction and a SHAP explainability plot.
          </Typography>
          <AudioUploader />
        </Paper>
      </Container>
    </Box>
  )
}
