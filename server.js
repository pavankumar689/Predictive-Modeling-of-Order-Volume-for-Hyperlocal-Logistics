const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

const app = express();
const PORT = process.env.PORT || 8501;

// Security and performance middleware
app.use(helmet({
  contentSecurityPolicy: false, // Streamlit needs this disabled
  crossOriginEmbedderPolicy: false
}));
app.use(compression());
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'Food Delivery Demand Predictor',
    timestamp: new Date().toISOString()
  });
});

// Root endpoint with information
app.get('/', (req, res) => {
  res.json({
    message: 'Food Delivery Demand Predictor API',
    description: 'Machine learning-powered demand forecasting for food delivery',
    endpoints: {
      health: '/health',
      streamlit: 'Streamlit app running on this port'
    },
    version: '1.0.0'
  });
});

// Start Streamlit app
function startStreamlit() {
  console.log('Starting Streamlit application...');
  
  const streamlitProcess = spawn('python3', ['-m', 'streamlit', 'run', 'streamlit_app.py', '--server.port', PORT, '--server.address', '0.0.0.0', '--server.headless', 'true'], {
    stdio: 'inherit',
    cwd: __dirname
  });

  streamlitProcess.on('error', (error) => {
    console.error('Failed to start Streamlit:', error);
    process.exit(1);
  });

  streamlitProcess.on('close', (code) => {
    console.log(`Streamlit process exited with code ${code}`);
    if (code !== 0) {
      process.exit(1);
    }
  });

  return streamlitProcess;
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start the application
console.log(`Starting Food Delivery Demand Predictor on port ${PORT}`);
startStreamlit();