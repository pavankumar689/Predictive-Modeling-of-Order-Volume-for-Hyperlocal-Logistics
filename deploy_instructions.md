# 🚀 Deployment Instructions

## Platform-Specific Deployment

### 1. Heroku Deployment

```bash
# Install Heroku CLI
# Create new Heroku app
heroku create your-food-delivery-predictor

# Set stack to container (for Docker)
heroku stack:set container -a your-food-delivery-predictor

# Deploy
git add .
git commit -m "Deploy food delivery predictor"
git push heroku main
```

### 2. Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 3. Render Deployment

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

### 4. Streamlit Cloud

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository and deploy

### 5. Docker Deployment

```bash
# Build image
docker build -t food-delivery-predictor .

# Run container
docker run -p 8501:8501 food-delivery-predictor

# Or use docker-compose
docker-compose up
```

### 6. Google Cloud Run

```bash
# Build and deploy
gcloud run deploy food-delivery-predictor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Environment Variables

Set these if needed:
- `PORT`: Application port (default: 8501)
- `STREAMLIT_SERVER_HEADLESS`: Set to `true` for production

## Post-Deployment Checklist

- ✅ Model file (`hyperlocal_demand_predictor.pkl`) is included
- ✅ Dataset file (`multi_city_food_delivery_demand.csv`) is included  
- ✅ All dependencies are in `requirements.txt`
- ✅ Health check endpoint works
- ✅ Application loads without errors
- ✅ Predictions work correctly

## Monitoring

- Check application logs for errors
- Monitor memory usage (ML models can be memory-intensive)
- Set up alerts for downtime
- Monitor prediction accuracy over time