# 🍕 Food Delivery Demand Predictor

A machine learning-powered Streamlit application that predicts hourly food delivery demand based on location, time, and weather conditions.

## 📁 Project Structure

```
food-delivery-predictor/
│
├── streamlit_app.py              # Main Streamlit application
├── train_model.py               # Model training script
├── requirements.txt             # Python dependencies
├── multi_city_food_delivery_demand.csv  # Your dataset
├── hyperlocal_demand_predictor.pkl      # Trained model (generated)
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Ensure your CSV file named `multi_city_food_delivery_demand.csv` is in the project directory with the following columns:
- `hour`: Hour of the day (0-23)
- `city`: City name
- `zone_id`: Zone identifier
- `zone_type`: Type of zone (Urban, Suburban, Metropolitan)
- `weather_condition`: Weather condition
- `num_orders`: Number of orders (target variable)

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess your data
- Train multiple models (Linear Regression, Random Forest, LightGBM)
- Perform hyperparameter tuning
- Save the best model as `hyperlocal_demand_predictor.pkl`

### 4. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Features

### User Interface
- **Interactive Sidebar**: Easy input selection for all parameters
- **Real-time Predictions**: Instant demand forecasting
- **Visual Insights**: Color-coded demand levels and business metrics
- **Business Intelligence**: Actionable insights for operations

### Input Parameters
- **Hour of Day**: 24-hour format slider (0-23)
- **City**: Dropdown selection from major Indian cities
- **Zone ID**: Zone identifier for delivery areas
- **Zone Type**: Urban, Suburban, or Metropolitan
- **Weather Condition**: Clear, Rain, Fog, Cloudy, or Storm

### Output Insights
- **Predicted Orders**: Forecasted demand for the next hour
- **Demand Level**: High, Medium, or Low classification
- **Drivers Needed**: Recommended driver allocation
- **Preparation Time**: Suggested kitchen prep time

## 📊 Model Performance

- **Algorithm**: LightGBM (Gradient Boosting)
- **Accuracy**: ~85% R² Score
- **Features**: One-hot encoded categorical variables
- **Training**: Hyperparameter tuned via RandomizedSearchCV

## 🏢 Business Applications

### Operations Planning
- **Staff Scheduling**: Optimize driver and kitchen staff allocation
- **Inventory Management**: Predict ingredient requirements
- **Marketing**: Target promotions during low-demand periods

### Strategic Decisions
- **Expansion**: Identify high-demand zones for new restaurants
- **Pricing**: Dynamic pricing based on demand forecasts
- **Partnerships**: Weather-based promotional campaigns

## 🛠️ Customization

### Adding New Cities
Update the `cities` list in `streamlit_app.py`:
```python
cities = ['Mumbai', 'Delhi', 'Bangalore', 'YourCity']
```

### Modifying Zone Types
Update the `zone_types` list:
```python
zone_types = ['Urban', 'Suburban', 'Metropolitan', 'Rural']
```

### Changing Weather Conditions
Update the `weather_conditions` list:
```python
weather_conditions = ['Clear', 'Rain', 'Fog', 'Cloudy', 'Storm', 'Snow']
```

## 📈 Model Retraining

To retrain with new data:

1. Replace `multi_city_food_delivery_demand.csv` with your updated dataset
2. Run `python train_model.py`
3. The new model will automatically replace the existing one

## 🔧 Troubleshooting

### Common Issues

**Model file not found error:**
```
❌ Model file not found. Please ensure 'hyperlocal_demand_predictor.pkl' is in the same directory.
```
**Solution**: Run `python train_model.py` first to generate the model file.

**CSV file not found:**
```
Error: Could not find 'multi_city_food_delivery_demand.csv'
```
**Solution**: Ensure your dataset file is in the project directory with the exact filename.

**Import errors:**
**Solution**: Install all dependencies with `pip install -r requirements.txt`

### Performance Optimization

For large datasets:
- Increase `n_iter` in RandomizedSearchCV for better tuning
- Use `n_jobs=-1` to utilize all CPU cores
- Consider feature selection for high-dimensional data

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the console output for error messages
- Ensure all dependencies are installed correctly

## 🎯 Future Enhancements

- **Real-time Data Integration**: Connect to live weather APIs
- **Advanced Visualizations**: Interactive charts and dashboards  
- **Multi-model Ensemble**: Combine multiple algorithms
- **A/B Testing**: Compare different prediction strategies
- **Mobile Optimization**: Responsive design for mobile devices

## 📝 License

This project is for educational and commercial use. Modify as needed for your business requirements.

---

**Happy Predicting! 🚀**