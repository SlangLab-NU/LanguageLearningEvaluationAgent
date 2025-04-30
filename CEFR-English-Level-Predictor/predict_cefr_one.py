from cefr_predictor.inference import Model

def predict_cefr_level(text):
    # Initialize the model
    model = Model("cefr_predictor/models/xgboost.joblib")
    
    # Make prediction
    levels, scores = model.predict_decode([text])
    
    # Return the first prediction since we only input one text
    return levels[0], scores[0]

if __name__ == "__main__":
    # Example usage
    text = "This is a sample text that I want to analyze."
    level, scores = predict_cefr_level(text)
    
    print(f"CEFR Level: {level}") 