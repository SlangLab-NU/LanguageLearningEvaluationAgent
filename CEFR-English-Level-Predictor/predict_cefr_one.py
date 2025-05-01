from cefr_predictor.inference import Model

def predict_cefr_level(text, use_plus_levels=False):
    # Initialize the model
    model = Model("cefr_predictor/models/xgboost.joblib", use_plus_levels=use_plus_levels)
    
    # Make prediction
    levels, scores = model.predict_decode([text])
    
    # Return the first prediction since we only input one text
    return levels[0], scores[0]

if __name__ == "__main__":
    # Example usage
    text = "This is a sample text that I want to analyze."
    
    # Standard 6-level CEFR prediction
    level, scores = predict_cefr_level(text, use_plus_levels=False)
    print(f"Standard CEFR Level: {level}")
    
    # Original plus-level prediction (commented out by default)
    # level_plus, scores_plus = predict_cefr_level(text, use_plus_levels=True)
    # print(f"CEFR Level with plus: {level_plus}")