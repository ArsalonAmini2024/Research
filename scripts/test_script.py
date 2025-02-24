import os
from src.preprocessing import MRIPreprocessor

def main():
    # Define input and output directories
    input_dir = "data/raw"
    output_dir = "data/processed"
    
    # Create test directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the preprocessor
    preprocessor = MRIPreprocessor(input_dir=input_dir, output_dir=output_dir)
    
    try:
        # Process all images and get features DataFrame
        print("Starting preprocessing pipeline...")
        features_df = preprocessor.process_dataset()
        
        print("\nFeatures have been saved to:", os.path.join(output_dir, 'radiomics_features.csv'))
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
