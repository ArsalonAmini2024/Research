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
        # Process all images in the input directory
        print("Starting preprocessing pipeline...")
        features = preprocessor.process_dataset()
        
        # Print some basic information about the processed data
        print("\nProcessing complete!")
        print(f"Number of processed images: {len(features)}")
        
        # Print example features for the first image
        if features:
            first_image = list(features.keys())[0]
            print(f"\nExample features for {first_image}:")
            for roi, metrics in list(features[first_image].items())[:3]:  # Show first 3 ROIs
                print(f"\n{roi}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")
    
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
