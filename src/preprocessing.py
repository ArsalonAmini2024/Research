import os
import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.datasets import fetch_atlas_harvard_oxford
import ants
from tqdm import tqdm

class MRIPreprocessor:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the preprocessor with input and output directories
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Brainnetome atlas
        # Note: You'll need to download this separately as it's not included in nilearn
        self.atlas_path = "path/to/BNA_atlas.nii.gz"
        
    def load_nifti(self, filepath):
        """Load NIfTI file and return image data"""
        return nib.load(filepath)
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single MRI scan
        1. Convert to NIfTI if needed
        2. Perform bias field correction
        3. Register to MNI space
        4. Segment brain regions
        """
        # Load image
        img = self.load_nifti(image_path)
        
        # Convert to ANTs image
        ants_img = ants.from_nibabel(img)
        
        # N4 bias field correction
        ants_img_n4 = ants.n4_bias_field_correction(ants_img)
        
        # Register to MNI template
        mni_template = ants.get_template('MNI152Lin')
        registration = ants.registration(
            fixed=mni_template,
            moving=ants_img_n4,
            type_of_transform='SyN'
        )
        
        registered_img = registration['warpedmovout']
        
        return registered_img
    
    def extract_roi_features(self, registered_img):
        """
        Extract features from ROIs using Brainnetome atlas
        """
        # Load atlas
        atlas = nib.load(self.atlas_path)
        
        # Resample atlas to match registered image if needed
        atlas_resampled = image.resample_to_img(atlas, registered_img)
        
        # Get ROI masks
        atlas_data = atlas_resampled.get_fdata()
        
        # Extract features for each ROI
        roi_features = {}
        for roi_id in np.unique(atlas_data)[1:]:  # Skip background (0)
            roi_mask = atlas_data == roi_id
            roi_values = registered_img.numpy()[roi_mask]
            
            # Calculate basic statistics for each ROI
            roi_features[f'ROI_{int(roi_id)}'] = {
                'mean': np.mean(roi_values),
                'std': np.std(roi_values),
                'volume': np.sum(roi_mask)
            }
            
        return roi_features
    
    def process_dataset(self):
        """
        Process all images in the input directory
        """
        all_features = {}
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.nii', '.nii.gz'))]
        
        for img_file in tqdm(image_files, desc="Processing MRI scans"):
            img_path = os.path.join(self.input_dir, img_file)
            
            # Preprocess image
            processed_img = self.preprocess_single_image(img_path)
            
            # Extract ROI features
            features = self.extract_roi_features(processed_img)
            
            # Save features
            all_features[img_file] = features
            
            # Save processed image
            output_path = os.path.join(self.output_dir, f'processed_{img_file}')
            processed_img.to_nibabel().to_filename(output_path)
            
        return all_features 