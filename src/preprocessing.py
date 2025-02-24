import os
import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.datasets import fetch_atlas_harvard_oxford
import ants
from tqdm import tqdm
import gc  # For garbage collection
import logging
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRIPreprocessor:
    def __init__(self, input_dir, output_dir, config_path='config.yaml'):
        """
        Initialize the preprocessor with input and output directories
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Initializing MRIPreprocessor...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load Brainnetome atlas for registration
        logger.info("Loading Brainnetome atlas for registration...")
        atlas_path = self.config['data']['atlas_path']
        if not os.path.exists(atlas_path):
            raise FileNotFoundError(f"Atlas file not found at: {atlas_path}")
        
        # Load atlas for registration template
        logger.info("Loading template...")
        try:
            self.mni_template = ants.image_read(atlas_path)
            logger.info("Template loaded successfully")
            
            # Load atlas for ROI labels
            self.atlas_path = self.config['data']['atlas_labels']
            if not os.path.exists(self.atlas_path):
                raise FileNotFoundError(f"Atlas labels not found at: {self.atlas_path}")
            logger.info("Atlas labels loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading atlas/template: {str(e)}")
            raise
        
    def load_nifti(self, filepath):
        """Load NIfTI file and return image data"""
        try:
            logger.info(f"Loading NIfTI file: {filepath}")
            return nib.load(filepath)
        except Exception as e:
            logger.error(f"Error loading NIfTI file: {str(e)}")
            raise
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single MRI scan
        1. Convert to NIfTI if needed
        2. Perform bias field correction
        3. Register to MNI space
        4. Segment brain regions
        """
        logger.info(f"Starting preprocessing for: {image_path}")
        
        try:
            # Load image
            img = self.load_nifti(image_path)
            
            # Get image data and convert to float32
            logger.info("Converting image data...")
            img_data = np.array(img.get_fdata(), dtype='float32')
            
            # Get spacing from header
            spacing = img.header.get_zooms()[:3]
            
            # Calculate direction matrix
            direction = img.affine[:3, :3] / np.array(spacing)[:, None]
            
            # Calculate origin
            origin = img.affine[:3, 3]
            
            # Convert to ANTs image
            logger.info("Converting to ANTs image format...")
            ants_img = ants.from_numpy(
                data=img_data,
                origin=tuple(origin),  # Convert to tuple
                spacing=tuple(spacing),  # Convert to tuple
                direction=direction
            )
            
            # Clear memory
            del img_data
            gc.collect()
            
            # N4 bias field correction
            logger.info("Performing N4 bias field correction...")
            ants_img_n4 = ants.n4_bias_field_correction(ants_img)
            
            # Clear memory
            del ants_img
            gc.collect()
            
            # Register to MNI template
            logger.info("Performing registration to MNI space...")
            registration = ants.registration(
                fixed=self.mni_template,
                moving=ants_img_n4,
                type_of_transform='SyN',
                grad_step=0.2,  # Reduce step size for better stability
                flow_sigma=3,
                total_sigma=0,
                reg_iterations=(40,20,0)  # Reduce iterations for testing
            )
            
            registered_img = registration['warpedmovout']
            
            # Clear memory
            del ants_img_n4
            gc.collect()
            
            return registered_img
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def extract_roi_features(self, registered_img):
        """
        Extract features from ROIs using Harvard-Oxford atlas
        """
        logger.info("Extracting ROI features...")
        try:
            # Convert ANTs image to nibabel
            registered_nib = nib.Nifti1Image(
                registered_img.numpy(),
                affine=np.eye(4)  # Using identity matrix as we're already in MNI space
            )
            
            # Load atlas
            atlas = nib.load(self.atlas_path)
            
            # Resample atlas to match registered image
            atlas_resampled = image.resample_to_img(atlas, registered_nib)
            
            # Get ROI masks
            atlas_data = atlas_resampled.get_fdata()
            
            # Extract features for each ROI
            roi_features = {}
            for roi_id in np.unique(atlas_data)[1:]:  # Skip background (0)
                roi_mask = atlas_data == roi_id
                roi_values = registered_img.numpy()[roi_mask]
                
                if len(roi_values) > 0:  # Only calculate if ROI exists
                    roi_features[f'ROI_{int(roi_id)}'] = {
                        'mean': float(np.mean(roi_values)),  # Convert to native Python float
                        'std': float(np.std(roi_values)),
                        'volume': int(np.sum(roi_mask))  # Convert to native Python int
                    }
            
            return roi_features
            
        except Exception as e:
            logger.error(f"Error in ROI feature extraction: {str(e)}")
            raise
    
    def process_dataset(self):
        """
        Process all images in the input directory
        """
        all_features = {}
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.nii', '.nii.gz'))]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for img_file in tqdm(image_files, desc="Processing MRI scans"):
            img_path = os.path.join(self.input_dir, img_file)
            
            try:
                logger.info(f"\nProcessing: {img_file}")
                
                # Preprocess image
                processed_img = self.preprocess_single_image(img_path)
                
                # Extract ROI features
                features = self.extract_roi_features(processed_img)
                
                # Save features
                all_features[img_file] = features
                
                # Save processed image
                output_path = os.path.join(self.output_dir, f'processed_{img_file}')
                nib.save(
                    nib.Nifti1Image(processed_img.numpy(), np.eye(4)),
                    output_path
                )
                
                logger.info(f"Successfully processed: {img_file}")
                
                # Clear memory
                del processed_img
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
                logger.error(f"Full error: {type(e).__name__}")
                continue
            
        return all_features 