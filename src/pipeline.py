import os
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from radiomics import featureextractor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from lifelines import KaplanMeierFitter, CoxPHFitter

class NeuroImagePipeline:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(**self.config['radiomics'])
        
    def load_processed_data(self):
        """Load processed NIfTI files from CAT12 output"""
        processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_files = list(processed_dir.glob('gmv_*.nii'))
        return self.processed_files
    
    def extract_radiomic_features(self, image_path):
        """Extract radiomic features for each brain region"""
        image = nib.load(image_path)
        atlas = nib.load(self.config['data']['atlas_path'])
        
        features = {}
        for region in range(1, 247):  # 246 Brainnetome regions
            mask = atlas.get_fdata() == region
            region_features = self.radiomics_extractor.execute(image, mask)
            features[f'region_{region}'] = region_features
            
        return features
    
    def construct_r2sn(self, feature_dict):
        """Construct Radiomics Similarity Network"""
        n_regions = 246
        r2sn = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                corr, _ = pearsonr(
                    list(feature_dict[f'region_{i+1}'].values()),
                    list(feature_dict[f'region_{j+1}'].values())
                )
                r2sn[i,j] = r2sn[j,i] = corr
                
        return r2sn
    
    def calculate_rmcs(self, r2sn):
        """Calculate Regional Mean Connectivity Strength"""
        return np.mean(r2sn, axis=1)
    
    def train_svm_models(self, X_dict, y):
        """Train SVM models for each feature type"""
        models = {}
        for feature_type, X in X_dict.items():
            svm = SVC(probability=True)
            grid_search = GridSearchCV(
                svm, 
                self.config['svm']['param_grid'],
                cv=self.config['svm']['cv_folds']
            )
            grid_search.fit(X, y)
            models[feature_type] = grid_search.best_estimator_
        return models
    
    def compute_ibrain_score(self, svm_models, X_dict):
        """Compute IBRAIN score using linear regression ensemble"""
        svm_predictions = np.column_stack([
            model.predict_proba(X_dict[feat_type])[:,1]
            for feat_type, model in svm_models.items()
        ])
        
        ensemble = LinearRegression()
        ensemble.fit(svm_predictions, y)
        return ensemble.predict(svm_predictions)
    
    def run_pipeline(self, clinical_data):
        """Execute full pipeline"""
        # Load processed images
        processed_files = self.load_processed_data()
        
        # Extract features
        all_features = {}
        for file in processed_files:
            subject_features = self.extract_radiomic_features(file)
            r2sn = self.construct_r2sn(subject_features)
            rmcs = self.calculate_rmcs(r2sn)
            
            all_features[file.stem] = {
                'gmv': subject_features,
                'r2sn': r2sn,
                'rmcs': rmcs
            }
        
        # Train models
        X_dict = {
            'gmv': np.array([f['gmv'] for f in all_features.values()]),
            'r2sn': np.array([f['r2sn'].flatten() for f in all_features.values()]),
            'rmcs': np.array([f['rmcs'] for f in all_features.values()])
        }
        
        svm_models = self.train_svm_models(X_dict, clinical_data['diagnosis'])
        ibrain_scores = self.compute_ibrain_score(svm_models, X_dict)
        
        return ibrain_scores, svm_models, all_features

if __name__ == "__main__":
    pipeline = NeuroImagePipeline()
    # Add code to load clinical data and run pipeline 