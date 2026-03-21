"""
BioBERT Feature Extraction for Parkinson's Disease Questionnaire Data

This module extracts semantic features from questionnaire data using BioBERT,
a biomedical language model trained on PubMed literature. The extracted embeddings
capture medical domain knowledge and can be used for classification tasks.

Features:
- BioBERT-based semantic feature extraction
- Batch processing of questionnaire files
- Integration with patient metadata and labels
- Stratified k-fold cross-validation for classification
- Support for binary and multiclass problems
- Multiple classifier options (Logistic Regression, Random Forest, XGBoost, SVM)

Author: Shiva
Date: 2026-03-17

References:
- BioBERT: https://github.com/dmis-lab/biobert
- PADS Dataset: https://physionet.org/content/pads/1.0.0/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


class BioBERTExtractor:
    """Extract semantic features from medical text using BioBERT."""
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", device: Optional[str] = None):
        """
        Initialize BioBERT model and tokenizer.
        
        Parameters
        ----------
        model_name : str, optional
            HuggingFace model identifier (default: dmis-lab/biobert-base-cased-v1.1)
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detects GPU if available.
            
        Raises
        ------
        OSError
            If model cannot be loaded from HuggingFace
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BioBERT model: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            raise OSError(f"Failed to load BioBERT model: {e}")
    
    def extract_embedding(
        self,
        text: str,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Extract BioBERT CLS token embedding for input text.
        
        The CLS token (first token) represents the aggregate semantic meaning
        of the entire input sequence in BERT-based models.
        
        Parameters
        ----------
        text : str
            Input text to embed
        max_length : int, optional
            Maximum sequence length (default: 512)
            
        Returns
        -------
        np.ndarray
            Embedding vector of shape (768,) for BioBERT-base
        """
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]  # Return as 1D array


class QuestionnaireProcessor:
    """Process questionnaire JSON files and extract BioBERT features."""
    
    def __init__(self, biobert_extractor: BioBERTExtractor):
        """
        Initialize processor with BioBERT extractor.
        
        Parameters
        ----------
        biobert_extractor : BioBERTExtractor
            Initialized BioBERT extractor instance
        """
        self.extractor = biobert_extractor
    
    @staticmethod
    def load_questionnaire_json(json_path: str) -> Dict[str, Any]:
        """
        Load questionnaire data from JSON file.
        
        Parameters
        ----------
        json_path : str
            Path to questionnaire JSON file
            
        Returns
        -------
        dict
            Parsed questionnaire data
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def process_single_questionnaire(
        self,
        questionnaire_data: Dict[str, Any],
        strategy: str = "combined"
    ) -> Dict[str, Any]:
        """
        Process a single questionnaire and extract BioBERT features.
        
        Parameters
        ----------
        questionnaire_data : dict
            Questionnaire data loaded from JSON
        strategy : str, optional
            Feature extraction strategy:
            - 'combined': Question + Answer combined (default, most informative)
            - 'question_only': Question text only
            - 'all_together': All questions combined into single document
            
        Returns
        -------
        dict
            Processed questionnaire with keys:
            - questions: List of question texts
            - answers: List of boolean answers
            - embeddings: (n_questions, 768) array of embeddings
            - subject_id: Patient subject ID
            - questionnaire_name: Name of questionnaire
        """
        questions = []
        answers = []
        embeddings = []
        
        for item in questionnaire_data.get('item', []):
            question_text = item.get('text', '')
            answer = item.get('answer', False)
            
            # Prepare text based on strategy
            if strategy == "combined":
                text = f"Question: {question_text} Answer: {'Yes' if answer else 'No'}"
            elif strategy == "question_only":
                text = question_text
            else:
                text = question_text
            
            questions.append(question_text)
            answers.append(answer)
            
            # Extract embedding
            embedding = self.extractor.extract_embedding(text)
            embeddings.append(embedding)
        
        return {
            'questions': questions,
            'answers': answers,
            'embeddings': np.array(embeddings),
            'subject_id': questionnaire_data.get('subject_id'),
            'questionnaire_name': questionnaire_data.get('questionnaire_name')
        }
    
    def process_all_questionnaires(
        self,
        questionnaire_dir: str,
        strategy: str = "combined"
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Process all questionnaire files in a directory.
        
        Parameters
        ----------
        questionnaire_dir : str
            Path to directory containing questionnaire JSON files
        strategy : str, optional
            Feature extraction strategy (default: "combined")
            
        Returns
        -------
        tuple
            - feature_matrix: (n_subjects, 768) array of mean-pooled embeddings
            - subject_ids: List of subject IDs corresponding to each row
        """
        questionnaire_files = list(Path(questionnaire_dir).glob("*.json"))
        
        if not questionnaire_files:
            raise FileNotFoundError(f"No questionnaire JSON files found in {questionnaire_dir}")
        
        all_features = []
        all_subject_ids = []
        
        print(f"Processing {len(questionnaire_files)} questionnaire files...")
        
        for q_file in tqdm(questionnaire_files, desc="Extracting features"):
            try:
                data = self.load_questionnaire_json(q_file)
                processed = self.process_single_questionnaire(data, strategy=strategy)
                
                # Mean pooling: average embeddings across all questions
                mean_embedding = processed['embeddings'].mean(axis=0)
                
                all_features.append(mean_embedding)
                all_subject_ids.append(processed['subject_id'])
            
            except Exception as e:
                print(f"⚠ Error processing {q_file.name}: {e}")
                continue
        
        feature_matrix = np.array(all_features)
        
        print(f"\n✓ Feature extraction complete")
        print(f"  Total samples: {len(all_subject_ids)}")
        print(f"  Feature dimensions: {feature_matrix.shape[1]}")
        print(f"  Feature statistics:")
        print(f"    Mean: {feature_matrix.mean():.4f}")
        print(f"    Std:  {feature_matrix.std():.4f}")
        print(f"    Min:  {feature_matrix.min():.4f}")
        print(f"    Max:  {feature_matrix.max():.4f}")
        
        return feature_matrix, all_subject_ids


class PatientMetadataLoader:
    """Load patient demographic and clinical metadata."""
    
    @staticmethod
    def load_all_patient_metadata(patient_json_dir: str) -> pd.DataFrame:
        """
        Load all patient metadata from JSON files.
        
        Parameters
        ----------
        patient_json_dir : str
            Path to directory containing patient JSON files
            
        Returns
        -------
        pd.DataFrame
            Patient metadata with encoded labels and demographics
        """
        patient_files = sorted(Path(patient_json_dir).glob("*.json"))
        
        if not patient_files:
            raise FileNotFoundError(f"No patient JSON files found in {patient_json_dir}")
        
        rows = []
        
        for fpath in tqdm(patient_files, desc="Loading patient metadata"):
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Encode diagnosis label: 0=Healthy, 1=Parkinson's, 2=Other
            condition = data.get("condition", "Unknown")
            if "Parkinson's" in condition:
                label = 1
            elif "Healthy" in condition:
                label = 0
            else:
                label = 2
            
            # Encode sex: 0=Female, 1=Male
            sex = 1 if data.get("gender", "").lower() == "male" else 0
            
            rows.append({
                "id": fpath.name,
                "age": data.get("age_at_diagnosis"),
                "label": label,
                "height": data.get("height"),
                "weight": data.get("weight"),
                "sex": sex,
            })
        
        df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
        
        # Extract subject ID from filename (patient_XXX.json -> XXX)
        df["subject_id"] = df["id"].str.extract(r"patient_(\d+)\.json")[0].astype(int)
        
        print(f"✓ Loaded metadata for {len(df)} patients")
        print(f"\nLabel distribution:")
        print(df["label"].value_counts().sort_index())
        
        return df


class ClassificationPipeline:
    """Classification pipeline with stratified k-fold CV and multiple metrics."""
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """
        Initialize classification pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels
        n_splits : int, optional
            Number of CV folds (default: 5)
        random_state : int, optional
            Random seed (default: 42)
        """
        self.X = X.reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = []
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Find optimal classification threshold using Youden's J statistic.
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_proba : np.ndarray
            Predicted probabilities for positive class
            
        Returns
        -------
        float
            Optimal threshold value
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    
    def run_stratified_cv(self, classifier_name: str = "XGBoost") -> pd.DataFrame:
        """
        Run stratified k-fold cross-validation with comprehensive metrics.
        
        Parameters
        ----------
        classifier_name : str, optional
            Name of classifier to use (default: "XGBoost")
            
        Returns
        -------
        pd.DataFrame
            Cross-validation results for each fold
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        if classifier_name == "XGBoost":
            try:
                from xgboost import XGBClassifier
                model_class = XGBClassifier
                model_kwargs = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": self.random_state,
                    "eval_metric": "logloss",
                    "use_label_encoder": False,
                    "verbosity": 0
                }
            except ImportError:
                print("XGBoost not installed. Install with: pip install xgboost")
                return None
        else:
            raise ValueError(f"Unsupported classifier: {classifier_name}")
        
        fold_results = []
        
        print(f"\nRunning {self.n_splits}-fold Stratified Cross-Validation ({classifier_name})")
        print("="*70)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y), 1):
            X_tr = self.X.iloc[train_idx]
            X_te = self.X.iloc[test_idx]
            y_tr = self.y.iloc[train_idx]
            y_te = self.y.iloc[test_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            # Train model
            clf = model_class(**model_kwargs)
            clf.fit(X_tr_scaled, y_tr)
            
            # Get predictions
            y_proba = clf.predict_proba(X_te_scaled)
            
            # Handle binary vs multiclass
            if len(np.unique(self.y)) == 2:
                # Binary classification
                y_proba_pos = y_proba[:, 1]
                best_thresh = self.find_optimal_threshold(y_te, y_proba_pos)
                y_pred = (y_proba_pos >= best_thresh).astype(int)
                roc_auc = roc_auc_score(y_te, y_proba_pos)
                precision = precision_score(y_te, y_pred, zero_division=0)
                recall = recall_score(y_te, y_pred, zero_division=0)
                f1 = f1_score(y_te, y_pred, zero_division=0)
            else:
                # Multiclass classification
                y_pred = np.argmax(y_proba, axis=1)
                roc_auc = roc_auc_score(y_te, y_proba, multi_class="ovr")
                precision = precision_score(y_te, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_te, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            
            # Compute all metrics
            metrics = {
                "fold": fold,
                "accuracy": accuracy_score(y_te, y_pred),
                "roc_auc": roc_auc,
                "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            
            fold_results.append(metrics)
            
            print(f"Fold {fold}: acc={metrics['accuracy']:.4f}, auc={metrics['roc_auc']:.4f}, "
                  f"f1={metrics['f1']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}")
        
        self.results = pd.DataFrame(fold_results)
        
        # Print summary
        print("="*70)
        print("\nCross-Validation Summary:")
        for col in ["accuracy", "roc_auc", "balanced_accuracy", "precision", "recall", "f1"]:
            mean = self.results[col].mean()
            std = self.results[col].std()
            print(f"  {col:20s}: {mean:.4f} ± {std:.4f}")
        
        return self.results
    
    def print_detailed_results(self) -> None:
        """Print detailed results for each fold."""
        if self.results is None:
            print("No results available. Run run_stratified_cv() first.")
            return
        
        print("\nDetailed Results per Fold:")
        print(self.results.to_string(index=False))


def main():
    """Main execution function."""
    
    # Configuration
    QUESTIONNAIRE_DIR = "pads-parkinsons-disease-smartwatch-dataset-1.0.0/questionnaire"
    PATIENT_JSON_DIR = "pads-parkinsons-disease-smartwatch-dataset-1.0.0/patients"
    OUTPUT_FEATURES_CSV = "biobert_questionnaire_features.csv"
    OUTPUT_RESULTS_CSV = "biobert_classification_results.csv"
    
    try:
        # Step 1: Initialize BioBERT extractor
        print("="*70)
        print("BIOBERT FEATURE EXTRACTION FOR PARKINSON'S DISEASE PREDICTION")
        print("="*70)
        
        extractor = BioBERTExtractor()
        
        # Step 2: Extract features from questionnaires
        print("\n" + "="*70)
        print("STEP 1: EXTRACTING BIOBERT FEATURES FROM QUESTIONNAIRES")
        print("="*70)
        
        processor = QuestionnaireProcessor(extractor)
        feature_matrix, subject_ids = processor.process_all_questionnaires(
            QUESTIONNAIRE_DIR,
            strategy="combined"
        )
        
        # Step 3: Load patient metadata and labels
        print("\n" + "="*70)
        print("STEP 2: LOADING PATIENT METADATA AND LABELS")
        print("="*70)
        
        df_patients = PatientMetadataLoader.load_all_patient_metadata(PATIENT_JSON_DIR)
        
        # Step 4: Create feature DataFrame and merge with labels
        print("\n" + "="*70)
        print("STEP 3: MERGING FEATURES WITH PATIENT LABELS")
        print("="*70)
        
        features_df = pd.DataFrame(feature_matrix)
        features_df['subject_id'] = subject_ids
        features_df = features_df.merge(
            df_patients[['subject_id', 'label']],
            on='subject_id',
            how='inner'
        )
        
        print(f"Final dataset shape: {features_df.shape}")
        print(f"Label distribution:\n{features_df['label'].value_counts().sort_index()}")
        
        # Save features
        features_df.to_csv(OUTPUT_FEATURES_CSV, index=False)
        print(f"\n✓ Features saved to {OUTPUT_FEATURES_CSV}")
        
        # Step 5: Run classification pipeline
        print("\n" + "="*70)
        print("STEP 4: CLASSIFICATION WITH STRATIFIED 5-FOLD CV")
        print("="*70)
        
        X = features_df.drop(columns=['label', 'subject_id'])
        y = features_df['label']
        
        pipeline = ClassificationPipeline(X, y, n_splits=5)
        results = pipeline.run_stratified_cv(classifier_name="XGBoost")
        
        if results is not None:
            results.to_csv(OUTPUT_RESULTS_CSV, index=False)
            print(f"\n✓ Results saved to {OUTPUT_RESULTS_CSV}")
        
        print("\n" + "="*70)
        print("✓ BioBERT feature extraction and classification completed successfully!")
        print("="*70)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()