"""
Questionnaire Data Processor for PADS Parkinson's Disease Dataset

This module processes questionnaire JSON files from the PADS dataset and converts
them into wide-format CSV files suitable for machine learning analysis.

Author: Shiva
Date: 2026-03-17
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from tqdm import tqdm


class QuestionnaireProcessor:
    """Process and transform questionnaire data from JSON to wide-format CSV."""
    
    def __init__(self, questionnaire_dir: str):
        """
        Initialize the processor with a questionnaire directory path.
        
        Parameters
        ----------
        questionnaire_dir : str
            Path to directory containing questionnaire JSON files
        """
        self.questionnaire_dir = Path(questionnaire_dir)
        self.questionnaire_files = list(self.questionnaire_dir.glob("*.json"))
        self.data_long = None
        self.data_wide = None
        
        if not self.questionnaire_files:
            raise FileNotFoundError(f"No JSON files found in {questionnaire_dir}")
        
        print(f"Found {len(self.questionnaire_files)} questionnaire files")
    
    @staticmethod
    def load_questionnaire_file(json_path: str) -> Dict[str, Any]:
        """
        Load a single questionnaire JSON file.
        
        Parameters
        ----------
        json_path : str
            Path to the JSON file
            
        Returns
        -------
        dict
            Parsed JSON data
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def process_all_files(self) -> pd.DataFrame:
        """
        Process all questionnaire files and return long-format DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: subject_id, link_id, answer
        """
        rows = []
        
        for file in tqdm(self.questionnaire_files, desc="Processing files"):
            try:
                data = self.load_questionnaire_file(file)
                subject_id = data.get("subject_id")
                
                if subject_id is None:
                    print(f"Warning: No subject_id in {file.name}")
                    continue
                
                items = data.get("item", [])
                for item in items:
                    rows.append({
                        "subject_id": subject_id,
                        "link_id": item.get("link_id"),
                        "answer": int(bool(item.get("answer", False)))  # False->0, True->1
                    })
            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        if not rows:
            raise ValueError("No data extracted from any files")
        
        self.data_long = pd.DataFrame(rows).sort_values("link_id").reset_index(drop=True)
        print(f"Extracted {len(self.data_long)} responses from {len(self.questionnaire_files)} subjects")
        
        return self.data_long
    
    def to_wide_format(self) -> pd.DataFrame:
        """
        Convert long-format data to wide-format (one row per subject).
        
        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame with subject_id and question columns
        """
        if self.data_long is None:
            raise ValueError("No data loaded. Call process_all_files() first.")
        
        # Format IDs with leading zeros
        df = self.data_long.copy()
        df["subject_id"] = df["subject_id"].astype(str).str.zfill(3)
        df["link_id"] = df["link_id"].astype(str).str.zfill(2)
        
        # Pivot to wide format
        wide = (
            df.pivot_table(
                index="subject_id",
                columns="link_id",
                values="answer",
                aggfunc="first"  # one value per subject_id+link_id
            )
            .reset_index()
        )
        
        # Rename question columns
        wide.columns = ["subject_id"] + [f"q_{c}" for c in wide.columns[1:]]
        
        self.data_wide = wide
        print(f"Wide format shape: {wide.shape}")
        
        return wide
    
    def save_to_csv(self, output_path: str = "questionnaire_data_wide.csv") -> None:
        """
        Save wide-format data to CSV file.
        
        Parameters
        ----------
        output_path : str
            Output CSV file path (default: questionnaire_data_wide.csv)
        """
        if self.data_wide is None:
            raise ValueError("No wide-format data. Call to_wide_format() first.")
        
        self.data_wide.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the questionnaire data.
        
        Returns
        -------
        dict
            Summary statistics
        """
        if self.data_wide is None:
            raise ValueError("No wide-format data. Call to_wide_format() first.")
        
        question_cols = [c for c in self.data_wide.columns if c.startswith("q_")]
        
        stats = {
            "total_subjects": len(self.data_wide),
            "total_questions": len(question_cols),
            "response_rate": self.data_wide[question_cols].notna().mean().mean(),
            "question_means": self.data_wide[question_cols].mean().to_dict(),
            "question_stds": self.data_wide[question_cols].std().to_dict(),
        }
        
        return stats
    
    def print_summary(self) -> None:
        """Print a summary of the processed data."""
        if self.data_wide is None:
            print("No data processed yet.")
            return
        
        stats = self.get_summary_stats()
        
        print("\n" + "="*60)
        print("QUESTIONNAIRE DATA SUMMARY")
        print("="*60)
        print(f"Total Subjects:    {stats['total_subjects']}")
        print(f"Total Questions:   {stats['total_questions']}")
        print(f"Response Rate:     {stats['response_rate']:.2%}")
        print("="*60)


def main():
    """Main execution function."""
    # Configuration
    QUESTIONNAIRE_DIR = "pads-parkinsons-disease-smartwatch-dataset-1.0.0/questionnaire"
    OUTPUT_CSV = "questionnaire_data_wide.csv"
    
    try:
        # Initialize processor
        processor = QuestionnaireProcessor(QUESTIONNAIRE_DIR)
        
        # Process all files
        processor.process_all_files()
        
        # Convert to wide format
        processor.to_wide_format()
        
        # Print summary
        processor.print_summary()
        
        # Save to CSV
        processor.save_to_csv(OUTPUT_CSV)
        
        print(f"\nSuccessfully processed questionnaire data!")
        print(f"Output saved to: {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()