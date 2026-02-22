"""
Complete Training Pipeline for Smart Contract Decompilation

This module orchestrates the entire training process and implements the evaluation
framework as described in the paper, including semantic similarity, edit distance,
and structural fidelity metrics.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import sqlite3

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# NLP and evaluation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Our modules
from .bytecode_analyzer import BytecodeAnalyzer
from .dataset_pipeline import DatasetBuilder
from .model_setup import SmartContractModelTrainer, ModelConfig, SmartContractDecompiler

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics as described in the paper."""
    semantic_similarity: float
    normalized_edit_distance: float
    bleu_score: float
    rouge_l_score: float
    token_accuracy: float
    structural_preservation: float
    function_signature_match: bool
    visibility_match: bool
    metadata: Dict[str, Any] = None

@dataclass
class TrainingConfig:
    """Configuration for the complete training pipeline."""
    # Data collection
    etherscan_api_key: str
    contract_addresses_file: Optional[str] = None
    target_dataset_size: int = 238446  # As mentioned in paper
    
    # Dataset processing
    min_function_length: int = 50
    max_sequence_length: int = 20000
    train_test_split: float = 0.85
    validation_split: float = 0.1
    
    # Model training
    model_config: ModelConfig = None
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    
    # Evaluation
    evaluation_sample_size: int = 9731  # As mentioned in paper
    
    # Output directories
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()

class SmartContractEvaluator:
    """
    Comprehensive evaluation framework implementing metrics from the paper.
    
    Includes semantic similarity, edit distance, token frequency analysis,
    and structural fidelity measurements.
    """
    
    def __init__(self):
        """Initialize the evaluator with required NLP models and scorers.

        Sets up the sentence transformer model for semantic similarity,
        ROUGE scorer for text overlap metrics, and ensures NLTK punkt
        tokenizer is available.

        Raises:
            Exception: If sentence transformer model fails to load.
        """
        # Initialize evaluation models
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def compute_semantic_similarity(self, original: str, decompiled: str) -> float:
        """Compute semantic similarity using sentence transformers.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Cosine similarity score between 0 and 1, where 1 indicates
            identical semantic meaning.
        """
        try:
            # Encode both texts
            embeddings = self.semantic_model.encode([original, decompiled])
            
            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def compute_normalized_edit_distance(self, original: str, decompiled: str) -> float:
        """Compute normalized edit distance (Levenshtein distance).

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Normalized edit distance between 0 and 1, where 0 indicates
            identical strings.
        """
        try:
            # Normalize whitespace
            original = ' '.join(original.split())
            decompiled = ' '.join(decompiled.split())
            
            # Compute edit distance
            distance = difflib.SequenceMatcher(None, original, decompiled).ratio()
            
            # Return as distance (1 - similarity)
            return 1.0 - distance
            
        except Exception as e:
            logger.error(f"Error computing edit distance: {e}")
            return 1.0
    
    def compute_bleu_score(self, original: str, decompiled: str) -> float:
        """Compute BLEU score for code similarity.

        Uses smoothing to handle short sequences appropriately.

        Args:
            original: Original Solidity code (reference).
            decompiled: Decompiled Solidity code (candidate).

        Returns:
            BLEU score between 0 and 1, where 1 indicates perfect match.
        """
        try:
            # Tokenize
            reference = [original.split()]
            candidate = decompiled.split()
            
            # Use smoothing function for short sequences
            smoothing_function = SmoothingFunction().method1
            
            # Compute BLEU score
            score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
            return float(score)
            
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0
    
    def compute_rouge_score(self, original: str, decompiled: str) -> float:
        """Compute ROUGE-L score.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            ROUGE-L F1 score between 0 and 1.
        """
        try:
            scores = self.rouge_scorer.score(original, decompiled)
            return float(scores['rougeL'].fmeasure)
            
        except Exception as e:
            logger.error(f"Error computing ROUGE score: {e}")
            return 0.0
    
    def compute_token_accuracy(self, original: str, decompiled: str) -> float:
        """Compute token-level accuracy using Jaccard similarity.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Token accuracy between 0 and 1, computed as the intersection
            over union of token sets.
        """
        try:
            original_tokens = set(original.split())
            decompiled_tokens = set(decompiled.split())
            
            if not original_tokens:
                return 1.0 if not decompiled_tokens else 0.0
            
            intersection = original_tokens.intersection(decompiled_tokens)
            union = original_tokens.union(decompiled_tokens)
            
            return len(intersection) / len(union) if union else 1.0
            
        except Exception as e:
            logger.error(f"Error computing token accuracy: {e}")
            return 0.0
    
    def analyze_structural_preservation(self, original: str, decompiled: str) -> float:
        """Analyze how well control flow and structure are preserved.

        Compares counts of structural keywords (if, else, for, while,
        function, return, require, assert, revert, braces, parentheses).

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Structural preservation score between 0 and 1, where 1 indicates
            identical structural element counts.
        """
        try:
            # Count key structural elements
            structural_keywords = [
                'if', 'else', 'for', 'while', 'function', 'return',
                'require', 'assert', 'revert', '{', '}', '(', ')'
            ]
            
            original_counts = {}
            decompiled_counts = {}
            
            for keyword in structural_keywords:
                original_counts[keyword] = original.count(keyword)
                decompiled_counts[keyword] = decompiled.count(keyword)
            
            # Compute similarity of structural element counts
            total_difference = 0
            total_count = 0
            
            for keyword in structural_keywords:
                orig_count = original_counts[keyword]
                decomp_count = decompiled_counts[keyword]
                
                if orig_count + decomp_count > 0:
                    difference = abs(orig_count - decomp_count) / max(orig_count + decomp_count, 1)
                    total_difference += difference
                    total_count += 1
            
            if total_count == 0:
                return 1.0
            
            return max(0.0, 1.0 - (total_difference / total_count))
            
        except Exception as e:
            logger.error(f"Error analyzing structural preservation: {e}")
            return 0.0
    
    def extract_function_metadata(self, code: str) -> Dict[str, Any]:
        """Extract function metadata from Solidity code.

        Args:
            code: Solidity source code to analyze.

        Returns:
            Dictionary containing extracted metadata including visibility,
            payable status, view/pure modifiers, and presence of require
            statements.
        """
        try:
            metadata = {
                'has_function_keyword': 'function' in code,
                'visibility': None,
                'is_payable': 'payable' in code,
                'is_view': 'view' in code or 'pure' in code,
                'has_return': 'return' in code,
                'has_require': 'require' in code
            }
            
            # Extract visibility
            for visibility in ['private', 'internal', 'external', 'public']:
                if visibility in code:
                    metadata['visibility'] = visibility
                    break
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def evaluate_function(self, original: str, decompiled: str, metadata: Optional[Dict] = None) -> EvaluationMetrics:
        """Comprehensive evaluation of a single function decompilation.

        Computes all metrics including semantic similarity, edit distance,
        BLEU, ROUGE-L, token accuracy, and structural preservation.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.
            metadata: Optional metadata about the function for additional
                context in the evaluation results.

        Returns:
            EvaluationMetrics dataclass containing all computed metrics
            and metadata comparison results.
        """
        # Compute all metrics
        semantic_similarity = self.compute_semantic_similarity(original, decompiled)
        normalized_edit_distance = self.compute_normalized_edit_distance(original, decompiled)
        bleu_score = self.compute_bleu_score(original, decompiled)
        rouge_l_score = self.compute_rouge_score(original, decompiled)
        token_accuracy = self.compute_token_accuracy(original, decompiled)
        structural_preservation = self.analyze_structural_preservation(original, decompiled)
        
        # Extract and compare metadata
        original_metadata = self.extract_function_metadata(original)
        decompiled_metadata = self.extract_function_metadata(decompiled)
        
        function_signature_match = (
            original_metadata.get('has_function_keyword') == 
            decompiled_metadata.get('has_function_keyword')
        )
        
        visibility_match = (
            original_metadata.get('visibility') == 
            decompiled_metadata.get('visibility')
        )
        
        return EvaluationMetrics(
            semantic_similarity=semantic_similarity,
            normalized_edit_distance=normalized_edit_distance,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
            token_accuracy=token_accuracy,
            structural_preservation=structural_preservation,
            function_signature_match=function_signature_match,
            visibility_match=visibility_match,
            metadata={
                'original_metadata': original_metadata,
                'decompiled_metadata': decompiled_metadata,
                'function_metadata': metadata
            }
        )

class SmartContractTrainingPipeline:
    """
    Complete training pipeline for smart contract decompilation.
    
    Orchestrates data collection, preprocessing, training, and evaluation
    as described in the paper.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize the training pipeline with the given configuration.

        Args:
            config: TrainingConfig object containing all pipeline settings
                including API keys, paths, and hyperparameters.
        """
        self.config = config
        
        # Create output directories
        for dir_path in [config.data_dir, config.models_dir, config.results_dir]:
            Path(dir_path).mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset_builder = DatasetBuilder(
            config.etherscan_api_key, 
            output_dir=config.data_dir
        )
        
        self.model_trainer = SmartContractModelTrainer(
            config.model_config,
            output_dir=config.models_dir
        )
        
        self.evaluator = SmartContractEvaluator()
    
    def collect_and_prepare_dataset(self) -> Tuple[str, str, str]:
        """Collect contracts and prepare training dataset.

        Loads contract addresses from file or uses sample addresses,
        collects contracts via Etherscan API, processes them into
        function pairs, filters and cleans the dataset, and splits
        into train/validation/test sets.

        Returns:
            Tuple of (train_path, validation_path, test_path) pointing
            to the JSONL dataset files.

        Raises:
            FileNotFoundError: If contract_addresses_file is specified
                but does not exist.
        """
        logger.info("Starting dataset collection and preparation...")
        
        # Load contract addresses
        if self.config.contract_addresses_file:
            with open(self.config.contract_addresses_file, 'r') as f:
                contract_addresses = [line.strip() for line in f if line.strip()]
        else:
            # For demonstration, create a sample list
            # In practice, you would have a comprehensive list of verified contracts
            contract_addresses = [
                "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI token
                "0xA0b86a33E6411a3b4E4c3c4C4e4b5b5b5b5b5b5b",  # Example addresses
                # Add more real verified contract addresses here
            ]
            logger.warning("Using sample contract addresses. Please provide a comprehensive list.")
        
        # Collect contracts
        logger.info(f"Collecting {len(contract_addresses)} contracts...")
        collected = self.dataset_builder.collect_contracts(contract_addresses)
        
        # Process to function pairs
        logger.info("Processing contracts to function pairs...")
        total_pairs = self.dataset_builder.process_contracts_to_function_pairs()
        
        # Filter and clean dataset
        logger.info("Filtering and cleaning dataset...")
        filtered_pairs = self.dataset_builder.filter_and_clean_dataset(
            min_length=self.config.min_function_length,
            max_length=self.config.max_sequence_length
        )
        
        if filtered_pairs < 1000:  # Minimum viable dataset size
            logger.warning(f"Dataset too small ({filtered_pairs} pairs). Consider collecting more contracts.")
        
        # Export dataset
        dataset_path = self.dataset_builder.export_dataset("jsonl")
        
        # Split dataset
        train_path, val_path, test_path = self._split_dataset(dataset_path)
        
        # Print statistics
        stats = self.dataset_builder.get_dataset_statistics()
        logger.info(f"Dataset statistics: {stats}")
        
        return train_path, val_path, test_path
    
    def _split_dataset(self, dataset_path: str) -> Tuple[str, str, str]:
        """Split dataset into train, validation, and test sets.

        Args:
            dataset_path: Path to the complete JSONL dataset file.

        Returns:
            Tuple containing paths to (train_dataset, validation_dataset,
            test_dataset) JSONL files.
        """
        # Load data
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data, 
            test_size=1 - self.config.train_test_split,
            random_state=42
        )
        
        # Second split: separate validation set from training
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=self.config.validation_split,
            random_state=42
        )
        
        # Save splits
        data_dir = Path(self.config.data_dir)
        
        train_path = data_dir / "train_dataset.jsonl"
        val_path = data_dir / "validation_dataset.jsonl"
        test_path = data_dir / "test_dataset.jsonl"
        
        for data_split, path in [(train_data, train_path), (val_data, val_path), (test_data, test_path)]:
            with open(path, 'w') as f:
                for item in data_split:
                    f.write(json.dumps(item) + '\n')
        
        logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return str(train_path), str(val_path), str(test_path)
    
    def train_model(self, train_path: str, val_path: str) -> str:
        """Train the model on the prepared dataset.

        Args:
            train_path: Path to training dataset JSONL file.
            val_path: Path to validation dataset JSONL file.

        Returns:
            Path to the directory containing the trained model weights
            and configuration.
        """
        logger.info("Starting model training...")
        
        model_path = self.model_trainer.train(
            train_dataset_path=train_path,
            eval_dataset_path=val_path,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs
        )
        
        logger.info(f"Model training completed. Model saved to {model_path}")
        return model_path
    
    def evaluate_model(self, model_path: str, test_path: str) -> Dict[str, Any]:
        """Comprehensive evaluation of the trained model.

        Loads the trained model, runs inference on test samples, and
        computes aggregate statistics across all evaluation metrics.
        Results are saved to a timestamped JSON file.

        Args:
            model_path: Path to trained model directory.
            test_path: Path to test dataset JSONL file.

        Returns:
            Dictionary containing aggregate statistics (mean, std, median,
            min, max, percentiles) for all metrics.
        """
        logger.info("Starting model evaluation...")
        
        # Load test data
        test_data = []
        with open(test_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        # Sample for evaluation if dataset is large
        if len(test_data) > self.config.evaluation_sample_size:
            test_data = np.random.choice(
                test_data, 
                size=self.config.evaluation_sample_size, 
                replace=False
            ).tolist()
        
        # Initialize decompiler
        decompiler = SmartContractDecompiler(model_path)
        
        # Evaluate each function
        results = []
        
        for item in tqdm(test_data, desc="Evaluating functions"):
            try:
                # Generate decompiled code
                decompiled = decompiler.decompile_tac_to_solidity(
                    item['input'],
                    metadata=item.get('metadata', {})
                )
                
                # Evaluate
                metrics = self.evaluator.evaluate_function(
                    item['output'],
                    decompiled,
                    item.get('metadata', {})
                )
                
                results.append({
                    'original': item['output'],
                    'decompiled': decompiled,
                    'metrics': asdict(metrics),
                    'metadata': item.get('metadata', {})
                })
                
            except Exception as e:
                logger.error(f"Error evaluating function: {e}")
                continue
        
        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_statistics(results)
        
        # Save detailed results
        results_path = Path(self.config.results_dir) / f"evaluation_results_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'aggregate_statistics': aggregate_stats,
                'detailed_results': results
            }, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        return aggregate_stats
    
    def _compute_aggregate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics from evaluation results.

        Args:
            results: List of dictionaries containing individual evaluation
                results with metrics for each function.

        Returns:
            Dictionary containing aggregate statistics (mean, std, median,
            min, max, percentiles) for each metric, plus paper-specific
            threshold metrics.
        """
        if not results:
            return {}
        
        # Extract metric values
        metrics_data = {}
        for result in results:
            for key, value in result['metrics'].items():
                if key not in ['metadata']:
                    if key not in metrics_data:
                        metrics_data[key] = []
                    
                    if isinstance(value, bool):
                        metrics_data[key].append(float(value))
                    elif isinstance(value, (int, float)):
                        metrics_data[key].append(float(value))
        
        # Compute statistics
        stats = {}
        for metric, values in metrics_data.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
                
                # Add percentiles for key metrics
                if metric in ['semantic_similarity', 'normalized_edit_distance']:
                    stats[metric]['percentiles'] = {
                        '25th': np.percentile(values, 25),
                        '75th': np.percentile(values, 75),
                        '90th': np.percentile(values, 90),
                        '95th': np.percentile(values, 95)
                    }
        
        # Add paper-specific metrics
        if 'semantic_similarity' in metrics_data:
            semantic_values = metrics_data['semantic_similarity']
            stats['paper_metrics'] = {
                'functions_above_0_8_semantic_similarity': sum(1 for v in semantic_values if v > 0.8) / len(semantic_values),
                'functions_above_0_9_semantic_similarity': sum(1 for v in semantic_values if v > 0.9) / len(semantic_values),
            }
        
        if 'normalized_edit_distance' in metrics_data:
            edit_values = metrics_data['normalized_edit_distance']
            stats['paper_metrics']['functions_below_0_4_edit_distance'] = sum(1 for v in edit_values if v < 0.4) / len(edit_values)
        
        return stats
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training and evaluation pipeline.

        Executes the full workflow: dataset collection and preparation,
        model training, and comprehensive evaluation.

        Returns:
            Dictionary containing final evaluation results with aggregate
            statistics for all metrics.
        """
        logger.info("Starting complete smart contract decompilation pipeline...")
        
        # Step 1: Collect and prepare dataset
        train_path, val_path, test_path = self.collect_and_prepare_dataset()
        
        # Step 2: Train model
        model_path = self.train_model(train_path, val_path)
        
        # Step 3: Evaluate model
        evaluation_results = self.evaluate_model(model_path, test_path)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Key results:")
        logger.info(f"- Semantic similarity: {evaluation_results.get('semantic_similarity', {}).get('mean', 'N/A'):.3f}")
        logger.info(f"- Edit distance: {evaluation_results.get('normalized_edit_distance', {}).get('mean', 'N/A'):.3f}")
        
        return evaluation_results

def main():
    """Run an example demonstration of the complete training pipeline.

    Sets up logging, loads configuration from environment variables,
    and executes the full data collection, training, and evaluation
    pipeline with reduced parameters suitable for demonstration.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for API key
    api_key = os.getenv('ETHERSCAN_API_KEY')
    if not api_key:
        print("Please set ETHERSCAN_API_KEY environment variable")
        return
    
    # Create configuration
    config = TrainingConfig(
        etherscan_api_key=api_key,
        target_dataset_size=1000,  # Smaller for demo
        evaluation_sample_size=100,  # Smaller for demo
        num_epochs=1,  # Quick training for demo
        batch_size=2   # Smaller batch for demo
    )
    
    # Run pipeline
    pipeline = SmartContractTrainingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
