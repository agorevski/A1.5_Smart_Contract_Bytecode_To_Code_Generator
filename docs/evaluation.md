# Evaluation Metrics

This document explains the comprehensive evaluation framework used to assess decompilation quality.

## Overview

The system uses multiple complementary metrics to evaluate decompilation quality across semantic, syntactic, and structural dimensions.

## Paper Target Metrics

| Metric | Target Value | Description |
|--------|--------------|-------------|
| **Semantic Similarity (avg)** | 0.82 | Average embedding similarity |
| **Functions > 0.8 similarity** | 78.3% | High-quality decompilations |
| **Functions < 0.4 edit distance** | 82.5% | Syntactically accurate |

## Semantic Preservation Metrics

### Semantic Similarity

Measures how well the decompiled code preserves the meaning of the original using Sentence-BERT embeddings.

**Method**:
```python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
embedding1 = model.encode(original_code)
embedding2 = model.encode(decompiled_code)

# Calculate cosine similarity
similarity = 1 - cosine(embedding1, embedding2)
```

**Interpretation**:
- **0.95-1.0**: Nearly perfect semantic match
- **0.85-0.95**: Excellent quality (peak in distribution)
- **0.70-0.85**: Good quality
- **0.50-0.70**: Moderate quality
- **< 0.50**: Poor quality

**Distribution**: Bimodal with peaks at 0.85 and 0.95

### BLEU Score

Measures n-gram overlap between original and decompiled code.

**Method**:
```python
from nltk.translate.bleu_score import sentence_bleu

reference = [original_code.split()]
candidate = decompiled_code.split()

bleu = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
```

**Interpretation**:
- **0.8-1.0**: Excellent match
- **0.6-0.8**: Good match
- **0.4-0.6**: Moderate match
- **< 0.4**: Poor match

### ROUGE-L Score

Measures longest common subsequence, capturing sequential similarity.

**Method**:
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
scores = scorer.score(original_code, decompiled_code)
rouge_l = scores['rougeL'].fmeasure
```

## Syntactic Accuracy Metrics

### Normalized Edit Distance

Measures character-level similarity using Levenshtein distance.

**Method**:
```python
from Levenshtein import distance

edit_dist = distance(original_code, decompiled_code)
max_len = max(len(original_code), len(decompiled_code))
normalized_distance = edit_dist / max_len
```

**Interpretation**:
- **0.0-0.2**: Excellent match
- **0.2-0.4**: Good match (paper target)
- **0.4-0.6**: Moderate match
- **> 0.6**: Poor match

**Paper Result**: 82.5% of functions achieve < 0.4 normalized distance

### Token-Level Accuracy

Measures token-by-token correctness.

**Method**:
```python
def token_accuracy(original, decompiled):
    orig_tokens = original.split()
    decomp_tokens = decompiled.split()
    
    # Align tokens and count matches
    matches = sum(1 for o, d in zip(orig_tokens, decomp_tokens) if o == d)
    total = max(len(orig_tokens), len(decomp_tokens))
    
    return matches / total
```

### Exact Match Rate

Percentage of functions that are perfectly decompiled.

**Method**:
```python
def exact_match(original, decompiled):
    # Normalize whitespace
    orig_norm = ' '.join(original.split())
    decomp_norm = ' '.join(decompiled.split())
    
    return orig_norm == decomp_norm
```

## Structural Fidelity Metrics

### Control Flow Preservation

Evaluates preservation of if/else, loops, and other control structures.

**Method**:
```python
import re

def count_control_structures(code):
    structures = {
        'if': len(re.findall(r'\bif\s*\(', code)),
        'else': len(re.findall(r'\belse\b', code)),
        'while': len(re.findall(r'\bwhile\s*\(', code)),
        'for': len(re.findall(r'\bfor\s*\(', code)),
        'require': len(re.findall(r'\brequire\s*\(', code)),
    }
    return structures

def control_flow_preservation(original, decompiled):
    orig_struct = count_control_structures(original)
    decomp_struct = count_control_structures(decompiled)
    
    # Calculate similarity
    total_diff = sum(abs(orig_struct[k] - decomp_struct.get(k, 0)) 
                     for k in orig_struct)
    total_structures = sum(orig_struct.values())
    
    return 1 - (total_diff / max(total_structures, 1))
```

### Function Signature Accuracy

Evaluates correct recovery of function names, parameters, and modifiers.

**Components**:
- Function name match
- Parameter count and types
- Visibility modifier (public/external/internal/private)
- State mutability (view/pure/payable)
- Return type

**Method**:
```python
def parse_function_signature(code):
    pattern = r'function\s+(\w+)\s*\((.*?)\)\s+(public|external|internal|private)(?:\s+(view|pure|payable))?\s*(?:returns\s*\((.*?)\))?'
    match = re.search(pattern, code)
    
    if match:
        return {
            'name': match.group(1),
            'params': match.group(2),
            'visibility': match.group(3),
            'mutability': match.group(4),
            'returns': match.group(5)
        }
    return None

def signature_accuracy(original, decompiled):
    orig_sig = parse_function_signature(original)
    decomp_sig = parse_function_signature(decompiled)
    
    if not orig_sig or not decomp_sig:
        return 0.0
    
    score = 0.0
    if orig_sig['name'] == decomp_sig['name']:
        score += 0.3
    if orig_sig['visibility'] == decomp_sig['visibility']:
        score += 0.2
    if orig_sig['mutability'] == decomp_sig['mutability']:
        score += 0.2
    if orig_sig['params'] == decomp_sig['params']:
        score += 0.2
    if orig_sig['returns'] == decomp_sig['returns']:
        score += 0.1
    
    return score
```

### Code Length Correlation

Measures similarity in code length (as proxy for complexity).

**Method**:
```python
def length_correlation(original, decompiled):
    orig_len = len(original.split())
    decomp_len = len(decompiled.split())
    
    ratio = min(orig_len, decomp_len) / max(orig_len, decomp_len)
    return ratio
```

## Advanced Analysis Metrics

### Token Frequency Distribution

Compares distribution of tokens between original and decompiled.

**Method**:
```python
from collections import Counter
from scipy.stats import wasserstein_distance

def token_distribution_similarity(original, decompiled):
    orig_tokens = Counter(original.split())
    decomp_tokens = Counter(decompiled.split())
    
    # Get all unique tokens
    all_tokens = set(orig_tokens.keys()) | set(decomp_tokens.keys())
    
    # Create frequency vectors
    orig_freq = [orig_tokens.get(t, 0) for t in all_tokens]
    decomp_freq = [decomp_tokens.get(t, 0) for t in all_tokens]
    
    # Calculate Wasserstein distance
    distance = wasserstein_distance(orig_freq, decomp_freq)
    similarity = 1 / (1 + distance)
    
    return similarity
```

### Security Pattern Preservation

Evaluates preservation of critical security patterns.

**Patterns**:
- `require` statements
- Access control modifiers
- Overflow checks
- Zero-address checks
- Reentrancy guards

**Method**:
```python
def security_pattern_preservation(original, decompiled):
    patterns = [
        r'\brequire\s*\(',
        r'\bmodifier\s+\w+',
        r'\bmsg\.sender\b',
        r'address\(0\)',
        r'nonReentrant'
    ]
    
    scores = []
    for pattern in patterns:
        orig_count = len(re.findall(pattern, original))
        decomp_count = len(re.findall(pattern, decompiled))
        
        if orig_count == 0 and decomp_count == 0:
            scores.append(1.0)
        elif orig_count == 0:
            scores.append(0.0)
        else:
            scores.append(min(decomp_count / orig_count, 1.0))
    
    return sum(scores) / len(scores)
```

### Complexity Score Alignment

Measures cyclomatic complexity similarity.

**Method**:
```python
def cyclomatic_complexity(code):
    # Count decision points
    decision_points = (
        len(re.findall(r'\bif\s*\(', code)) +
        len(re.findall(r'\belse\s+if\s*\(', code)) +
        len(re.findall(r'\bwhile\s*\(', code)) +
        len(re.findall(r'\bfor\s*\(', code)) +
        len(re.findall(r'\b\|\|\b', code)) +
        len(re.findall(r'\b&&\b', code))
    )
    return decision_points + 1

def complexity_alignment(original, decompiled):
    orig_complexity = cyclomatic_complexity(original)
    decomp_complexity = cyclomatic_complexity(decompiled)
    
    return min(orig_complexity, decomp_complexity) / max(orig_complexity, decomp_complexity)
```

## Composite Metrics

### Overall Quality Score

Weighted combination of all metrics:

```python
def overall_quality_score(original, decompiled):
    weights = {
        'semantic_similarity': 0.35,
        'edit_distance': 0.20,
        'bleu_score': 0.15,
        'structural_preservation': 0.15,
        'signature_accuracy': 0.10,
        'security_patterns': 0.05
    }
    
    scores = {
        'semantic_similarity': semantic_similarity(original, decompiled),
        'edit_distance': 1 - normalized_edit_distance(original, decompiled),
        'bleu_score': bleu_score(original, decompiled),
        'structural_preservation': control_flow_preservation(original, decompiled),
        'signature_accuracy': signature_accuracy(original, decompiled),
        'security_patterns': security_pattern_preservation(original, decompiled)
    }
    
    overall = sum(scores[k] * weights[k] for k in weights)
    return overall, scores
```

## Evaluation Pipeline

### Complete Evaluation Function

```python
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    semantic_similarity: float
    edit_distance: float
    bleu_score: float
    rouge_l: float
    control_flow: float
    signature_accuracy: float
    length_correlation: float
    complexity_alignment: float
    security_patterns: float
    overall_score: float

def evaluate_function(original, decompiled):
    return EvaluationResult(
        semantic_similarity=semantic_similarity(original, decompiled),
        edit_distance=normalized_edit_distance(original, decompiled),
        bleu_score=bleu_score(original, decompiled),
        rouge_l=rouge_l_score(original, decompiled),
        control_flow=control_flow_preservation(original, decompiled),
        signature_accuracy=signature_accuracy(original, decompiled),
        length_correlation=length_correlation(original, decompiled),
        complexity_alignment=complexity_alignment(original, decompiled),
        security_patterns=security_pattern_preservation(original, decompiled),
        overall_score=overall_quality_score(original, decompiled)[0]
    )
```

### Batch Evaluation

```python
def evaluate_dataset(test_data):
    results = []
    
    for item in test_data:
        original = item['output']
        decompiled = model.decompile(item['input'])
        
        result = evaluate_function(original, decompiled)
        results.append(result)
    
    # Aggregate statistics
    stats = {
        'avg_semantic_similarity': np.mean([r.semantic_similarity for r in results]),
        'avg_edit_distance': np.mean([r.edit_distance for r in results]),
        'avg_bleu_score': np.mean([r.bleu_score for r in results]),
        'high_similarity_pct': sum(1 for r in results if r.semantic_similarity > 0.8) / len(results) * 100,
        'low_edit_distance_pct': sum(1 for r in results if r.edit_distance < 0.4) / len(results) * 100
    }
    
    return results, stats
```

## Visualization

### Distribution Plots

```python
import matplotlib.pyplot as plt

def plot_metric_distribution(results, metric_name):
    values = [getattr(r, metric_name) for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel(metric_name.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric_name.replace("_", " ").title()}')
    plt.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
    plt.legend()
    plt.savefig(f'{metric_name}_distribution.png')
    plt.close()
```

### Comparison Tables

```python
def generate_comparison_table(results):
    df = pd.DataFrame([
        {
            'Metric': 'Semantic Similarity',
            'Mean': np.mean([r.semantic_similarity for r in results]),
            'Median': np.median([r.semantic_similarity for r in results]),
            'Std': np.std([r.semantic_similarity for r in results])
        },
        {
            'Metric': 'Edit Distance',
            'Mean': np.mean([r.edit_distance for r in results]),
            'Median': np.median([r.edit_distance for r in results]),
            'Std': np.std([r.edit_distance for r in results])
        },
        # ... more metrics
    ])
    
    return df
```

## Benchmarking

### Comparison with Baselines

| Metric | Traditional Decompilers | Our Approach | Improvement |
|--------|------------------------|--------------|-------------|
| **Semantic Similarity > 0.8** | 40-50% | 78.3% | +56% |
| **Edit Distance < 0.4** | 50-60% | 82.5% | +38% |
| **BLEU Score** | 0.3-0.4 | 0.6-0.7 | +75% |

## Next Steps

- Review [Training Pipeline](training-pipeline.md) for model training
- Check [Model Details](model-details.md) for configuration
- See [Usage Guide](usage.md) for running evaluations
- Explore [Comparisons](comparisons.md) for detailed benchmarks
