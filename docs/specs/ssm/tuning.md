# SSM Threshold Tuning Guide

This guide explains how to tune the Semantic State Monitor (SSM) parameters for optimal boundary detection in your application.

## Overview

The SSM uses variance-based detection with EMA smoothing to identify semantic boundaries in conversations. The key parameters are:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `variance_threshold` | 0.15 | 0.01-0.5 | Threshold θ for boundary detection |
| `smoothing_factor` | 0.3 | 0.0-1.0 | EMA coefficient α for variance smoothing |
| `window_size` | 5 | 1-50 | Number of turns in sliding window |
| `prepare_ratio` | 0.5 | 0.0-1.0 | Ratio of θ for PREPARE state |
| `consolidate_ratio` | 2.0 | 1.0-10.0 | Ratio of θ for RETRIEVE_CONSOLIDATE |

## Understanding Variance Behavior

### L2-Normalized Embeddings

The SSM uses L2-normalized embeddings, which constrains variance values:

```
Typical variance ranges:
- Same topic: 0.001 - 0.01
- Gradual drift: 0.01 - 0.05
- Topic change: 0.05 - 0.15
- Major shift: 0.10 - 0.30
```

These ranges vary by embedding model:
- **MiniLM (384d)**: Tighter clusters, lower variance
- **BGE-large (1024d)**: Higher dimensional, slightly higher variance

### Ward Variance Formula

```python
# 1. Compute centroid
centroid = mean(embeddings)

# 2. Compute L2 distances from centroid
distances = [||e_i - centroid||₂ for e_i in embeddings]

# 3. Return variance of distances
variance = Var(distances)
```

## Tuning `variance_threshold`

The variance threshold θ determines when boundary detection triggers.

### Start with Default (0.15)

This value works well for most conversational applications.

### Adjust Based on Use Case

| Use Case | Recommended θ | Rationale |
|----------|---------------|-----------|
| **Chatbot** | 0.10-0.15 | Moderate sensitivity |
| **Document Q&A** | 0.15-0.25 | Topics shift more explicitly |
| **Customer Support** | 0.08-0.12 | Catch topic changes early |
| **Research Assistant** | 0.20-0.30 | Tolerate more drift |

### Tuning Process

1. **Collect baseline**: Log variance values in production
2. **Identify boundaries**: Mark actual topic changes
3. **Calculate optimal θ**:
   - θ should be > 95th percentile of same-topic variance
   - θ should be < 50th percentile of topic-change variance

```python
# Example tuning analysis
same_topic_variances = [...]  # Collected from logs
topic_change_variances = [...]

# Optimal threshold between these
theta_min = np.percentile(same_topic_variances, 95)
theta_max = np.percentile(topic_change_variances, 50)
theta = (theta_min + theta_max) / 2
```

## Tuning `smoothing_factor`

The EMA smoothing factor α controls responsiveness vs stability.

```python
smoothed = α * current + (1-α) * previous
```

### Tradeoffs

| α Value | Behavior | Use When |
|---------|----------|----------|
| **0.1-0.3** | Very stable, slow to react | Noisy embeddings, false positives |
| **0.3-0.5** | Balanced (default) | General use |
| **0.5-0.7** | Responsive, faster transitions | Need quick boundary detection |
| **0.8-1.0** | Minimal smoothing | Testing, or very clean signals |

### Visual Example

```
Turn:      1    2    3    4    5    6    7    8
Raw var:   0.02 0.03 0.02 0.25 0.28 0.30 0.15 0.12

α=0.3:     0.02 0.02 0.02 0.09 0.15 0.19 0.18 0.16
α=0.7:     0.02 0.03 0.02 0.18 0.25 0.28 0.19 0.14
```

Higher α reacts faster to the spike at turn 4, but also drops faster when variance decreases.

## Tuning `window_size`

The window size determines how much history influences variance calculation.

### Tradeoffs

| Window | Pros | Cons |
|--------|------|------|
| **Small (3-5)** | Fast response to changes | Noisy, sensitive to single turns |
| **Medium (5-10)** | Balanced stability | May lag on quick changes |
| **Large (10-20)** | Very stable | Slow to detect boundaries |

### Memory Considerations

Each embedding in the window uses:
- 384d model: 1.5 KB per turn (384 × 4 bytes)
- 1024d model: 4 KB per turn

For window_size=20 with 1024d: ~80 KB per session.

## Action State Thresholds

### State Determination

```
variance < 0.5θ  → CONTINUE
0.5θ ≤ variance < θ  → PREPARE
θ ≤ variance < 2θ  → RETRIEVE
variance ≥ 2θ  → RETRIEVE_CONSOLIDATE
```

### Customizing `prepare_ratio`

The PREPARE state allows pre-emptive cache warming before full boundary detection.

- **Lower (0.3-0.4)**: Earlier warning, more cache warming
- **Default (0.5)**: Balanced
- **Higher (0.6-0.8)**: Less pre-warming, fewer false positives

### Customizing `consolidate_ratio`

RETRIEVE_CONSOLIDATE indicates a major topic shift requiring memory consolidation.

- **Lower (1.5-2.0)**: More aggressive consolidation
- **Default (2.0)**: Triggered on significant shifts
- **Higher (3.0-5.0)**: Only on dramatic changes

## Domain-Specific Tuning

### Code Assistance

```python
SSMConfig(
    variance_threshold=0.12,  # Code topics are specific
    window_size=7,            # Consider recent context
    smoothing_factor=0.4,     # Moderately responsive
)
```

### Multi-turn FAQ

```python
SSMConfig(
    variance_threshold=0.20,  # FAQ queries vary more
    window_size=3,            # Short interactions
    smoothing_factor=0.5,     # Quick response
)
```

### Research Chat

```python
SSMConfig(
    variance_threshold=0.25,  # Tolerate exploration
    window_size=10,           # Longer context
    smoothing_factor=0.25,    # Very stable
)
```

## Monitoring and Observability

### Key Metrics to Track

```python
state = ssm.get_state()
# state contains:
# - turn_number: Total turns processed
# - window_size: Current buffer size
# - smoothed_variance: Current smoothed value
# - variance_threshold: Configured θ
# - last_action: Current action state
```

### Logging for Tuning

```python
result = ssm.update(user_message)

logger.info(
    "SSM update",
    turn=result.turn_number,
    raw_variance=result.variance,
    smoothed_variance=result.smoothed_variance,
    action=result.action.value,
    boundary_crossed=result.boundary_crossed,
)
```

### Dashboard Metrics

1. **Boundary Detection Rate**: % of turns triggering RETRIEVE/CONSOLIDATE
   - Target: 10-30% for typical conversations

2. **False Positive Rate**: Boundaries detected without actual topic change
   - Track by comparing with user feedback or topic labels

3. **Variance Distribution**: Histogram of smoothed variance values
   - Should show bimodal distribution (same-topic vs boundaries)

## Performance Considerations

### Latency

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Variance calculation | <0.1ms | NumPy vectorized |
| EMA smoothing | <0.01ms | Single float op |
| State determination | <0.01ms | Simple comparisons |
| **Total SSM overhead** | **<0.5ms** | Excluding encoding |

### With Real Encoder

| Encoder | SSM Update p50 | Notes |
|---------|----------------|-------|
| MiniLM (CPU) | ~10ms | Most time in encoding |
| MiniLM (GPU) | ~3ms | GPU accelerated |
| BGE-large (CPU) | ~30ms | Larger model |
| BGE-large (GPU) | ~8ms | GPU accelerated |

## Troubleshooting

### Too Many False Positives

1. Increase `variance_threshold` (try 0.20-0.25)
2. Decrease `smoothing_factor` (try 0.2)
3. Increase `window_size` (try 7-10)

### Missing Actual Boundaries

1. Decrease `variance_threshold` (try 0.10)
2. Increase `smoothing_factor` (try 0.5)
3. Decrease `window_size` (try 3)

### Delayed Detection

1. Increase `smoothing_factor` for faster response
2. Decrease `window_size` to reduce history influence
3. Lower `prepare_ratio` for earlier PREPARE state

## Example Tuning Session

```python
from prime.ssm import SemanticStateMonitor, SSMConfig
from prime.encoder import YEncoder, MINILM_CONFIG

# Start with defaults
encoder = YEncoder(MINILM_CONFIG)
config = SSMConfig(embedding_dim=384)
ssm = SemanticStateMonitor(encoder, config)

# Log variance for 100 conversations
variances = []
for conversation in conversations:
    ssm.reset()
    for turn in conversation.turns:
        result = ssm.update(turn.text)
        variances.append({
            "variance": result.variance,
            "smoothed": result.smoothed_variance,
            "is_boundary": turn.is_topic_change,  # Ground truth
        })

# Analyze
same_topic = [v["smoothed"] for v in variances if not v["is_boundary"]]
boundaries = [v["smoothed"] for v in variances if v["is_boundary"]]

# Find optimal threshold
optimal_theta = (np.percentile(same_topic, 95) + np.percentile(boundaries, 25)) / 2

# Create tuned config
tuned_config = SSMConfig(
    embedding_dim=384,
    variance_threshold=optimal_theta,
)
```
