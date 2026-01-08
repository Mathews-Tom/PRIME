# PRED-001: Embedding Predictor Task Breakdown

**Epic:** PRED-001 - Implement Embedding Predictor
**Generated:** 2026-01-08
**Total Story Points:** 31 (with 20% buffer: 37)

---

## Task Summary

| Story ID | Title | Points | Dependencies | Critical Path |
|----------|-------|--------|--------------|---------------|
| PRED-001-S1 | Foundation: Config, Types, Exceptions | 3 | ENC-001 | ✓ |
| PRED-001-S2 | Core: EmbeddingPredictor Module | 8 | PRED-001-S1 | ✓ |
| PRED-001-S3 | Training: Loss Functions and Lightning Module | 8 | PRED-001-S2 | ✓ |
| PRED-001-S4 | Optimization: Static KV Cache and torch.compile | 5 | PRED-001-S2 | |
| PRED-001-S5 | Testing: Comprehensive Test Suite | 5 | PRED-001-S3 | ✓ |
| PRED-001-S6 | Export: ONNX and Checkpoint Management | 2 | PRED-001-S2 | |

---

## Critical Path

```
ENC-001 (external)
    │
    └──▶ PRED-001-S1 (3)
              │
              └──▶ PRED-001-S2 (8) ──┬──▶ PRED-001-S3 (8) ──▶ PRED-001-S5 (5)
                                     │
                                     ├──▶ PRED-001-S4 (5)
                                     │
                                     └──▶ PRED-001-S6 (2)
```

**Critical Path Duration:** 24 story points (S1 → S2 → S3 → S5)
**Full Epic Duration:** 31 story points (all stories)

---

## Story Details

### PRED-001-S1: Foundation - Config, Types, Exceptions

**Points:** 3
**Priority:** P0
**Dependencies:** ENC-001 (for Encoder protocol)

#### Description
Establish the foundational types, configuration schemas, and exception hierarchy for the Predictor module. This includes PredictorConfig, TrainingConfig, and custom exceptions.

#### Acceptance Criteria
- [ ] `src/prime/core/` directory structure created
- [ ] `predictor_config.py` with PredictorConfig and TrainingConfig (frozen Pydantic models)
- [ ] PredictorError, CheckpointError exceptions
- [ ] All types pass mypy --strict
- [ ] Unit tests for config validation

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| PRED-001-S1-T1 | Create directory structure and __init__.py | XS |
| PRED-001-S1-T2 | Implement PredictorConfig with all fields | S |
| PRED-001-S1-T3 | Implement TrainingConfig with LR, batch, epochs | S |
| PRED-001-S1-T4 | Implement exception hierarchy | S |
| PRED-001-S1-T5 | Add unit tests for config validation | S |

#### Target Files
- `src/prime/core/__init__.py`
- `src/prime/core/predictor_config.py`
- `tests/unit/core/test_predictor_config.py`

---

### PRED-001-S2: Core - EmbeddingPredictor Module

**Points:** 8
**Priority:** P0
**Dependencies:** PRED-001-S1

#### Description
Implement the core EmbeddingPredictor PyTorch module with transformer architecture, [PRED] token, input/output projections, and L2 normalization.

#### Acceptance Criteria
- [ ] PredictorTransformerBlock with pre-norm architecture
- [ ] EmbeddingPredictor class with forward() method
- [ ] Context and query projections (D → H)
- [ ] Learnable [PRED] token prepended to sequence
- [ ] Positional embeddings
- [ ] Output projection (H → D) with L2 normalization
- [ ] Shape validation at forward() entry
- [ ] save_checkpoint() and load_checkpoint() methods

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| PRED-001-S2-T1 | Implement PredictorTransformerBlock | M |
| PRED-001-S2-T2 | Implement input projections (context, query) | S |
| PRED-001-S2-T3 | Implement [PRED] token and positional embeddings | M |
| PRED-001-S2-T4 | Implement forward() with transformer layers | L |
| PRED-001-S2-T5 | Implement output projection and L2 normalize | S |
| PRED-001-S2-T6 | Implement shape validation | S |
| PRED-001-S2-T7 | Implement checkpoint save/load | M |
| PRED-001-S2-T8 | Implement _init_weights() | S |

#### Target Files
- `src/prime/core/predictor.py`
- `src/prime/core/__init__.py` (update exports)

---

### PRED-001-S3: Training - Loss Functions and Lightning Module

**Points:** 8
**Priority:** P0
**Dependencies:** PRED-001-S2

#### Description
Implement InfoNCE contrastive loss function and PyTorch Lightning training module with training/validation steps, optimizer configuration, and Y-Encoder integration.

#### Acceptance Criteria
- [ ] info_nce_loss() with in-batch negatives
- [ ] Temperature parameter τ = 0.07
- [ ] PredictorLightningModule class
- [ ] training_step() with Y-Encoder targets
- [ ] validation_step() with metrics logging
- [ ] configure_optimizers() with AdamW and scheduler
- [ ] Gradient checkpointing support
- [ ] Mixed precision support

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| PRED-001-S3-T1 | Implement info_nce_loss() function | M |
| PRED-001-S3-T2 | Implement PredictorLightningModule init | M |
| PRED-001-S3-T3 | Implement training_step() | L |
| PRED-001-S3-T4 | Implement validation_step() | M |
| PRED-001-S3-T5 | Implement configure_optimizers() | M |
| PRED-001-S3-T6 | Add gradient checkpointing support | S |
| PRED-001-S3-T7 | Add mixed precision support | S |

#### Target Files
- `src/prime/training/__init__.py`
- `src/prime/training/losses.py`
- `src/prime/training/trainer.py`

---

### PRED-001-S4: Optimization - Static KV Cache and torch.compile

**Points:** 5
**Priority:** P1
**Dependencies:** PRED-001-S2

#### Description
Implement OptimizedPredictor wrapper with Static KV Cache and torch.compile() for 4× inference speedup via CUDA graph capture.

#### Acceptance Criteria
- [ ] OptimizedPredictor wrapper class
- [ ] Static batch padding for CUDA graph compatibility
- [ ] torch.compile() with reduce-overhead mode
- [ ] warmup() method for compilation trigger
- [ ] Flash Attention integration via sdpa_kernel
- [ ] create_predictor() factory function
- [ ] Benchmark showing 4× speedup

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| PRED-001-S4-T1 | Implement OptimizedPredictor class | M |
| PRED-001-S4-T2 | Implement static batch padding | S |
| PRED-001-S4-T3 | Implement torch.compile integration | M |
| PRED-001-S4-T4 | Implement warmup() method | S |
| PRED-001-S4-T5 | Implement create_predictor() factory | S |
| PRED-001-S4-T6 | Create latency benchmark | M |

#### Target Files
- `src/prime/core/optimized.py`
- `tests/benchmark/predictor/test_latency.py`

---

### PRED-001-S5: Testing - Comprehensive Test Suite

**Points:** 5
**Priority:** P0
**Dependencies:** PRED-001-S3

#### Description
Create comprehensive test suite achieving ≥85% coverage including forward pass, gradient flow, checkpoint, loss function, and training tests.

#### Acceptance Criteria
- [ ] Forward pass shape tests
- [ ] Output L2 normalization tests
- [ ] Batch inference consistency tests
- [ ] Invalid input raises tests
- [ ] Gradient flow tests
- [ ] Checkpoint save/load tests
- [ ] InfoNCE loss tests
- [ ] Test coverage ≥85%

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| PRED-001-S5-T1 | Create test fixtures (config, predictor, sample_input) | S |
| PRED-001-S5-T2 | Implement forward pass tests | M |
| PRED-001-S5-T3 | Implement input validation tests | S |
| PRED-001-S5-T4 | Implement gradient flow tests | S |
| PRED-001-S5-T5 | Implement checkpoint tests | M |
| PRED-001-S5-T6 | Implement loss function tests | M |
| PRED-001-S5-T7 | Verify ≥85% coverage | S |

#### Target Files
- `tests/unit/core/test_predictor.py`
- `tests/unit/training/test_losses.py`
- `tests/unit/training/test_trainer.py`

---

### PRED-001-S6: Export - ONNX and Checkpoint Management

**Points:** 2
**Priority:** P2
**Dependencies:** PRED-001-S2

#### Description
Implement ONNX export functionality for optimized deployment and enhanced checkpoint management with versioning.

#### Acceptance Criteria
- [ ] export_onnx() method
- [ ] ONNX model validation
- [ ] Checkpoint versioning with metadata
- [ ] Checkpoint integrity verification

#### Subtasks
| ID | Task | Effort |
|----|------|--------|
| PRED-001-S6-T1 | Implement export_onnx() | M |
| PRED-001-S6-T2 | Add checkpoint metadata and versioning | S |
| PRED-001-S6-T3 | Test ONNX export | S |

#### Target Files
- `src/prime/core/predictor.py` (update)
- `tests/unit/core/test_onnx_export.py`

---

## Effort Legend

| Size | Story Points | Time Estimate |
|------|--------------|---------------|
| XS | 0.5 | <1 hour |
| S | 1 | 1-2 hours |
| M | 2 | 2-4 hours |
| L | 3 | 4-8 hours |
| XL | 5 | 1-2 days |

---

## Dependencies on Other Epics

**Blocks:**
- API-001: Uses Predictor for embedding prediction in PRIME orchestration

**Blocked By:**
- ENC-001: Requires Y-Encoder for training targets

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Training doesn't converge | Medium | High | Careful LR scheduling, gradient clipping |
| Latency exceeds 30ms target | Medium | Medium | Static KV Cache, torch.compile |
| OOM during training | Medium | Medium | Gradient checkpointing, mixed precision |
| Y-Encoder drift during training | Low | Medium | EMA update, frozen Y-Encoder option |
| CUDA graph capture fails | Low | Medium | Fallback to non-compiled mode |

---

## Key Algorithms

### Forward Pass
```
1. Project context: B × N × D → B × N × H
2. Project query: B × D → B × 1 × H
3. Prepend [PRED]: B × (1+N+1) × H
4. Add positional embeddings
5. Apply transformer layers (bidirectional)
6. Extract [PRED] position: B × H
7. Project output: B × H → B × D
8. L2 normalize
```

### InfoNCE Loss
```
L = -log( exp(sim(p, t⁺) / τ) / Σᵢ exp(sim(p, tᵢ) / τ) )
where τ = 0.07, sim = cosine similarity
```

### Static KV Cache Optimization
```
1. Pre-allocate KV cache tensors at max shapes
2. Pad input batches to static size
3. Enable CUDA graph capture via torch.compile
4. Use Flash Attention backend
5. Remove padding from output
```

---

## Definition of Done (Epic Level)

- [ ] All P0 stories completed
- [ ] Test coverage ≥85%
- [ ] mypy --strict passes
- [ ] ruff check passes
- [ ] Inference latency <30ms p50
- [ ] Forward pass produces correct shape (B × 1024)
- [ ] Output is L2 normalized
- [ ] InfoNCE loss computes correctly
- [ ] Training converges on sample dataset
- [ ] Ready for API-001 integration
