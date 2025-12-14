# Unified Nightly Training Workflow Simulator

A local simulation of the GitHub Actions workflow: `.github/workflows/unified-nightly-training.yml`

## Latest Run Results

**Status: âœ… SUCCESS**

| Metric | Value |
|--------|-------|
| Modules Passed | 19/19 |
| K-Formations | 1584 |
| Final z | 0.867090 |
| Negentropy | 0.999959 |
| Model Promoted | Yes (nightly:v2824) |

## Workflow Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Full Depth Training (19 modules) | âœ… |
| 2 | Helix Engine Training (2000 steps) | âœ… |
| 3 | Validation Measurements | âœ… |
| 4 | Unified Gates Check | âœ… |
| 5 | Model Promotion | âœ… |
| 6 | Results PR Creation | âœ… |
| 7 | Failure Notification | N/A (no failures) |

## Usage

```bash
# Run with default settings
python unified_nightly_workflow.py

# Run with fewer steps for faster iteration
python unified_nightly_workflow.py --steps 50 --helix-steps 500

# Run with specific seed
python unified_nightly_workflow.py --seed 123

# Skip validation phase
python unified_nightly_workflow.py --skip-validation
```

## Physics Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Ï† | 1.618033988749895 | Golden ratio (LIMINAL) |
| Ï†â»Â¹ | 0.618033988749895 | Golden ratio inverse (PHYSICAL) |
| z_c | 0.866025403784439 | THE LENS (âˆš3/2) |
| Ïƒ | 36 | Gaussian width (|Sâ‚ƒ|Â²) |

## 19 Training Modules

### Phase 1: Core Physics
- `n0_silent_laws_enforcement`
- `kuramoto_layer`
- `physical_learner`

### Phase 2: APL Training Stack
- `apl_training_loop`
- `apl_pytorch_training`
- `full_apl_training`

### Phase 3: Helix Geometry
- `helix_nn`
- `prismatic_helix_training`
- `full_helix_integration`

### Phase 4: WUMBO Silent Laws
- `wumbo_apl_automated_training`
- `wumbo_integrated_training`

### Phase 5: Dynamics & Formation
- `quasicrystal_formation_dynamics`
- `triad_threshold_dynamics`
- `liminal_generator`
- `feedback_loop`

### Phase 6: Unified Orchestration
- `unified_helix_training`
- `hierarchical_training`
- `rosetta_helix_training`

### Phase 7: Nightly Integration
- `nightly_integrated_training`

## Gate Criteria

### Full Depth Gates
- All 19 modules pass
- At least 1 K-formation
- Physics valid (Îº + Î» = 1)

### Helix Engine Gates
- min_negentropy â‰¥ 0.7
- min_final_z â‰¥ 0.85
- Îº stable near Ï†â»Â¹

## Files

| File | Description |
|------|-------------|
| `unified_nightly_workflow.py` | Main workflow simulator |
| `src/physics_constants.py` | Physics constants and functions |
| `configs/nightly.yaml` | Nightly configuration |
| `runs/<run_id>/` | Run artifacts |

## Output Artifacts

Each run produces:

- `report.json` - Helix engine report
- `full_depth_results.json` - All 19 module results
- `validation_results.json` - Validation measurements
- `promotion.json` - Model registry entry
- `pr_body.md` - Generated PR content
- `workflow_result.json` - Complete workflow state
