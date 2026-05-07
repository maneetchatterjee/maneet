# PyTorch 2.0+ Checkpoint Loading Fix

## Issue
The `test_save_load_checkpoint` test was failing with PyTorch 2.0+.

## Root Cause
PyTorch 2.0 introduced a breaking change where `torch.load()` now uses `weights_only=True` by default for security reasons. This prevents loading checkpoints that contain arbitrary Python objects.

Our checkpoints include:
- Model state dictionaries (tensors) ✓
- Optimizer state dictionaries (contains tensors and Python objects) ✗
- RNG state (contains Python objects from `random.getstate()` and `np.random.get_state()`) ✗
- Metadata (Python dict) ✗

## Solution
Added `weights_only=False` parameter to all `torch.load()` calls throughout the codebase.

### Files Modified
1. `src/utils/checkpointing.py` - Main checkpoint loading utility
2. `src/algorithms/sac/sac.py` - SAC model loading
3. `src/algorithms/dreamer/dreamer.py` - Dreamer model loading
4. `src/algorithms/hierarchical/hierarchical.py` - Hierarchical model loading

### Example Fix
```python
# Before (fails in PyTorch 2.0+):
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# After (works with all PyTorch versions):
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
```

## Security Considerations
Setting `weights_only=False` is safe in this context because:
1. **Trusted Source**: These checkpoints are created by our own training code
2. **Internal Use**: Checkpoints are not loaded from external/untrusted sources
3. **Necessary**: We need to preserve optimizer state and RNG state for reproducibility
4. **Expected Behavior**: This matches PyTorch's behavior before version 2.0

If loading checkpoints from untrusted sources in the future, consider:
- Keeping `weights_only=True` (default in PyTorch 2.0+)
- Only loading model weights, not optimizer/RNG state
- Validating checkpoint contents before loading

## Testing
The fix allows all tests to pass, including:
- ✅ `test_save_load_checkpoint` - Verifies checkpoint save/load cycle
- ✅ All other tests remain passing

## References
- [PyTorch 2.0 Release Notes](https://pytorch.org/docs/stable/notes/serialization.html#torch.load)
- [torch.load() Documentation](https://pytorch.org/docs/stable/generated/torch.load.html)
