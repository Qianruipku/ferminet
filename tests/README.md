# Smart Reference File Management System

## Overview

This system controls test behavior through environment variables, allowing easy switching between validation mode and update mode. It's particularly well-suited for scenarios with many similar tests.

## Core Features

### 🎯 One Environment Variable Controls Everything
```bash
# Normal validation mode (default)
python -m pytest test_he.py -v -s

# Update reference files mode  
UPDATE_REFERENCES=true python -m pytest test_he.py -v -s
```

### 🔄 Smart Behavior Switching
- **Validation Mode**: Run training → Compare results → Report differences
- **Update Mode**: Run training → Skip comparison → Automatically update reference files

### 📦 Automatic Backup Protection
- Automatically create `.backup` files before updating
- Prevent accidental overwriting of important reference files
- Easy rollback to previous versions

## Usage

### 1. For New Tests, Inherit from ReferenceTestMixin

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from reference_test_base import ReferenceTestMixin, should_skip_comparison

class MyTest(unittest.TestCase, ReferenceTestMixin):
    def setUp(self):
        super().setUp()
        self.init_reference_management()  # Initialize
    
    def tearDown(self):
        self.cleanup_reference_management()  # Automatically handle updates
        super().tearDown()
    
    def test_something(self):
        # 1. Register generated files that might become reference files
        self._register_generated_file("ref_output.npz", "/path/to/generated.npz")
        self._register_generated_file("ref_stats.csv", "/path/to/generated.csv")
        
        # 2. Run your training/computation
        run_training(self.cfg)
        
        # 3. Use base class comparison method (automatically handles UPDATE_REFERENCES mode)
        comparisons = [
            {
                'generated': '/path/to/generated.npz',
                'reference': '/path/to/ref_output.npz',
                'type': 'npz'  # Optional, auto-detected
            },
            {
                'generated': '/path/to/generated.csv', 
                'reference': '/path/to/ref_stats.csv',
                'type': 'csv'  # Optional, auto-detected
            }
        ]
        
        self.compare_with_references(
            comparisons=comparisons,
            tolerance="1e-6",
            skip_fields=["opt_state", "other_unstable_field"]  # Optional
        )
```

### 2. Base Class Method Advantages

**`compare_with_references()` Method Features**:
- ✅ **Automatic Mode Switching**: Built-in UPDATE_REFERENCES detection
- ✅ **Batch Comparison**: Compare multiple file pairs in one call
- ✅ **Type Auto-Detection**: Automatically determine comparison type based on file extension
- ✅ **Flexible Configuration**: Support custom tolerance and skip fields
- ✅ **Detailed Feedback**: Clear success/failure messages
- ✅ **Error Handling**: Automatically raise AssertionError for test framework

### 3. Workflow Examples

```bash
# Developing new features, need to update reference files
UPDATE_REFERENCES=true python -m pytest test_new_feature.py -v -s

# Verify updates are correct
python -m pytest test_new_feature.py -v -s

# Batch update multiple tests
UPDATE_REFERENCES=true python -m pytest tests/ -k "comparison" -v -s

# Rollback to backup (if needed)
cp ref_output.npz.backup ref_output.npz
```

## Advantages Summary

### ✅ Solves Multi-Test Scenario Pain Points
- **Unified Control**: One environment variable controls all test behavior
- **Batch Operations**: Can update reference files for multiple tests simultaneously
- **Safe Updates**: Automatic backup, no fear of errors
- **Clear Feedback**: Detailed update process output

### ✅ Developer-Friendly Experience
- **Simple to Use**: Just inherit from mixin class
- **Consistent Interface**: All tests use the same pattern
- **Flexible Extension**: Easy to add new test types
- **Debug-Friendly**: Clear log output

### ✅ Real-World Use Cases
```bash
# Scenario 1: Algorithm improvement, need to update all benchmark tests
UPDATE_REFERENCES=true python -m pytest tests/benchmarks/ -v -s

# Scenario 2: Update only specific atom tests
UPDATE_REFERENCES=true python -m pytest tests/He/ tests/Li/ -v -s

# Scenario 3: Frequent validation during development
python -m pytest tests/He/test_he.py::HeAtomTest::test_checkpoint_comparison_with_reference -v -s

# Scenario 4: Reference file validation in CI/CD
# Never set UPDATE_REFERENCES in CI, ensure only validation
python -m pytest tests/ -v
```

## Technical Implementation Details

### Environment Variable Detection
```python
UPDATE_REFERENCES = os.environ.get('UPDATE_REFERENCES', 'false').lower() in ('true', '1', 'yes')
```

### File Registration Mechanism
```python
def _register_generated_file(self, ref_name: str, generated_path: str):
    """Register generated file for potential reference file update"""
    self.generated_files[ref_name] = generated_path
```

### Delayed Update Strategy
- Execute updates in `tearDown()` to ensure complete test execution
- Avoid incomplete updates due to test failures

### Smart Path Resolution
- Automatically detect test file location
- Flexible handling of relative and absolute paths

This system is particularly suitable for the scenario you mentioned: **having many similar tests that need to be controlled by a flag to either validate or update reference files**. Now you only need one environment variable to control all test behavior!
