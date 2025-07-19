"""Base class for FermiNet tests with reference file management."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, List


# Environment variable to control reference file generation
UPDATE_REFERENCES = os.environ.get('UPDATE_REFERENCES', 'false').lower() in ('true', '1', 'yes')


class ReferenceTestMixin:
    """Mixin class for tests that need reference file management.
    
    This mixin provides functionality to:
    1. Register generated files for potential reference updating
    2. Automatically update reference files when UPDATE_REFERENCES=true
    3. Create backups of existing reference files
    4. Skip comparison when in update mode
    
    Usage:
        class MyTest(unittest.TestCase, ReferenceTestMixin):
            def setUp(self):
                super().setUp()
                self.init_reference_management()
    
            def tearDown(self):
                self.cleanup_reference_management()
                super().tearDown()
    
            def test_something(self):
                # Register files that might become new references
                self._register_generated_file("ref_output.txt", "/path/to/generated.txt")
                
                # Run your test...
                
                # Skip comparison if in update mode
                if UPDATE_REFERENCES:
                    return
                
                # Normal comparison logic...
    """
    
    def init_reference_management(self):
        """Initialize reference file management. Call this in setUp()."""
        self.generated_files: Dict[str, str] = {}
    
    def cleanup_reference_management(self):
        """Clean up and update references if needed. Call this in tearDown()."""
        if UPDATE_REFERENCES and hasattr(self, 'generated_files') and self.generated_files:
            self._update_reference_files()
    
    def _register_generated_file(self, ref_name: str, generated_path: str):
        """Register a generated file for potential reference updating.
        
        Args:
            ref_name: Name of the reference file (e.g., "ref_output.npz")
            generated_path: Path to the generated file that might become the new reference
        """
        if not hasattr(self, 'generated_files'):
            self.generated_files = {}
        self.generated_files[ref_name] = generated_path
    
    def _update_reference_files(self):
        """Update reference files with generated files."""
        # Try to get the test directory path
        if hasattr(self, '__class__') and hasattr(self.__class__, '__module__'):
            import inspect
            test_file = inspect.getfile(self.__class__)
            test_dir_path = Path(test_file).parent
        else:
            test_dir_path = Path('.')
        
        print(f"\n UPDATE_REFERENCES=true detected, updating reference files...")
        
        for ref_name, generated_path in self.generated_files.items():
            ref_path = test_dir_path / ref_name
            
            if os.path.exists(generated_path):
                # # Create backup of existing reference
                # if ref_path.exists():
                #     backup_path = ref_path.with_suffix(ref_path.suffix + '.backup')
                #     shutil.copy2(ref_path, backup_path)
                #     print(f" Backed up: {ref_name} → {backup_path.name}")
                
                # Copy new reference
                shutil.copy2(generated_path, ref_path)
                print(f" Updated reference: {ref_name}")
                
                # Show file info
                size = os.path.getsize(ref_path)
                print(f" Size: {size:,} bytes")
            else:
                print(f"  Generated file not found: {generated_path}")
        
        print(" Reference files update completed!")

    def compare_with_references(self, comparisons: List[Dict[str, str]], tolerance: str = "1e-6", 
                               atol: Optional[str] = None, rtol: Optional[str] = None,
                               skip_fields: Optional[List[str]] = None) -> bool:
        """Compare generated files with reference files using the comparison utility.
        
        Args:
            comparisons: List of comparison dictionaries with keys:
                        - 'generated': path to generated file
                        - 'reference': path to reference file  
                        - 'type': file type ('npz', 'csv', 'txt', 'npy'), optional, auto-detected from extension
            tolerance: Numerical tolerance for comparison (sets both atol and rtol if atol/rtol not specified)
            atol: Absolute tolerance (overrides tolerance if specified)
            rtol: Relative tolerance (overrides tolerance if specified)
            skip_fields: Fields to skip in comparison (e.g., ['opt_state'])
            
        Returns:
            True if all comparisons pass, False otherwise
            
        Raises:
            AssertionError: If any comparison fails (when not in update mode)
        """
        if should_skip_comparison():
            print("🔄 UPDATE_REFERENCES=true, skipping comparison (will update references)")
            return True
            
        # Get the comparison script path
        if hasattr(self, '__class__') and hasattr(self.__class__, '__module__'):
            import inspect
            test_file = inspect.getfile(self.__class__)
            test_dir_path = Path(test_file).parent
        else:
            test_dir_path = Path('.')
        
        compare_script = test_dir_path / "../tools/compare_files.py"
        if not compare_script.exists():
            # Try alternative path
            compare_script = test_dir_path / "tools/compare_files.py"
        
        all_passed = True
        
        for i, comp in enumerate(comparisons):
            generated_file = comp['generated']
            reference_file = comp['reference']
            file_type = comp.get('type', None)
            
            # Auto-detect file type if not specified
            if file_type is None:
                if generated_file.endswith('.csv'):
                    file_type = 'csv'
                elif generated_file.endswith('.npz'):
                    file_type = 'npz'
                elif generated_file.endswith('.txt'):
                    file_type = 'txt'
                elif generated_file.endswith('.npy'):
                    file_type = 'npy'
                else:
                    file_type = 'npz'  # default
            
            # Build command
            cmd = [
                "python", str(compare_script),
                str(generated_file), str(reference_file)
            ]
            
            # Add tolerance parameters
            cmd.extend(["--tolerance", tolerance])
            if atol is not None:
                cmd.extend(["--atol", atol])
            if rtol is not None:
                cmd.extend(["--rtol", rtol])
                
            
            # Add type specification for non-default types
            if file_type in ['csv', 'txt', 'npy', 'npz']:
                cmd.extend(["--type", file_type])
            
            # Add skip fields (only for NPZ files)
            if skip_fields and file_type in ['npz']:
                cmd.extend(["--skip"] + skip_fields)
            
            # Run comparison
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                file_type_name = {
                    'csv': 'CSV',
                    'txt': 'TXT', 
                    'npy': 'NPY',
                    'npz': 'NPZ'
                }.get(file_type, 'File')
                print(f"\n❌ {file_type_name} comparison failed (comparison {i+1}):")
                print(f"Generated: {generated_file}")
                print(f"Reference: {reference_file}")
                print("Output:", result.stdout)
                if result.stderr:
                    print("Error:", result.stderr)
                all_passed = False
            else:
                file_type_name = {
                    'csv': 'CSV',
                    'txt': 'TXT',
                    'npy': 'NPY', 
                    'npz': 'NPZ'
                }.get(file_type, 'File')
                print(f"✅ {file_type_name} comparison passed (comparison {i+1})")
        
        if all_passed:
            print("🎯 All comparisons passed within tolerance!")
        else:
            # Raise assertion error for test framework
            raise AssertionError("One or more file comparisons failed within tolerance")
            
        return all_passed


def should_update_references() -> bool:
    """Check if references should be updated."""
    return UPDATE_REFERENCES


def should_skip_comparison() -> bool:
    """Check if comparison should be skipped (when updating references)."""
    return UPDATE_REFERENCES
