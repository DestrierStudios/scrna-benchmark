#!/usr/bin/env python
"""Verify all dependencies are installed correctly."""

import sys
import warnings
warnings.filterwarnings("ignore")

def check_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'OK')
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name}: {e}")
        return False

def check_r_packages():
    try:
        from rpy2.robjects.packages import importr
        import rpy2.robjects as ro
        
        # Suppress R warnings
        ro.r('options(warn=-1)')
        
        r_packages = ['scran', 'scater', 'SingleCellExperiment', 'Seurat', 'sctransform', 'Dino']
        all_ok = True
        for pkg in r_packages:
            try:
                importr(pkg)
                print(f"[OK] R::{pkg}")
            except Exception as e:
                print(f"[FAIL] R::{pkg}: {e}")
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"[FAIL] rpy2: {e}")
        return False

def main():
    print("=" * 50)
    print("Environment Verification")
    print("=" * 50)
    
    print("\n[Python Packages]")
    python_ok = all([
        check_import('scanpy'),
        check_import('anndata'),
        check_import('numpy'),
        check_import('pandas'),
        check_import('scipy'),
        check_import('sklearn', 'scikit-learn'),
        check_import('matplotlib'),
        check_import('seaborn'),
        check_import('h5py'),
        check_import('leidenalg'),
        check_import('louvain'),
        check_import('snakemake'),
        check_import('geosketch'),
        check_import('rpy2'),
    ])
    
    print("\n[R Packages]")
    r_ok = check_r_packages()
    
    print("\n" + "=" * 50)
    if python_ok and r_ok:
        print("[SUCCESS] All checks passed!")
    else:
        print("[ERROR] Some checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()