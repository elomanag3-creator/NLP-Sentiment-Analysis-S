#!/usr/bin/env python
"""
Setup and Installation Script
One-command setup for the entire project
"""

import os
import subprocess
import sys
import platform
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def print_step(text):
    """Print step message"""
    print(f"\n▶ {text}")


def run_command(command, description=""):
    """Run shell command"""
    if description:
        print_step(description)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✓ Success")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_python_version():
    """Check Python version"""
    print_step("Checking Python version")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python 3.7+ required (found {version.major}.{version.minor})")
        return False


def create_directories():
    """Create project directories"""
    print_step("Creating project directories")
    
    directories = [
        'data',
        'models',
        'results',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")


def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing Python dependencies")
    
    # Check if requirements.txt exists
    if not Path('requirements.txt').exists():
        print("  ✗ requirements.txt not found")
        return False
    
    command = f"{sys.executable} -m pip install -q -r requirements.txt"
    return run_command(command)


def download_nltk_data():
    """Download required NLTK data"""
    print_step("Downloading NLTK data")
    
    python_code = """
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print('✓ NLTK data downloaded')
"""
    
    command = f"{sys.executable} -c \"{python_code}\""
    return run_command(command)


def verify_installation():
    """Verify all packages are installed"""
    print_step("Verifying installation")
    
    packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'nltk',
        'matplotlib',
        'seaborn'
    ]
    
    all_installed = True
    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def run_tests():
    """Run unit tests"""
    print_step("Running unit tests")
    
    command = f"{sys.executable} src/tests.py"
    return run_command(command)


def setup_complete_message():
    """Print setup complete message"""
    print_header("✓ SETUP COMPLETE!")
    
    print("""
Next Steps:

1. Run the complete pipeline:
   python main.py

2. Run individual components:
   - Preprocessing:  python src/preprocess.py
   - Training:       python src/train.py
   - Prediction:     python src/predict.py
   - Analysis:       (in main.py)

3. Start the API server:
   python api.py
   Then: curl -X POST http://localhost:5000/api/v1/predict \\
         -H "Content-Type: application/json" \\
         -d '{"review": "Great product!"}'

4. Explore the code:
   - src/preprocess.py  - Text preprocessing
   - src/train.py       - Model training
   - src/predict.py     - Predictions
   - src/analysis.py    - Visualization
   - src/config.py      - Configuration

5. Review documentation:
   - README.md          - Project overview
   - CV_TEMPLATE.txt    - CV description

Project Structure:
├── data/              # Data files & preprocessor
├── models/            # Trained model
├── results/           # Output visualizations
├── src/               # Source code
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── analysis.py
│   ├── config.py
│   └── tests.py
├── main.py            # Main pipeline
├── api.py             # Flask API
├── requirements.txt   # Dependencies
└── README.md          # Documentation

Troubleshooting:

If you get import errors:
  pip install -r requirements.txt

If NLTK data is missing:
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

If tests fail:
  Check that all dependencies are installed
  Verify Python version (3.7+)

Happy Learning! 🎉
""")


def main():
    """Run complete setup"""
    print_header("NLP SENTIMENT ANALYSIS PROJECT SETUP")
    
    print(f"\nPlatform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠ Warning: Some dependencies may not have installed properly")
    
    # Download NLTK data
    download_nltk_data()
    
    # Verify installation
    if not verify_installation():
        print("\n⚠ Warning: Some packages are missing")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run tests
    print("\nWould you like to run tests? (y/n): ", end="")
    response = input().lower()
    if response == 'y':
        run_tests()
    
    # Print completion message
    setup_complete_message()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)
