import os
import subprocess

def setup_project():
    """Create necessary directories and install requirements"""
    
    # Create directories
    directories = ['model', 'frontend']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Install requirements
    print("Installing requirements...")
    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
    
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your monthly-car-sales.csv file in the root directory")
    print("2. Run: python train_model.py")
    print("3. Run: python app.py")
    print("4. Visit: http://localhost:8000")

if __name__ == "__main__":
    setup_project()
