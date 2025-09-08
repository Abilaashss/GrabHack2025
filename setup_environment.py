#!/usr/bin/env python3
"""
Project Nova - Environment Setup Script
Creates virtual environment and installs all requirements
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class EnvironmentSetup:
    """Setup virtual environment and dependencies for Project Nova"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_cmd = self._get_python_command()
        self.venv_name = "nova_env"
        self.venv_path = Path(self.venv_name)
        
    def _get_python_command(self):
        """Get the appropriate Python command for the system"""
        # Try different Python commands
        python_commands = ['python3', 'python', 'py']
        
        for cmd in python_commands:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, check=True)
                if 'Python 3.' in result.stdout:
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        raise RuntimeError("Python 3 not found. Please install Python 3.8 or higher.")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🔍 Checking Python version...")
        
        try:
            result = subprocess.run([self.python_cmd, '--version'], 
                                  capture_output=True, text=True, check=True)
            version_str = result.stdout.strip()
            print(f"   Found: {version_str}")
            
            # Extract version numbers
            version_parts = version_str.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major < 3 or (major == 3 and minor < 8):
                raise RuntimeError(f"Python 3.8+ required, found {version_str}")
            
            print("   ✅ Python version is compatible")
            return True
            
        except subprocess.CalledProcessError:
            raise RuntimeError("Failed to check Python version")
    
    def create_virtual_environment(self):
        """Create virtual environment"""
        print(f"\n🏗️  Creating virtual environment '{self.venv_name}'...")
        
        if self.venv_path.exists():
            print(f"   ⚠️  Virtual environment '{self.venv_name}' already exists")
            response = input("   Do you want to recreate it? (y/n): ").lower().strip()
            
            if response == 'y':
                print("   🗑️  Removing existing virtual environment...")
                if self.system == 'windows':
                    subprocess.run(['rmdir', '/s', '/q', str(self.venv_path)], shell=True)
                else:
                    subprocess.run(['rm', '-rf', str(self.venv_path)])
            else:
                print("   📁 Using existing virtual environment")
                return True
        
        try:
            # Create virtual environment
            subprocess.run([self.python_cmd, '-m', 'venv', self.venv_name], check=True)
            print(f"   ✅ Virtual environment '{self.venv_name}' created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to create virtual environment: {e}")
            return False
    
    def get_activation_command(self):
        """Get the command to activate virtual environment"""
        if self.system == 'windows':
            return str(self.venv_path / 'Scripts' / 'activate.bat')
        else:
            return f"source {self.venv_path}/bin/activate"
    
    def get_pip_command(self):
        """Get the pip command for the virtual environment"""
        if self.system == 'windows':
            return str(self.venv_path / 'Scripts' / 'pip.exe')
        else:
            return str(self.venv_path / 'bin' / 'pip')
    
    def get_python_venv_command(self):
        """Get the Python command for the virtual environment"""
        if self.system == 'windows':
            return str(self.venv_path / 'Scripts' / 'python.exe')
        else:
            return str(self.venv_path / 'bin' / 'python')
    
    def upgrade_pip(self):
        """Upgrade pip in virtual environment"""
        print("\n📦 Upgrading pip...")
        
        try:
            pip_cmd = self.get_pip_command()
            subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
            print("   ✅ Pip upgraded successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Failed to upgrade pip: {e}")
            return False
    
    def install_requirements(self):
        """Install requirements with fallback options"""
        print("\n📚 Installing requirements...")
        
        # Check which requirements files exist
        req_files = []
        if Path('requirements-minimal.txt').exists():
            req_files.append(('requirements-minimal.txt', 'Minimal (faster, core packages only)'))
        if Path('requirements.txt').exists():
            req_files.append(('requirements.txt', 'Full (all packages, may take longer)'))
        
        if not req_files:
            print("   ❌ No requirements files found")
            return False
        
        # Let user choose if multiple options available
        if len(req_files) > 1:
            print("   Available installation options:")
            for i, (file, desc) in enumerate(req_files, 1):
                print(f"   {i}. {file} - {desc}")
            
            while True:
                try:
                    choice = input("   Choose installation type (1 for minimal, 2 for full): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(req_files):
                        req_file = req_files[choice_idx][0]
                        break
                    else:
                        print("   Invalid choice. Please enter 1 or 2.")
                except ValueError:
                    print("   Invalid input. Please enter a number.")
        else:
            req_file = req_files[0][0]
        
        print(f"   Installing from {req_file}...")
        
        try:
            pip_cmd = self.get_pip_command()
            
            # Install requirements with progress
            process = subprocess.Popen(
                [pip_cmd, 'install', '-r', req_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print("   Installing packages...")
            for line in process.stdout:
                if any(keyword in line for keyword in ['Collecting', 'Installing', 'Successfully installed', 'Requirement already satisfied']):
                    print(f"   {line.strip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print("   ✅ Requirements installed successfully")
                
                # If minimal was installed, offer to install additional packages
                if req_file == 'requirements-minimal.txt' and Path('requirements.txt').exists():
                    print("\n   💡 You installed minimal requirements.")
                    install_full = input("   Install additional packages for full functionality? (y/n): ").lower().strip()
                    if install_full == 'y':
                        return self._install_additional_packages()
                
                return True
            else:
                print("   ❌ Failed to install some requirements")
                
                # Try minimal if full failed
                if req_file == 'requirements.txt' and Path('requirements-minimal.txt').exists():
                    print("   🔄 Trying minimal installation as fallback...")
                    return self._install_minimal_fallback()
                
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install requirements: {e}")
            return False
    
    def _install_minimal_fallback(self):
        """Install minimal requirements as fallback"""
        try:
            pip_cmd = self.get_pip_command()
            result = subprocess.run([pip_cmd, 'install', '-r', 'requirements-minimal.txt'], 
                                  capture_output=True, text=True, check=True)
            print("   ✅ Minimal requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("   ❌ Even minimal installation failed")
            return False
    
    def _install_additional_packages(self):
        """Install additional packages beyond minimal"""
        additional_packages = [
            'catboost>=1.1.0',
            'tensorflow>=2.10.0',
            'plotly>=5.10.0',
            'shap>=0.41.0',
            'fairlearn>=0.8.0'
        ]
        
        pip_cmd = self.get_pip_command()
        
        for package in additional_packages:
            try:
                print(f"   Installing {package.split('>=')[0]}...")
                subprocess.run([pip_cmd, 'install', package], 
                             capture_output=True, text=True, check=True)
                print(f"   ✅ {package.split('>=')[0]} installed")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  Failed to install {package.split('>=')[0]} (skipping)")
        
        return True
    
    def verify_installation(self):
        """Verify that key packages are installed"""
        print("\n🔍 Verifying installation...")
        
        # Core packages (must have)
        core_packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'matplotlib']
        
        # Optional packages (nice to have)
        optional_packages = ['catboost', 'tensorflow', 'seaborn', 'plotly', 'shap', 'fairlearn']
        
        python_cmd = self.get_python_venv_command()
        failed_core = []
        failed_optional = []
        
        # Check core packages
        print("   Core packages:")
        for package in core_packages:
            try:
                # Handle sklearn import name
                import_name = 'sklearn' if package == 'scikit-learn' else package
                result = subprocess.run(
                    [python_cmd, '-c', f'import {import_name}; print(f"{package} OK")'],
                    capture_output=True, text=True, check=True
                )
                print(f"   ✅ {package}")
            except subprocess.CalledProcessError:
                print(f"   ❌ {package}")
                failed_core.append(package)
        
        # Check optional packages
        print("   Optional packages:")
        for package in optional_packages:
            try:
                result = subprocess.run(
                    [python_cmd, '-c', f'import {package}; print(f"{package} OK")'],
                    capture_output=True, text=True, check=True
                )
                print(f"   ✅ {package}")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {package} (optional)")
                failed_optional.append(package)
        
        if failed_core:
            print(f"\n   ❌ Critical packages failed: {', '.join(failed_core)}")
            print("   The pipeline may not work properly.")
            return False
        else:
            print(f"\n   ✅ All core packages verified successfully")
            if failed_optional:
                print(f"   ⚠️  Optional packages not available: {', '.join(failed_optional)}")
                print("   Some advanced features may be limited.")
            return True
    
    def create_activation_scripts(self):
        """Create convenient activation scripts"""
        print("\n📝 Creating activation scripts...")
        
        # Create activation script for different platforms
        if self.system == 'windows':
            # Windows batch file
            batch_content = f"""@echo off
echo Activating Project Nova environment...
call {self.venv_name}\\Scripts\\activate.bat
echo.
echo ✅ Project Nova environment activated!
echo.
echo Available commands:
echo   python quick_start.py          - Quick pipeline test
echo   python main_training_pipeline.py - Full pipeline with tuning
echo   python deploy_model.py         - Deploy trained models
echo   python project_summary.py      - Show project overview
echo.
cmd /k
"""
            with open('activate_nova.bat', 'w') as f:
                f.write(batch_content)
            print("   ✅ Created activate_nova.bat")
            
        else:
            # Unix shell script
            shell_content = f"""#!/bin/bash
echo "Activating Project Nova environment..."
source {self.venv_name}/bin/activate

echo ""
echo "✅ Project Nova environment activated!"
echo ""
echo "Available commands:"
echo "  python quick_start.py          - Quick pipeline test"
echo "  python main_training_pipeline.py - Full pipeline with tuning"
echo "  python deploy_model.py         - Deploy trained models"
echo "  python project_summary.py      - Show project overview"
echo ""

# Keep shell open
exec "$SHELL"
"""
            with open('activate_nova.sh', 'w') as f:
                f.write(shell_content)
            
            # Make executable
            os.chmod('activate_nova.sh', 0o755)
            print("   ✅ Created activate_nova.sh")
    
    def display_next_steps(self):
        """Display next steps for the user"""
        print("\n" + "="*60)
        print("🎉 PROJECT NOVA ENVIRONMENT SETUP COMPLETE!")
        print("="*60)
        
        activation_cmd = self.get_activation_command()
        
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Activate the virtual environment:")
        
        if self.system == 'windows':
            print(f"   {activation_cmd}")
            print("   OR double-click: activate_nova.bat")
        else:
            print(f"   {activation_cmd}")
            print("   OR run: ./activate_nova.sh")
        
        print(f"\n2. Test the setup:")
        print(f"   python quick_start.py")
        
        print(f"\n3. Run full pipeline:")
        print(f"   python main_training_pipeline.py")
        
        print(f"\n4. Deploy models:")
        print(f"   python deploy_model.py")
        
        print(f"\n📁 Virtual Environment Location: {self.venv_path.absolute()}")
        print(f"🐍 Python Command: {self.get_python_venv_command()}")
        print(f"📦 Pip Command: {self.get_pip_command()}")
        
        print(f"\n💡 TIP: Always activate the virtual environment before running scripts!")
        print("="*60)
    
    def run_setup(self):
        """Run the complete setup process"""
        print("="*60)
        print("PROJECT NOVA - ENVIRONMENT SETUP")
        print("="*60)
        
        try:
            # Step 1: Check Python version
            self.check_python_version()
            
            # Step 2: Create virtual environment
            if not self.create_virtual_environment():
                return False
            
            # Step 3: Upgrade pip
            self.upgrade_pip()
            
            # Step 4: Install requirements
            if not self.install_requirements():
                return False
            
            # Step 5: Verify installation
            if not self.verify_installation():
                print("\n⚠️  Some packages failed to install, but continuing...")
            
            # Step 6: Create activation scripts
            self.create_activation_scripts()
            
            # Step 7: Display next steps
            self.display_next_steps()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False

def main():
    """Main function"""
    
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  You're already in a virtual environment.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting...")
            return
    
    # Run setup
    setup = EnvironmentSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now start using Project Nova!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()