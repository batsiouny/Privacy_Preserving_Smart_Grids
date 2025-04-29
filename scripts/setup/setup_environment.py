#!/usr/bin/env python
import os
import sys
import subprocess

def setup_environment():
    print('Setting up environment for Privacy-Preserving Smart Grid project...')
    
    # Install Python dependencies
    print('Installing Python dependencies...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    print('Environment setup complete!')
    
if __name__ == '__main__':
    setup_environment()
