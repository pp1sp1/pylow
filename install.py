import os
import sys
import subprocess
import platform
import shutil
import urllib.request
import zipfile
import json
import tempfile

# --- CONFIGURATION ---
REPO_OWNER = "pp1sp1"
REPO_NAME = "pylow"
COMMAND_NAME = "pylow"
ENTRY_POINT = "pylow.py"

if platform.system() == "Windows":
    INSTALL_DIR = os.path.join(os.environ["USERPROFILE"], ".pylow")
    VENV_PYTHON = os.path.join(INSTALL_DIR, "venv", "Scripts", "python.exe")
else:
    INSTALL_DIR = os.path.expanduser("~/.pylow")
    VENV_PYTHON = os.path.join(INSTALL_DIR, "venv", "bin", "python")

SYSTEM_DEPENDENCIES = {
    "Linux": ["build-essential", "clang", "libllvm19", "python3-venv"],
    "Darwin": ["llvm"],
    "Windows": ["LLVM.LLVM"]
}
PIP_DEPENDENCIES = ["lief", "llvmlite"]

def print_status(message, status="INFO"):
    colors = {"INFO": "🔵", "SUCCESS": "🟢", "WARN": "🟡", "ERROR": "🔴", "OK": "✅"}
    print(f"{colors.get(status, '⚪')} [{status}] {message}")

def ask_confirmation(question):
    return input(f"{question} (y/n): ").lower() == 'y'

def run_cmd(cmd, sudo=False):
    if sudo and platform.system() != "Windows":
        cmd = ["sudo"] + cmd
    subprocess.run(cmd, check=True)

def install_project():
    print_status("Checking for the latest version on GitHub...", "INFO")
    tag_name, zip_url = None, None
    
    try:
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            tag_name = data['tag_name']
            zip_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/archive/refs/tags/{tag_name}.zip"
    except Exception:
        print_status("Latest Release endpoint not found. Checking tags fallback...", "WARN")
        
    if not zip_url:
        try:
            tags_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/tags"
            req = urllib.request.Request(tags_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                tags_data = json.loads(response.read().decode())
                if tags_data:
                    tag_name = tags_data[0]['name']
                    zip_url = tags_data[0]['zipball_url']
                else: raise Exception("No tags found.")
        except Exception as e:
            print_status(f"Fetch error: {e}", "ERROR")
            sys.exit(1)
            
    print_status(f"Found version: {tag_name}. Downloading...", "INFO")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "pylow.zip")
            req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
                
            if os.path.exists(INSTALL_DIR):
                # Keep venv if exists to speed up, or wipe everything
                shutil.rmtree(INSTALL_DIR)
            os.makedirs(INSTALL_DIR, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                top_folder = zip_ref.namelist()[0].split('/')[0]
                zip_ref.extractall(temp_dir)
                actual_content_path = os.path.join(temp_dir, top_folder)
                for item in os.listdir(actual_content_path):
                    shutil.move(os.path.join(actual_content_path, item), INSTALL_DIR)
                    
        print_status(f"Source installed to {INSTALL_DIR}", "SUCCESS")
    except Exception as e:
        print_status(f"Installation failed: {e}", "ERROR"); sys.exit(1)

def setup_venv_and_deps():
    print_status("Setting up virtual environment (venv)...", "INFO")
    venv_dir = os.path.join(INSTALL_DIR, "venv")
    
    try:
        # Create venv
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        
        # Install pip deps inside venv
        print_status("Installing Python packages inside venv...", "INFO")
        subprocess.run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([VENV_PYTHON, "-m", "pip", "install"] + PIP_DEPENDENCIES, check=True)
        print_status("Python dependencies installed successfully.", "SUCCESS")
    except Exception as e:
        print_status(f"Venv setup failed: {e}", "ERROR"); sys.exit(1)

def install_system_dependencies():
    system = platform.system()
    for dep in SYSTEM_DEPENDENCIES.get(system, []):
        if shutil.which(dep) is None:
            if ask_confirmation(f"Missing system dependency: '{dep}'. Install it now?"):
                try:
                    if system == "Linux": run_cmd(["apt-get", "install", "-y", dep], sudo=True)
                    elif system == "Darwin": run_cmd(["brew", "install", dep])
                    elif system == "Windows": run_cmd(["winget", "install", "-e", "--id", dep])
                except Exception as e: 
                    print_status(f"Failed to install {dep}: {e}", "ERROR")

def setup_global_command():
    system = platform.system()
    script_full_path = os.path.join(INSTALL_DIR, ENTRY_POINT)
    
    try:
        if system in ["Linux", "Darwin"]:
            bin_path = f"/usr/local/bin/{COMMAND_NAME}"
            # Tworzymy wrapper, który odpala skrypt używając venv
            wrapper_content = f'#!/bin/bash\n"{VENV_PYTHON}" "{script_full_path}" "$@"\n'
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
                tf.write(wrapper_content)
                temp_wrapper = tf.name
            
            run_cmd(["mv", temp_wrapper, bin_path], sudo=True)
            run_cmd(["chmod", "+x", bin_path], sudo=True)
            
        elif system == "Windows":
            bat_path = os.path.join(INSTALL_DIR, f"{COMMAND_NAME}.bat")
            with open(bat_path, "w") as bat:
                bat.write(f"@echo off\n\"{VENV_PYTHON}\" \"{script_full_path}\" %*")
            
            # Add INSTALL_DIR to PATH if not there
            if INSTALL_DIR not in os.environ.get("PATH", ""):
                subprocess.run(f'setx PATH "%PATH%;{INSTALL_DIR}"', shell=True)
                print_status("Restart your terminal to use 'pylow' command.", "WARN")
                
        print_status(f"Global command '{COMMAND_NAME}' is ready!", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to set global command: {e}", "ERROR")

def main():
    print("\n🚀 Welcome to the PyLow Installer (Venv Mode)\n" + "="*40)
    if not ask_confirmation("Proceed with installation?"): sys.exit()
        
    install_system_dependencies()
    install_project()
    setup_venv_and_deps()
    
    if ask_confirmation("Do you want to configure the global 'pylow' command?"):
        setup_global_command()
        
    print("\n✅ Setup Complete! Type 'pylow' to start.")

if __name__ == "__main__":
    main()
