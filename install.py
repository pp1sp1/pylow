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
else:
    INSTALL_DIR = os.path.expanduser("~/.pylow")

SYSTEM_DEPENDENCIES = {
    "Linux": [
        "build-essential",  # Zawiera gcc i g++
        "clang",            # Zawiera clang i clang++
        "libllvm19"         # Konkretna wersja biblioteki LLVM
    ],
    "Darwin": ["llvm"],
    "Windows": ["LLVM.LLVM"]
}
PIP_DEPENDENCIES = ["lief","llvmlite"]

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
    """Downloads the latest code from GitHub Releases or Tags fallback."""
    print_status("Checking for the latest version on GitHub...", "INFO")
    
    tag_name = None
    zip_url = None
    
    # 1. Try to fetch from the latest Release API endpoint
    try:
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            tag_name = data['tag_name']
            zip_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/archive/refs/tags/{tag_name}.zip"
    except Exception:
        print_status("Latest Release endpoint not found. Checking tags fallback...", "WARN")
        
    # 2. Fallback to Tags API if no Release is found
    if not zip_url:
        try:
            tags_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/tags"
            req = urllib.request.Request(tags_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                tags_data = json.loads(response.read().decode())
                if tags_data:
                    tag_name = tags_data[0]['name']
                    zip_url = tags_data[0]['zipball_url']
                else:
                    raise Exception("No tags found in the repository.")
        except Exception as e:
            print_status(f"Authentication or Fetch error: {e}", "ERROR")
            print_status("Make sure you pushed your tag using: git push origin --tags", "WARN")
            sys.exit(1)
            
    print_status(f"Found version/tag: {tag_name}. Downloading source...", "INFO")
    
    # 3. Secure download and extraction using system temp directory
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "pylow.zip")
            
            req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
                
            if os.path.exists(INSTALL_DIR):
                shutil.rmtree(INSTALL_DIR)
            os.makedirs(INSTALL_DIR, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # GitHub archives root directory name format: repo_owner-repo_name-commit_hash
                top_folder = zip_ref.namelist()[0].split('/')[0]
                zip_ref.extractall(temp_dir)
                
                actual_content_path = os.path.join(temp_dir, top_folder)
                for item in os.listdir(actual_content_path):
                    shutil.move(os.path.join(actual_content_path, item), INSTALL_DIR)
                    
        print_status(f"Version {tag_name} successfully installed to {INSTALL_DIR}", "SUCCESS")
        
    except Exception as e:
        print_status(f"Extraction and installation failed: {e}", "ERROR")
        sys.exit(1)

def install_dependencies():
    system = platform.system()
    for dep in SYSTEM_DEPENDENCIES.get(system, []):
        if shutil.which(dep) is None:
            if ask_confirmation(f"Missing system dependency: '{dep}'. Install it now?"):
                try:
                    if system == "Linux": run_cmd(["apt-get", "install", "-y", dep], sudo=True)
                    elif system == "Darwin": run_cmd(["brew", "install", dep])
                    elif system == "Windows": run_cmd(["winget", "install", "-e", "--id", dep])
                except Exception as e: 
                    print_status(f"Failed to install system package {dep}: {e}", "ERROR")
                    
    if PIP_DEPENDENCIES:
        print_status("Installing required Python packages...", "INFO")
        try:
            run_cmd([sys.executable, "-m", "pip", "install"] + PIP_DEPENDENCIES)
        except Exception as e:
            print_status(f"Pip installation failed: {e}", "ERROR")

def setup_global_command():
    system = platform.system()
    script_full_path = os.path.join(INSTALL_DIR, ENTRY_POINT)
    
    if not os.path.exists(script_full_path):
        print_status(f"Entrypoint script '{ENTRY_POINT}' not found in source directory.", "ERROR")
        return

    try:
        if system in ["Linux", "Darwin"]:
            bin_path = f"/usr/local/bin/{COMMAND_NAME}"
            
            # Prepend Shebang if not present
            with open(script_full_path, 'r+') as f:
                content = f.read()
                if not content.startswith("#!"):
                    f.seek(0, 0)
                    f.write("#!/usr/bin/env python3\n" + content)
                    
            run_cmd(["ln", "-sf", script_full_path, bin_path], sudo=True)
            run_cmd(["chmod", "+x", script_full_path], sudo=True)
            
        elif system == "Windows":
            bat_path = os.path.join(INSTALL_DIR, f"{COMMAND_NAME}.bat")
            with open(bat_path, "w") as bat:
                bat.write(f"@echo off\npython \"{script_full_path}\" %*")
            old_path = os.environ.get("PATH", "")
            if INSTALL_DIR not in old_path:
                subprocess.run(f'setx PATH "%PATH%;{INSTALL_DIR}"', shell=True)
                print_status("Added to PATH environment variable. Restart your terminal session!", "WARN")
                
        print_status(f"Global executable command '{COMMAND_NAME}' is now configured!", "SUCCESS")
    except Exception as e:
        print_status(f"Configuration failed: {e}", "ERROR")

def main():
    print("\n🚀 Welcome to the PyLow Installer\n" + "="*35)
    if not ask_confirmation("Do you want to install PyLow from the latest remote source?"):
        print_status("Installation aborted by user.", "WARN")
        sys.exit()
        
    install_project()
    install_dependencies()
    
    if ask_confirmation("Do you want to configure the global 'pylow' shell command?"):
        setup_global_command()
        
    print("\n✅ Setup Complete! Type 'pylow' in your terminal to begin.")

if __name__ == "__main__":
    main()
