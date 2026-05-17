import os
import sys
import subprocess
import platform
import shutil
import urllib.request
import zipfile
import json

# --- KONFIGURACJA ---
REPO_OWNER = "pp1sp1"
REPO_NAME = "pylow"
COMMAND_NAME = "pylow"
ENTRY_POINT = "pylow.py"

if platform.system() == "Windows":
    INSTALL_DIR = os.path.join(os.environ["USERPROFILE"], ".pylow")
else:
    INSTALL_DIR = os.path.expanduser("~/.pylow")

SYSTEM_DEPENDENCIES = {
    "Linux": ["build-essential", "cmake", "git"],
    "Darwin": ["cmake", "git"],
    "Windows": ["git", "cmake"]
}
PIP_DEPENDENCIES = ["colorama"]

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
    """Pobiera najnowszą wersję z GitHub Releases."""
    print_status("Sprawdzanie najnowszej wersji w GitHub Releases...", "INFO")
    
    try:
        # 1. Pobranie informacji o najnowszym release z API GitHub
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            tag_name = data['tag_name']
            # Link do automatycznego zipa z kodem źródłowym konkretnego taga
            zip_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/archive/refs/tags/{tag_name}.zip"
        
        print_status(f"Znaleziono wersję {tag_name}. Pobieranie...", "INFO")
        
        # 2. Pobieranie pliku ZIP
        zip_path = os.path.join(os.path.dirname(INSTALL_DIR), "pylow_temp.zip")
        urllib.request.urlretrieve(zip_url, zip_path)
        
        # 3. Rozpakowywanie
        if os.path.exists(INSTALL_DIR):
            shutil.rmtree(INSTALL_DIR)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # GitHub pakuje pliki w folder 'pylow-tagname', musimy to wyciągnąć
            top_folder = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall(INSTALL_DIR)
            
            # Przenosimy zawartość z folderu 'pylow-xxx' bezpośrednio do INSTALL_DIR
            actual_content_path = os.path.join(INSTALL_DIR, top_folder)
            for item in os.listdir(actual_content_path):
                shutil.move(os.path.join(actual_content_path, item), INSTALL_DIR)
            os.rmdir(actual_content_path)
            
        os.remove(zip_path)
        print_status(f"Wersja {tag_name} została zainstalowana w {INSTALL_DIR}", "SUCCESS")
        
    except Exception as e:
        print_status(f"Błąd podczas pobierania z Releases: {e}", "ERROR")
        print_status("Upewnij się, że stworzyłeś 'Release' w swoim repozytorium na GitHubie!", "WARN")
        sys.exit(1)

# --- Reszta funkcji (install_dependencies, setup_global_command, main) zostaje bez zmian jak w poprzednim kodzie ---
# (Tutaj wstaw funkcje z poprzedniej odpowiedzi)

def install_dependencies():
    system = platform.system()
    for dep in SYSTEM_DEPENDENCIES.get(system, []):
        if shutil.which(dep) is None:
            if ask_confirmation(f"Brak {dep}. Zainstalować?"):
                try:
                    if system == "Linux": run_cmd(["apt-get", "install", "-y", dep], sudo=True)
                    elif system == "Darwin": run_cmd(["brew", "install", dep])
                    elif system == "Windows": run_cmd(["winget", "install", "-e", "--id", dep])
                except Exception as e: print_status(f"Błąd {dep}: {e}", "ERROR")
    if PIP_DEPENDENCIES:
        print_status("Instalowanie bibliotek Python...", "INFO")
        run_cmd([sys.executable, "-m", "pip", "install"] + PIP_DEPENDENCIES)

def setup_global_command():
    system = platform.system()
    script_full_path = os.path.join(INSTALL_DIR, ENTRY_POINT)
    try:
        if system in ["Linux", "Darwin"]:
            bin_path = f"/usr/local/bin/{COMMAND_NAME}"
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
                print_status("Dodano do PATH. Zrestartuj terminal!", "WARN")
        print_status(f"Komenda '{COMMAND_NAME}' skonfigurowana globalnie!", "SUCCESS")
    except Exception as e:
        print_status(f"Błąd konfiguracji: {e}", "ERROR")

def main():
    print("\n🚀 Witamy w instalatorze PyLow\n" + "="*30)
    if not ask_confirmation("Czy chcesz zainstalować PyLow z najnowszej wersji (Releases)?"):
        sys.exit()
    install_project()
    install_dependencies()
    if ask_confirmation("Czy chcesz dodać komendę 'pylow' do terminala?"):
        setup_global_command()
    print("\n✅ Instalacja zakończona! Wpisz 'pylow', aby zacząć.")

if __name__ == "__main__":
    main()