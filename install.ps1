# Pobiera install.py z Git i uruchamia go za pomocą Pythona
$RepoUrl = "https://raw.githubusercontent.com/pp1sp1/pylow/main/install.py"

Write-Host "🚀 Starting PyLow installation..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $RepoUrl -OutFile "install.py"
python install.py
Remove-Item "install.py"
