import subprocess

with open('requirements.txt', 'r') as file:
    for line in file:
        package = line.strip()
        if package:
            result = subprocess.run(['pip', 'install', package], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to install {package}")
                print(result.stderr)