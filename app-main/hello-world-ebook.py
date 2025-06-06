import zipfile
from pathlib import Path

def main():
    zip_path = Path(__file__).with_name('Kongetro.epub.zip')
    if not zip_path.exists():
        print(f"{zip_path} not found")
        return
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print('Files in zip:', zf.namelist())
        first_file = zf.namelist()[0]
        with zf.open(first_file) as f:
            content = f.read().decode('utf-8')
            print('Content:', content)

if __name__ == '__main__':
    main()