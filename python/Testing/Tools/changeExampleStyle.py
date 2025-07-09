import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def underscore_to_camel(word):
    parts = word.split('_')
    return parts[0] + ''.join(x.capitalize() for x in parts[1:])


def process_file(file_path):
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            pattern = re.compile(r'\b[a-z_]+[a-z0-9_]*\b')
            new_content = pattern.sub(
                lambda m: underscore_to_camel(m.group(0)),
                content
            )
            f.seek(0)
            f.write(new_content)
            f.truncate()
        print(f'✓ Processed: {file_path}')
    except Exception as e:
        print(f'✗ Error processing {file_path}: {str(e)}')


def process_directory(directory):
    try:
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.is_dir():
            print(f'✗ Error: {dir_path} is not a valid directory')
            return False

        py_files = [str(p) for p in dir_path.rglob('*.py')]
        if not py_files:
            print('ℹ No .py files found in directory')
            return False

        print(f'ℹ Found {len(py_files)} .py files to process')
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
            list(executor.map(process_file, py_files))
        return True
    except Exception as e:
        print(f'✗ Directory processing error: {str(e)}')
        return False


if __name__ == '__main__':
    print("Python文件命名风格转换工具 (下划线 → 驼峰)")
    print("=" * 50)

    while True:
        target_dir = input("\n请输入要处理的目录路径(或输入q退出): ").strip()
        if target_dir.lower() == 'q':
            break

        if not target_dir:
            print("⚠ 请输入有效路径")
            continue

        if process_directory(target_dir):
            print("\n✅ 转换完成!")
        else:
            print("\n❌ 处理失败，请检查路径是否正确")
