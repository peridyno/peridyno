import os

path_dirs = os.environ.get('PATH', '').split(os.pathsep)
for dir_path in path_dirs:
    if 'qt' in dir_path.lower() and 'bin' in dir_path:
        normalized_path = os.path.normpath(dir_path)
        if os.path.exists(normalized_path):
            os.add_dll_directory(normalized_path)
