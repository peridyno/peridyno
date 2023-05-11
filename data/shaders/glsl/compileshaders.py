import argparse
import fileinput
import os
import subprocess
import sys
import json
import hashlib
import pickle

# json configuration for dynamic shader. type_list: dynamic type; file_list: dynamic shader folder and file name
json_config_str = {
    "type_list": [
        "int",
        "uint",
        "float"
    ],
    "file_list": [
        {
            "name": "Add.comp",
            "folder": "core"
        },
        {
            "name": "Reduce.comp",
            "folder": "core"
        },
        {
            "name": "Scan.comp",
            "folder": "core"
        },
        {
            "name": "Sort.comp",
            "folder": "core"
        },
        {
            "name": "SortByKey.comp",
            "folder": "core"
        }
    ]
}

# replace content for dynamic shader
fixed_type_def = "#define DataType T"
dot = "."

# use md5 to check that if shader changed
md5_modified = False
md5_file = "md5.data"

def loadMd5FileData():
    md5_path_file = os.path.abspath(os.path.dirname(__file__)) + os.sep + md5_file
    if not os.path.exists(md5_path_file):
        return {}

    with open(md5_path_file, 'rb') as f:
        md5_data = pickle.load(f)
        if checkIncFileChanged(md5_data):
            return {}
        return md5_data


def checkIncFileChanged(dict_data):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace('\\', '/')
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # .glsl is head file, so recompile all shaders if changed
            if file.endswith(".glsl"):
                input_file = os.path.join(root, file)
                if checkFileChanged(dict_data, input_file, False):
                    return True
    return False


def checkFileChanged(dict_data, file_path_name, update_dict):
    if not os.path.exists(file_path_name):
        return False

    with open(file_path_name, 'rb') as f:
        md5_obj = hashlib.md5()
        md5_obj.update(f.read())
        hash_value = md5_obj.hexdigest()
        file_path_name = file_path_name.replace('\\', '/')
        if dict_data.get(file_path_name, '') != hash_value:
            if update_dict:
                global md5_modified
                md5_modified = True
                dict_data[file_path_name] = hash_value
            return True
    return False


def writeMd5File(dict_data):
    if not md5_modified:
        return

    md5_file_path = os.path.abspath(os.path.dirname(__file__)) + os.sep + md5_file
    with open(md5_file_path, 'wb') as f:
        pickle.dump(dict_data, f)


def genNewFileContent(content, type_def):
    new_str = "#define DataType " + type_def
    new_content = content.replace(fixed_type_def, new_str)
    return new_content


def genTypeFile(content, type_def, dst_file):
    new_content = genNewFileContent(content, type_def)
    with open(dst_file, 'w') as f:
        f.write(new_content)


def genNewFiles(file_name, file_path, json_config, md5_dict):
    src_file = file_path + os.sep + file_name
    if not os.path.exists(src_file):
        print("the source file(%s) is not exist!" % src_file)
        return

    with open(src_file, 'r') as f:
        content = f.read()

    for type_def in json_config['type_list']:
        # file name format: {name}.{type}.{suffix}  e.g. Add.int.comp
        name = file_name.rsplit(dot, 1)[0]
        suffix = file_name.rsplit(dot, 1)[1]
        new_file_name = name + dot + type_def + dot + suffix
        dst_file = file_path + os.sep + new_file_name

        # create target file conditions: 1.target file not exists; 2.source file changed; 3. target file changed
        if not os.path.exists(dst_file) \
                or checkFileChanged(md5_dict, src_file, False) \
                or checkFileChanged(md5_dict, dst_file, False):
            genTypeFile(content, type_def, dst_file)


def generateDynamicShader(md5_dict):
    json_config_dump = json.dumps(json_config_str)
    json_config = json.loads(json_config_dump)
    for item in json_config['file_list']:
        file_name = item['name']
        file_folder = item['folder']
        file_path = os.path.abspath(os.path.dirname(__file__)) + os.sep + file_folder
        genNewFiles(file_name, file_path, json_config, md5_dict)


parser = argparse.ArgumentParser(description='Compile all GLSL shaders')
parser.add_argument('--glslang', type=str, help='path to glslangvalidator executable')
parser.add_argument('--g', action='store_true', help='compile with debug symbols')
args = parser.parse_args()

def findGlslang():
    def isExe(path):
        return os.path.isfile(path) and os.access(path, os.X_OK)

    if args.glslang != None and isExe(args.glslang):
        return args.glslang

    exe_name = "glslangvalidator"
    if os.name == "nt":
        exe_name += ".exe"

    for exe_dir in os.environ["PATH"].split(os.pathsep):
        full_path = os.path.join(exe_dir, exe_name)
        if isExe(full_path):
            return full_path
   
    #if os is linux platom
    #return "/usr/bin/glslangValidator"
    sys.exit("Could not find DXC executable on PATH, and was not specified with --dxc")

glslang_path = findGlslang()
md5_dict = loadMd5FileData()
generateDynamicShader(md5_dict)

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\', '/')
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".glsl"):
            input_file = os.path.join(root, file)
            checkFileChanged(md5_dict, input_file, True)

        if file.endswith(".vert") or file.endswith(".frag") or file.endswith(".comp") or file.endswith(".geom") or file.endswith(".tesc") or file.endswith(".tese") or file.endswith(".rgen") or file.endswith(".rchit") or file.endswith(".rmiss"):
            input_file = os.path.join(root, file)
            output_file = input_file + ".spv"

            # input file not changed and output file exists, skip
            if os.path.exists(output_file) and not checkFileChanged(md5_dict, input_file, True):
                continue

            add_params = ""
            if args.g:
                add_params = "-g"

            if file.endswith(".rgen") or file.endswith(".rchit") or file.endswith(".rmiss"):
                    add_params = add_params + " --target-env vulkan1.2"
            else:
                add_params = add_params + " --target-env vulkan1.1";

            res = subprocess.call("\"%s\" -V %s -o %s %s" % (glslang_path, input_file, output_file, add_params), shell=True)
            # res = subprocess.call([glslang_path, '-V', input_file, '-o', output_file, add_params], shell=True)
            if res != 0:
                sys.exit()

writeMd5File(md5_dict)
