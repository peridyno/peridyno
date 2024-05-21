import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.getcwd()
    relative_path = "../../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scene = dyno.SceneGraph()
gltf = dyno.GltfLoader3f()
gltf.var_file_name().set_value(dyno.FilePath(filePath("Jeep/JeepGltf/jeep.gltf")))

scene.add_node(gltf)

