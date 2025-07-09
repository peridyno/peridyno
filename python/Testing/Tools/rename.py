import os

# 要删除的文件路径
file_to_delete = "E:\\Program\\Simulation\\Wt-test\\build\\bin\\Debug\\PyPeridyno.pyd"
# 要重命名的文件路径
file_to_rename = "E:\\Program\\Simulation\\Wt-test\\build\\bin\\Debug\\PyPeridyno-1.2.0d.cp312-win_amd64.pyd"
# 重命名后的文件路径
new_file_name = "E:\\Program\\Simulation\\Wt-test\\build\\bin\\Debug\\PyPeridyno.pyd"

# # 要删除的文件路径
# file_to_delete = "H:\\program\\Simulation\\unibeam\\wt-test\\build\\bin\\Debug\\PyPeridyno.pyd"
# # 要重命名的文件路径
# file_to_rename = "H:\\program\\Simulation\\unibeam\\wt-test\\build\\bin\\Debug\\PyPeridyno-1.2.0d.cp312-win_amd64.pyd"
# # 重命名后的文件路径
# new_file_name = "H:\\program\\Simulation\\unibeam\\wt-test\\build\\bin\\Debug\\PyPeridyno.pyd"
# 删除文件
try:
    os.remove(file_to_delete)
    print(f"文件 {file_to_delete} 已被删除")
except FileNotFoundError:
    print(f"文件 {file_to_delete} 不存在")
except PermissionError:
    print(f"没有权限删除文件 {file_to_delete}")
except Exception as e:
    print(f"删除文件 {file_to_delete} 时发生错误：{e}")

# 重命名文件
try:
    os.rename(file_to_rename, new_file_name)
    print(f"文件 {file_to_rename} 已被重命名为 {new_file_name}")
except FileNotFoundError:
    print(f"文件 {file_to_rename} 不存在")
except FileExistsError:
    print(f"文件 {new_file_name} 已存在")
except PermissionError:
    print(f"没有权限重命名文件 {file_to_rename}")
except Exception as e:
    print(f"重命名文件 {file_to_rename} 时发生错误：{e}")