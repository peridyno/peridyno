import os
import re


def analyze_directory(directory):
    print(f"开始分析目录: {directory}")

    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return

    cpp_pattern = re.compile(
        r'\.def\s*\(\s*"([^"]+)"\s*,\s*&\s*([^:]+)::([^,\s]+)\s*,\s*[^)]+\)')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.cpp', '.h')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        matches = cpp_pattern.finditer(content)

                        for match in matches:
                            print(f"\n在文件 {file_path} 中找到匹配:")
                            print(f"完整匹配: {match.group(0)}")
                            print(f"参数1: {match.group(1)}")
                            print(f"类名: {match.group(2)}")
                            print(f"方法名: {match.group(3)}")

                except UnicodeDecodeError:
                    print(f"无法读取文件(编码问题): {file_path}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")


if __name__ == "__main__":
    target_dir = "H:/program/Simulation/unibeam/wt-test/peridyno-web/python/Testing"
    analyze_directory(target_dir)
    print("\n分析完成！")
