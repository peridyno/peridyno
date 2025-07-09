import os
import re
from pathlib import Path


def process_def_statements(directory):
    print(f"开始处理目录: {directory}")

    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return

    # 更灵活的正则表达式，匹配各种格式
    pattern = re.compile(
        r'\.def\s*\(\s*"([^"]+)"\s*,\s*&\s*([^:]+)::([^,\s]+)\s*(?:,\s*[^)]+)?\)')

    processed_files = 0
    modified_statements = 0

    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in ('.cpp', '.h'):
            try:
                # 尝试多种编码
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content = file_path.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"无法解码文件: {file_path}")
                    continue

                processed_files += 1
                changes_made = False

                def replacer(match):
                    nonlocal changes_made
                    original_name = match.group(1)
                    class_name = match.group(2)
                    method_name = match.group(3)

                    print(f"\n在文件 {file_path} 中找到匹配:")
                    print(f"完整匹配: {match.group(0)}")
                    print(f"参数1: {match.group(1)}")
                    print(f"类名: {match.group(2)}")
                    print(f"方法名: {match.group(3)}")

                    # 标准化比较（忽略大小写、下划线和空格）
                    norm_original = original_name.lower().replace('_', '').replace(' ', '')
                    norm_method = method_name.lower().replace('_', '').replace(' ', '')

                    if norm_original == norm_method:
                        changes_made = True
                        # 保留原始格式，只修改名称部分
                        new_stmt = match.group(0).replace(
                            f'"{original_name}"',
                            f'"{method_name}"'
                        )
                        print(f"修改: {file_path}")
                        print(f"原始: {match.group(0)}")
                        print(f"更新: {new_stmt}")
                        return new_stmt
                    return match.group(0)

                new_content = pattern.sub(replacer, content)

                if new_content != content:
                    file_path.write_text(new_content, encoding='utf-8')
                    modified_statements += sum(
                        1 for _ in pattern.finditer(content))
                    print(f"已保存修改到 {file_path}")

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    print(f"\n处理完成！")
    print(f"扫描文件总数: {processed_files}")
    print(f"修改语句数量: {modified_statements}")


if __name__ == "__main__":
    target_dir = "H:/program/Simulation/unibeam/wt-test/peridyno-web/python/Testing"
    #target_dir = "E:/Program/Simulation/Wt-test/peridyno-web/python/Testing"
    process_def_statements(target_dir)
