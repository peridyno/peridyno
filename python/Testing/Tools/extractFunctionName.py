import re
import os
from pathlib import Path


def extract_quoted_text_from_file(input_path, output_path):
    """从单个文件中提取引号内容"""
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        valid_quotes = []

        for line in lines:
            quoted_texts = re.findall(r'["\'](.*?)["\']', line)

            for text in quoted_texts:
                if text.strip():
                    if '/' in text:
                        continue

                    if text and text[0].isupper() and line.strip().endswith('typestr;'):
                        text = text + '3f'

                    valid_quotes.append(text)

        with open(output_path, 'a', encoding='utf-8') as file:
            for text in valid_quotes:
                if text.strip():
                    file.write(text + '\n')

        return len(valid_quotes), input_path

    except Exception as e:
        print(f"处理文件 {input_path} 时发生错误：{e}")
        return 0, input_path


def extract_quoted_text_from_folder(folder_path, output_path, extensions=None):
    """
    从文件夹中所有文件中提取引号内容

    Args:
        folder_path: 输入文件夹路径
        output_path: 输出文件路径
        extensions: 要处理的文件扩展名列表，默认为None（处理所有文件）
    """
    if extensions is None:
        extensions = ['.h', '.cpp', '.c', '.py', '.java', '.js', '.html', '.css', '.txt']

    # 确保输出文件为空
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("")

    total_files = 0
    total_quotes = 0

    try:
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"错误：文件夹 {folder_path} 不存在")
            return

        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                quotes_count, processed_file = extract_quoted_text_from_file(str(file_path), output_path)
                total_quotes += quotes_count
                total_files += 1
                print(f"已处理: {processed_file} - 找到 {quotes_count} 个引号内容")

        print(f"\n处理完成！")
        print(f"共处理 {total_files} 个文件")
        print(f"共提取 {total_quotes} 个引号内容")
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"处理文件夹时发生错误：{e}")


def remove_duplicates_in_place(file_path):
    """
    对文本文件进行行级去重，并写回原文件（保持原顺序）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_count = len(lines)

        seen = set()
        unique_lines = []

        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)

        new_count = len(unique_lines)
        removed_count = original_count - new_count

        print(f"去重完成！")
        print(f"原文件行数: {original_count}")
        print(f"去重后行数: {new_count}")
        print(f"移除重复行: {removed_count}")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except PermissionError:
        print(f"错误：没有写入文件 {file_path} 的权限")
    except Exception as e:
        print(f"发生错误：{e}")

# 使用示例
if __name__ == "__main__":
    # 处理单个文件夹
    input_folder = "../../Dynamics"
    output_file = "PromptFunctionName"

    # 只处理特定扩展名的文件
    file_extensions = ['.h']

    extract_quoted_text_from_folder(input_folder, output_file, file_extensions)

    remove_duplicates_in_place(output_file)

    # 或者处理所有支持的文件类型
    # extract_quoted_text_from_folder(input_folder, output_file)
