import os
import glob
import sys
import pandas as pd
from bs4 import BeautifulSoup

def extract_text_from_html(html_content):
    """从HTML中提取原始文本"""
    if pd.isna(html_content) or not isinstance(html_content, str):
        return ''
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    return '\n'.join(lines)

def process_csv(input_path, output_path):
    """处理单个CSV文件：把 full_text 列替换为提取后的纯文本"""
    df = pd.read_csv(input_path)

    if 'full_text' not in df.columns:
        print(f"  [跳过] 没有 full_text 列: {input_path}")
        return False

    # 直接覆盖 full_text 列
    df['full_text'] = df['full_text'].apply(extract_text_from_html)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  [完成] {len(df)} 条记录 -> {output_path}")
    return True

def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not csv_files:
        print(f"目录 {input_dir} 下没有找到 CSV 文件")
        return

    print(f"共找到 {len(csv_files)} 个 CSV 文件\n")

    success, skipped, failed = 0, 0, 0
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"处理: {filename}")
        output_path = os.path.join(output_dir, filename)  # 保持原名

        # 如果输入输出是同一文件，避免把自己当输入边读边写
        if os.path.abspath(csv_file) == os.path.abspath(output_path):
            print("  [跳过] 输出路径与输入相同，请指定不同的输出目录")
            skipped += 1
            continue

        try:
            if process_csv(csv_file, output_path):
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [失败] {e}")
            failed += 1

    print(f"\n处理结束：成功 {success} 个，跳过 {skipped} 个，失败 {failed} 个")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python process.py <输入目录> <输出目录>")
        print("说明: 输出目录必须与输入目录不同，避免覆盖原始文件")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if output_dir is None:
        print("错误: 请指定输出目录，避免覆盖原始文件")
        sys.exit(1)

    main(input_dir, output_dir)