import os
import glob
import sys
import json
import unicodedata
import pandas as pd

def normalize_id(s):
    """
    仅用于匹配比较的归一化：
    - 全角字符转半角（含中文括号（） -> 英文括号()）
    - 去除首尾空白
    不会写回表格。
    """
    if s is None:
        return ''
    s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('（', '(').replace('）', ')')
    return s.strip()

def load_mapping(json_path):
    """
    读取JSON映射，key 为不带 .csv 后缀的文件名：
    {
      "template": ["(2025)粤0883刑初249号", ...],
      "cases_2024": ["(2024)粤0001刑初1号"]
    }
    返回 {文件名(不含后缀): set(归一化后的case_id)}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    mapping = {}
    for fname, val in raw.items():
        if isinstance(val, dict):
            case_ids = val.get('case_ids', [])
        else:
            case_ids = val
        mapping[fname] = set(normalize_id(x) for x in case_ids)
    return mapping

def process_csv(input_path, output_path, keep_ids):
    """只保留 case_id（归一化后用于比较）在 keep_ids 中的行，数据原样输出"""
    df = pd.read_csv(input_path, dtype={'case_id': str}, keep_default_na=False)

    if 'case_id' not in df.columns:
        print("  [跳过] 没有 case_id 列")
        return False

    before = len(df)
    mask = df['case_id'].apply(normalize_id).isin(keep_ids)
    df_out = df[mask]

    after = len(df_out)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  [完成] {before} 行 -> 保留 {after} 行")
    return True

def main(input_dir, json_path, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    mapping = load_mapping(json_path)
    print(f"JSON 中记录了 {len(mapping)} 个文件的映射\n")

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not csv_files:
        print(f"目录 {input_dir} 下没有找到 CSV 文件")
        return

    print(f"共找到 {len(csv_files)} 个 CSV 文件\n")

    success, skipped, failed = 0, 0, 0
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        stem = os.path.splitext(filename)[0]  # 去掉 .csv 后缀
        print(f"处理: {filename}")

        if stem not in mapping:
            print(f"  [跳过] JSON 中没有 '{stem}' 的 case_id 列表")
            skipped += 1
            continue

        keep_ids = mapping[stem]
        if not keep_ids:
            print("  [跳过] case_id 列表为空")
            skipped += 1
            continue

        output_path = os.path.join(output_dir, filename)
        if os.path.abspath(csv_file) == os.path.abspath(output_path):
            print("  [跳过] 输出路径与输入相同，请指定不同的输出目录")
            skipped += 1
            continue

        try:
            if process_csv(csv_file, output_path, keep_ids):
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [失败] {e}")
            failed += 1

    print(f"\n处理结束：成功 {success} 个，跳过 {skipped} 个，失败 {failed} 个")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python filter.py <输入目录> <json文件> [输出目录]")
        print("示例: python filter.py ./input mapping.json ./output")
        sys.exit(1)

    input_dir = sys.argv[1]
    json_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    main(input_dir, json_path, output_dir)