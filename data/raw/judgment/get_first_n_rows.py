import os
import glob
import sys
import pandas as pd

def process_csv(input_path, output_path, max_rows=100):
    """处理单个CSV文件：最多保留前max_rows行数据（不含表头）"""
    df = pd.read_csv(input_path)

    if len(df) > max_rows:
        df = df.head(max_rows)
        print(f"  [截断] 原 {len(df)} 行 -> 保留前 {max_rows} 行")
    else:
        print(f"  [保留] 共 {len(df)} 行，未超过 {max_rows} 行，原样输出")

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return True

def main(input_dir, output_dir=None, max_rows=100):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not csv_files:
        print(f"目录 {input_dir} 下没有找到 CSV 文件")
        return

    print(f"共找到 {len(csv_files)} 个 CSV 文件，每文件最多保留 {max_rows} 行数据\n")

    success, failed = 0, 0
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"处理: {filename}")
        output_path = os.path.join(output_dir, filename)

        if os.path.abspath(csv_file) == os.path.abspath(output_path):
            print("  [跳过] 输出路径与输入相同，请指定不同的输出目录")
            continue

        try:
            process_csv(csv_file, output_path, max_rows)
            success += 1
        except Exception as e:
            print(f"  [失败] {e}")
            failed += 1

    print(f"\n处理结束：成功 {success} 个，失败 {failed} 个")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python truncate.py <输入目录> [输出目录] [max_rows]")
        print("示例: python truncate.py ./input ./output 100")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    main(input_dir, output_dir, max_rows)