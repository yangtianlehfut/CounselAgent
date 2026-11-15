import csv
import sqlite3
import os

CSV_PATH = "crime.csv"
DB_PATH = "crime.db"


# ---------- 1. 建库建表 ----------
def create_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crime (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             VARCHAR(100) NOT NULL,
            article          VARCHAR(100),
            constitution     TEXT,
            non_constitution TEXT,
            UNIQUE(name)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crime_question (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            crime_id INTEGER NOT NULL,
            keyword  VARCHAR(20) NOT NULL,
            question TEXT,
            FOREIGN KEY (crime_id) REFERENCES crime(id),
            UNIQUE(crime_id, keyword)
        )
    """)
    conn.commit()


# ---------- 2. 读取 CSV 并入库 ----------
def load_csv(conn, csv_path):
    cur = conn.cursor()
    current_crime_id = None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过第一个表头

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue  # 跳过空行

            row = [c.strip() for c in row] + [""] * (6 - len(row))

            # 跳过重复出现的表头行
            if row[0] == "罪名":
                continue

            name, article, constitution, non_constitution, keyword, question = row[:6]

            if name:
                cur.execute("""
                    INSERT INTO crime(name, article, constitution, non_constitution)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        article          = excluded.article,
                        constitution     = excluded.constitution,
                        non_constitution = excluded.non_constitution
                """, (name, article, constitution, non_constitution))
                cur.execute("SELECT id FROM crime WHERE name=?", (name,))
                current_crime_id = cur.fetchone()[0]
            else:
                if current_crime_id is None:
                    print("警告：孤立的反问要素行，已跳过：", row)
                    continue

            if keyword:
                cur.execute("""
                    INSERT INTO crime_question(crime_id, keyword, question)
                    VALUES (?, ?, ?)
                    ON CONFLICT(crime_id, keyword) DO UPDATE SET
                        question = excluded.question
                """, (current_crime_id, keyword, question))

    conn.commit()


# ---------- 3. 全量校验 ----------
def verify(conn):
    cur = conn.cursor()

    # 3.1 拉取全部罪名
    cur.execute("""
        SELECT id, name, article, constitution, non_constitution
        FROM crime
        ORDER BY id
    """)
    crimes = cur.fetchall()

    # 3.2 拉取全部要素，按 crime_id 分组
    cur.execute("""
        SELECT crime_id, keyword, question
        FROM crime_question
        ORDER BY crime_id, id
    """)
    q_map = {}
    for cid, kw, qs in cur.fetchall():
        q_map.setdefault(cid, []).append((kw, qs))

    errors = []
    total_questions = 0

    print("=" * 80)
    print("全量校验明细")
    print("=" * 80)

    for cid, name, article, constitution, non_constitution in crimes:
        print(f"\n[{cid}] {name}")
        print(f"    法条  : {article or '【空】'}")
        print(f"    构成  : {(constitution or '【空】')[:60]}"
              f"{'...' if constitution and len(constitution) > 60 else ''}")
        print(f"    不构成: {(non_constitution or '【空】')[:60]}"
              f"{'...' if non_constitution and len(non_constitution) > 60 else ''}")

        for field, val in (("name", name), ("article", article),
                           ("constitution", constitution),
                           ("non_constitution", non_constitution)):
            if not val:
                errors.append(f"[{cid}] {name} 字段 {field} 为空")

        items = q_map.get(cid, [])
        if not items:
            errors.append(f"[{cid}] {name} 无任何反问要素")
            print("    要素  : 【无】")
        else:
            print(f"    要素 ({len(items)} 条):")
            for kw, qs in items:
                total_questions += 1
                mark = "" if qs else "  ⚠ question 为空"
                print(f"        - {kw:<8s} {qs}{mark}")
                if not kw:
                    errors.append(f"[{cid}] {name} 存在空 keyword")
                if not qs:
                    errors.append(f"[{cid}] {name} 要素 {kw} 的 question 为空")

    # 3.3 孤立要素检查
    crime_ids = {c[0] for c in crimes}
    cur.execute("SELECT id, crime_id, keyword FROM crime_question")
    for qid, cid, kw in cur.fetchall():
        if cid not in crime_ids:
            errors.append(f"crime_question.id={qid} 的 crime_id={cid} 无对应罪名（要素：{kw}）")

    # 3.4 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"罪名总数     : {len(crimes)}")
    print(f"反问要素总数 : {total_questions}")
    print(f"问题条数     : {len(errors)}")

    if errors:
        print("\n发现问题：")
        for e in errors:
            print("  ✗", e)
    else:
        print("\n✓ 全量校验通过，无异常。")

    return errors


# ---------- main ----------
if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # 演示用：每次重建；生产环境可去掉

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        create_tables(conn)
        load_csv(conn, CSV_PATH)
        verify(conn)
    finally:
        conn.close()

    print("\n完成，数据库：", DB_PATH)