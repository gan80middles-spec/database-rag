# tree_simple.py
# 递归打印目录层级（类似 Unix 的 tree），只用标准库
import os
import sys

def list_children(path):
    """返回按：目录在前、文件在后、名称升序 的条目列表"""
    try:
        with os.scandir(path) as it:
            dirs, files = [], []
            for e in it:
                (dirs if e.is_dir(follow_symlinks=False) else files).append(e)
            dirs.sort(key=lambda x: x.name.lower())
            files.sort(key=lambda x: x.name.lower())
            return dirs + files
    except PermissionError:
        print(f"[权限不足] {path}")
    except FileNotFoundError:
        pass
    return []

def print_tree(path, prefix=""):
    """打印 path 下的树形结构，返回 (目录数, 文件数)"""
    dirs_cnt = files_cnt = 0
    entries = list_children(path)
    total = len(entries)
    for i, e in enumerate(entries):
        connector = "└── " if i == total - 1 else "├── "
        line = prefix + connector + e.name
        print(line)
        if e.is_dir(follow_symlinks=False):
            dirs_cnt += 1
            ext = "    " if i == total - 1 else "│   "
            d, f = print_tree(e.path, prefix + ext)
            dirs_cnt += d
            files_cnt += f
        else:
            files_cnt += 1
    return dirs_cnt, files_cnt

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root_abs = os.path.abspath(root)
    if not os.path.exists(root_abs):
        print(f"[不存在] {root}")
        sys.exit(1)
    if os.path.isdir(root_abs):
        title = os.path.basename(root_abs) or root_abs
        print(title)
        d, f = print_tree(root_abs)
        print(f"\n{d} directories, {f} files")
    else:
        # 如果传的是文件，就只打印这个文件名
        print(os.path.basename(root_abs))
        print("\n0 directories, 1 files")

if __name__ == "__main__":
    main()
