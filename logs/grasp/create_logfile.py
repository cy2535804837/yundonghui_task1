#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志执行包装器 - 实时捕获并记录子进程的所有输出
用法: python3 create_logfile.py [--log-dir <目录>] <命令及其参数>
示例:
  python3 logs/grasp/create_logfile.py python3 -m compliant_grasp_execute.main
  python3 logs/grasp/create_logfile.py python3 -m grasp_pose_place_execute.main
  python3 logs/grasp/create_logfile.py --log-dir ./my_logs python3 -m compliant_grasp_execute.main
"""
import sys
import os
import datetime
import subprocess
import argparse
import threading

class Tee:
    """同时向多个文件对象（如终端和日志文件）写入数据"""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def read_output(pipe, write_func):
    """线程函数：读取管道并逐行写入"""
    try:
        for line in iter(pipe.readline, ''):
            write_func(line)
    finally:
        pipe.close()

def main():
    parser = argparse.ArgumentParser(description="运行命令并记录日志")
    parser.add_argument("--log-dir", default="logs/grasp",
                        help="日志文件存放目录（默认 logs/grasp）")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="要执行的命令及其参数")
    args = parser.parse_args()

    if not args.cmd:
        print("错误: 请提供要执行的命令")
        print("示例: python create_logfile.py python -m compliant_grasp_execute.main")
        sys.exit(1)

    # 创建日志目录（自动创建父目录）
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    now = datetime.datetime.now()
    log_name = f"grasp_{now.strftime('%y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_name)

    # 打开日志文件（UTF-8 编码）
    log_file = open(log_path, "w", encoding="utf-8")

    # 保存原始输出，替换为 Tee（终端 + 文件）
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print(f"[日志] 日志文件: {log_path}")
    print(f"[日志] 执行命令: {' '.join(args.cmd)}")

    # 启动子进程，管道捕获输出（文本模式，行缓冲）
    process = subprocess.Popen(
        args.cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1   # 行缓冲，保证实时性
    )

    # 创建两个线程分别读取 stdout 和 stderr
    t_out = threading.Thread(target=read_output, args=(process.stdout, sys.stdout.write))
    t_err = threading.Thread(target=read_output, args=(process.stderr, sys.stderr.write))
    t_out.start()
    t_err.start()

    # 等待子进程结束（同时线程也会自动结束）
    process.wait()
    t_out.join()
    t_err.join()

    # 获取子进程退出码
    exit_code = process.returncode

    # 清理还原
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

    print(f"[日志] 记录结束，日志已保存至 {log_path}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()