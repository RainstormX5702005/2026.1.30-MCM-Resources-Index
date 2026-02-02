"""
一键运行所有准确性检验分析
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    """运行指定的Python脚本"""
    print(f"\n{'='*80}")
    print(f"运行: {script_name}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, script_name], check=True, capture_output=False, text=True
        )
        print(f"✓ {script_name} 执行成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} 执行失败")
        print(f"错误信息: {e}")
        return False


def main():
    print("=" * 80)
    print("准确性检验 - 一键运行所有分析")
    print("=" * 80)

    scripts = [
        ("q5_accuracy_check.py", "主要准确性分析（生成图表和错误报告）"),
        ("q5_error_summary.py", "错误模式总结"),
    ]

    success_count = 0

    for script, description in scripts:
        print(f"\n{description}")
        if run_script(script):
            success_count += 1

    print("\n" + "=" * 80)
    print(f"完成！成功运行 {success_count}/{len(scripts)} 个脚本")
    print("=" * 80)

    print("\n生成的文件:")
    output_dir = Path("output/question5_res")
    if output_dir.exists():
        print("\n图片文件:")
        for png in output_dir.glob("*.png"):
            print(f"  - {png.name}")

        print("\n数据文件:")
        for csv in ["prediction_errors_pct.csv", "prediction_errors_rank.csv"]:
            if (output_dir / csv).exists():
                print(f"  - {csv}")

        print("\n文档:")
        for md in ["ACCURACY_CHECK_REPORT.md", "README_ACCURACY_CHECK.md"]:
            if (output_dir / md).exists():
                print(f"  - {md}")

    print(f"\n所有结果保存在: {output_dir.absolute()}")
    print("\n请查看 README_ACCURACY_CHECK.md 了解详细使用说明")


if __name__ == "__main__":
    main()
