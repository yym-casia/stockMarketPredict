# -*- coding: utf-8 -*-
"""批量补全股票池行业标签（东方财富行业板块）"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sector_service import enrich_pool_sectors, unify_pool_sectors_eastmoney


def main():
    parser = argparse.ArgumentParser(description='统一股票池为东方财富行业板块')
    parser.add_argument('--force', action='store_true', help='强制重新拉取行业映射')
    parser.add_argument('--overwrite-all', action='store_true', help='覆盖所有板块标签')
    parser.add_argument('--eastmoney-only', action='store_true', help='仅使用东方财富标准板块名')
    parser.add_argument('--no-fill', action='store_true', help='跳过 f127 桥接补全')
    args = parser.parse_args()

    print('=' * 60)
    print('统一板块标签（东方财富行业板块）')
    print('=' * 60)

    if args.eastmoney_only or args.overwrite_all:
        stats = unify_pool_sectors_eastmoney(
            force_fetch=args.force,
            fill_missing=not args.no_fill,
        )
    else:
        stats = enrich_pool_sectors(force_fetch=args.force, overwrite_all=args.overwrite_all)

    print('\n结果:')
    print(f"  股票池: {stats['pool_size']} 只")
    print(f"  本次更新: {stats['updated']} 只")
    print(f"  东方财富板块: {stats['mapped']} 只")
    print(f"  未归类(其他): {stats.get('other', 0)} 只")
    print(f"  行业库总量: {stats['industry_map_size']} 只")
    print(f"  标准板块数: {stats.get('board_count', 0)} 个")


if __name__ == '__main__':
    main()
