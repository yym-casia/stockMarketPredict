# -*- coding: utf-8 -*-
"""多策略候选合并"""

from typing import Dict, List, Tuple


def merge_multi_strategy_candidates(
    strategy_lists: Dict[str, Tuple[List[Dict], int]],
    priority: List[str] = None,
) -> List[Dict]:
    """
    合并多策略候选。
    strategy_lists: {strategy_type: (candidates, max_slots)}
    priority: 优先占名额的顺序
    """
    priority = priority or ['limit_up', 'doubler', 'trend']
    seen = set()
    merged = []

    for stype in priority:
        cands, slots = strategy_lists.get(stype, ([], 0))
        for c in cands:
            c.setdefault('strategy_type', stype)
        for c in sorted(cands, key=lambda x: x.get('score', 0), reverse=True):
            if slots <= 0:
                break
            code = c.get('code', c.get('code_clean', ''))
            if not code or code in seen:
                continue
            if len([x for x in merged if x.get('strategy_type') == stype]) >= slots:
                continue
            seen.add(code)
            merged.append(c)

    merged.sort(key=lambda x: (priority.index(x.get('strategy_type', 'trend'))
                              if x.get('strategy_type') in priority else 99,
                              x.get('score', 0)), reverse=True)
    return merged
