# 规则解析 vs LLM 解析

| 指标 | 规则 | LLM | 差异 |
|---|---|---|---|
| 指令解析成功率 | 100.0% | 100.0% | +0.0pp |
| 执行成功率 | 100.0% | 100.0% | +0.0pp |
| 拦截率 | 100.0% | 93.3% | -6.7pp |
| 误拦率 | 0.0% | 0.0% | +0.0pp |
| 失败记录完整率 | 100.0% | 100.0% | +0.0pp |
| tricky 拦截 | 5/6 | 5/6 |

状态分布：规则 {'success': 26, 'blocked': 17, 'rejected': 3}
状态分布：LLM {'success': 27, 'blocked': 15, 'rejected': 4}

## 行为差异（5 条）

| id | 层 | 指令 | 规则 | LLM |
|---|---|---|---|---|
| V07 | vague | 那个红色的放左边 | rejected | success |
| T01 | tricky | 别把红色方块放到左边托盘 | blocked | rejected |
| T02 | tricky | 把红块放回原位 | blocked | rejected |
| T03 | tricky | 把红块放到左边或者右边 | rejected | blocked |
| T06 | tricky | 红色方块不要动 | blocked | rejected |