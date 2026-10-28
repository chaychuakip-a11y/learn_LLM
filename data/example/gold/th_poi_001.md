## 口径
示例车厂A，泰语导航 POI，车端引擎，按句准。版本 eng_1.4.2，资源槽 nav-th-2026.03。要分清 เซ็นทรัล 是缺词，还是词在词典里但路径不通。

## 脚本在查什么
核对词典、WFST 的 path_complete 和 missing，并检查完整标注有没有进入 patch 或 base。

## 结论
u1 的 เซ็นทรัล 在词典中，lexicon_miss 为空。WFST 的 missing 仍是这个词，path_complete 为 false，in_lexicon_but_path_blocked 里有它。词在词典里不等于路径可达。完整标注没有出现在候选里，获胜分支是 patch，REC 也不是标注。这是路径不可达，不是出现后输给其他分支。

## 待核实
acoustic dump 不存在，dump_present 为 false。声学路径这次不能下结论。

## 正式门禁
这只说明这一条泰语 POI 的路径缺口。正式门禁要在同一车厂、泰语和固定基线上看完整回归有没有改善，以及有没有回退。
