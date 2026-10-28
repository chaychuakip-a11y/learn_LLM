## 口径
示例车厂B，德语导航，车端引擎，按句准。版本 eng_2.0.1，资源槽 nav-de-2026.03。看标注是没进候选，还是进了候选但没有获胜。

## 脚本在查什么
列出 patch 和 fsa 的分数，计算获胜分支与目标候选的分差，并对照声学 dump 的 best_path_text 和 state_map_applied。

## 结论
hause 在词典里。完整标注出现在 fsa 候选中，分数 -9.5，没有获胜。获胜分支是 patch，分数 -8，文本是 navigiere nach hotel。winner_minus_target 为 1.5。标注相对 REC 是 hause 被替换成 hotel。dump 的 best_path_text 与获胜分支一致。

## 待核实
dump 里的 state_ids 还没有做该声学模型的映射，state_map_applied 为 false。音素层不能写成已验证。分差只有 1.5，不能凭这一条去改权重。

## 正式门禁
这是单条分支竞争的修复证据。正式门禁要回答目标有没有改善，以及同一德语导航范围内有没有回退。
