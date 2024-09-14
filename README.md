# DataFountain-LLM-Text-Content-Judge

    Contest solution for 基于大模型的文本内容智能评判

----

Contest page: https://www.datafountain.cn/competitions/1032  
Team Name: 你也是龙  


### Quickstart

- `python judge.py`


### Experiments

Our final solution for each task:

- FLU: subjective output quality
  - no idea...
- NOR: subjective instruction obedience
  - summarize the given outputs via an LLM
  - measure the semantic similarities between requirements and summary with a bunch of sentence-vector models
  - regress by RandomForestRegressor with similarity scores and token lengths as features
- CHC: objective knowlege choices
  - pure regex-based template matching

| method | submit score ↑ | data split |
| :-: | :-: | :-: |
| random | 0.43252595156 | rank A |
| mean   | 0.49980007997 | rank A |
| final  |               | rank B |


----
by Armit
2024/08/24 
