# Unknown Difficulty 来源问题报告

生成日期：2026-07-08

## 结论

当前数据和脚本里没有发现真正的 `difficulty_name = "unknown"`。

目前被误判或容易被称为 “unknown difficulty” 的来源主要有两类：

1. **未知等级 level**：源 Maidata 中大量 `lv_5=?`，进入训练 manifest 后表现为 `level_raw="?"`、`level=null`，并被标记为 `non_numeric_level`。
2. **缺失 re:master 难度**：当前 parser 只读取 `inote_1` 到 `inote_5`，训练 manifest 也只包含 easy/basic/advanced/expert/master。ID 测试批请求的 re:master 在源 manifest 中不存在，因此被记录为 `missing_requested_difficulties`，不是 unknown difficulty。

## 证据

### 1. 难度名来源固定为 1-5

`src/maichart/preprocess.py` 中难度名映射为：

```python
DIFFICULTY_NAMES = {
    1: "easy",
    2: "basic",
    3: "advanced",
    4: "expert",
    5: "master",
}
```

生成训练 manifest 时使用：

```python
difficulty_name = DIFFICULTY_NAMES.get(difficulty_index, f"difficulty_{difficulty_index}")
```

因此当前训练 manifest 中不会生成 `unknown` 难度名；如果未来出现 6 号以上难度，会被命名成 `difficulty_6`，而不是 `unknown`。

### 2. Maidata parser 当前只读取 1-5

`src/maichart/maidata.py` 中：

```python
DIFFICULTY_RANGE = range(1, 6)
```

并且只从 `lv_1..lv_5` / `inote_1..inote_5` 建立 `RawDifficultyBlock`。

这意味着即使原始数据里存在 re:master/宴/第 6 难度形式，当前 parser 不会把它纳入训练 manifest。现有 `training_manifest_full.json` 的难度分布也只包含：

| difficulty_index | difficulty_name | count |
| --- | --- | ---: |
| 1 | easy | 1300 |
| 2 | basic | 1301 |
| 3 | advanced | 1298 |
| 4 | expert | 1298 |
| 5 | master | 1298 |

### 3. 全量 manifest 中没有 unknown difficulty

对 `manifests/*.json` 的检查结果：

| manifest | difficulty_name unknown | difficulty_index null | sample_id difficulty_unknown |
| --- | ---: | ---: | ---: |
| training_manifest_full.json | 0 | 0 | 0 |
| small_batch_v1.json | 0 | 0 | 0 |
| id_test_batch_v1.json | 0 | 0 | 0 |
| training_manifest_limit*.json | 0 | 0 | 0 |
| training_manifest_overfit1.json | 0 | 0 | 0 |

### 4. 实际大量存在的是 unknown level

`training_manifest_full.json` 中存在 1172 个 `level_raw="?"` 或 `level=null` 的 difficulty，其中大部分是 master：

| difficulty_index | difficulty_name | unknown level count |
| --- | --- | ---: |
| 2 | basic | 4 |
| 3 | advanced | 1 |
| 4 | expert | 1 |
| 5 | master | 1166 |

manifest summary 里也已有：

```json
"filter_reason_counts": {
  "missing_chart": 1300,
  "note_count_too_low": 1300,
  "non_numeric_level": 1172,
  "audio_unreadable": 4,
  "audio_features_failed": 4
}
```

以及 `samples_with_level_unknown` 列表。

所以最主要的问题不是 unknown difficulty，而是 master 等级字段大量不可解析。

### 5. ID 测试批中的 re:master 缺失不是 unknown

`manifests/id_test_batch_v1.json` 当前生成结果：

| item | count |
| --- | ---: |
| 输入 ID | 30 |
| 匹配歌曲 | 30 |
| expert 样本 | 30 |
| master 样本 | 30 |
| re:master/remaster 样本 | 0 |
| missing remaster entries | 30 |

原因是源 manifest 没有任何 `difficulty_name=remaster` 或 `difficulty_index=6` 条目。脚本将它们记录在：

```json
metadata.missing_requested_difficulties
```

这表示“请求的 re:master 在源数据中不存在”，不是 parser 识别成了 unknown difficulty。

### 6. 小批量脚本中唯一的 difficulty_unknown 兜底

`tools/build_small_batch_manifest.py` 中有一处 sample_id 兜底：

```python
sample_id = f"{music_id}_difficulty_{difficulty_index if difficulty_index is not None else 'unknown'}"
```

这只会在输入 manifest 缺少 `difficulty_index` / `index` 时生成 `difficulty_unknown` 的 sample_id。

当前检查结果显示：现有 manifest 没有触发这个兜底。因此它是潜在风险点，不是当前问题来源。

## 影响

1. 训练清单中 `level=null` 的样本通常会被 `non_numeric_level` 过滤，尤其是大量 master 谱面。
2. 如果下游 UI 或评估脚本把 `level=null` 显示为 unknown difficulty，会造成概念混淆。
3. 当前 ID 测试批无法包含 re:master，因为预处理源 manifest 根本没有 re:master 难度。
4. expert 源等级存在不少 `14`，这与“expert 通常不超过 13”的预期冲突；这些已在 `id_test_batch_v1.json` 中记录为 `expert_level_above_13`，但没有强行改写源数据。

## 建议修复方向

### 短期

1. 在所有输出 JSON 中继续区分：
   - `difficulty_name`: 难度名，例如 expert/master/remaster
   - `difficulty_index`: 原始槽位，例如 4/5
   - `level_raw`: 原始等级文本，例如 `"14"` / `"?"`
   - `level`: 可解析数值，无法解析时为 null
2. 下游展示时不要把 `level=null` 显示成 unknown difficulty，建议显示为：
   - `difficulty_name=master`
   - `level=unknown`
3. 对 `missing_requested_difficulties` 使用 “missing remaster in source manifest” 这类文案，不要显示为 unknown difficulty。

### 中期

1. 如果项目需要 re:master，需要扩展 parser 的 `DIFFICULTY_RANGE` 和 difficulty mapping。
2. 明确 Maidata/Simai-like 数据中 re:master 的字段形式：
   - 是否是 `lv_6` / `inote_6`
   - 是否是其他命名字段
   - 是否在当前 raw data 中完全缺失
3. 扩展 `DIFFICULTY_NAMES`，例如：

```python
DIFFICULTY_NAMES = {
    1: "easy",
    2: "basic",
    3: "advanced",
    4: "expert",
    5: "master",
    6: "remaster",
}
```

同时补齐 parser、preprocess、tests。

### 长期

建立一个 difficulty schema 校验，明确禁止把以下概念混用：

| 概念 | 字段 | 示例 |
| --- | --- | --- |
| 难度槽位 | `difficulty_index` | 4, 5, 6 |
| 难度名 | `difficulty_name` | expert, master, remaster |
| 谱面等级原文 | `level_raw` | "13+", "14", "?" |
| 谱面等级数值 | `level` | 13.0, 14.0, null |
| 数据是否可训练 | `usable_for_training` | true/false |

## 本次定位结论

“unknown difficulty” 当前不是由训练 manifest 的 difficulty_name 产生的。

真实来源是：

1. `lv_5=?` 等不可解析等级造成的 `level_raw="?"` / `level=null`。
2. 当前 parser/manifest 不支持或未包含 re:master，导致 ID 批中请求的 remaster 全部缺失。
3. `tools/build_small_batch_manifest.py` 有一个未触发的 `difficulty_unknown` sample_id 兜底，未来如果输入 manifest 缺少 difficulty index 才会出现。
