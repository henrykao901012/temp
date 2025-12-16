# 方案 B 訓練結果分析報告 v1

## 訓練數據總覽

### Loss 曲線
```
Epoch 01 | outcome_loss=0.4481 | conf_loss=0.0720
Epoch 05 | outcome_loss=0.4047 | conf_loss=0.0584
Epoch 10 | outcome_loss=0.4028 | conf_loss=0.0579
Epoch 15 | outcome_loss=0.4009 | conf_loss=0.0572
Epoch 20 | outcome_loss=0.4002 | conf_loss=0.0569
Epoch 25 | outcome_loss=0.3992 | conf_loss=0.0567
Epoch 30 | outcome_loss=0.3989 | conf_loss=0.0568
Epoch 35 | outcome_loss=0.3984 | conf_loss=0.0565
Epoch 40 | outcome_loss=0.3972 | conf_loss=0.0560
Epoch 45 | outcome_loss=0.3970 | conf_loss=0.0560
Epoch 50 | outcome_loss=0.3967 | conf_loss=0.0558
```

### Outcome Head 指標
```
Outcome accuracy: 0.9116143592909087
```

### Event Detection 指標
```
Event precision: 0.37938274196934707
Event recall   : 0.537797619047619
Direction acc | true event: 0.5160714285714286
```

### 5×5 Confusion Matrix (counts)
```
pred         strong_down  weak_down   flat  weak_up  strong_up
true                                                          
strong_down          728        278     35       18         23
weak_down            453        266     11        0          0
flat                2969      11027  12468    14144       7857
weak_up                0          3     17      214        328
strong_up             31         33     40      317        581
```

### 5×5 Confusion Matrix (row-normalized)
```
pred         strong_down  weak_down    flat  weak_up  strong_up
true                                                           
strong_down       0.6728     0.2569  0.0323   0.0166     0.0213
weak_down         0.6205     0.3644  0.0151   0.0000     0.0000
flat              0.0613     0.2275  0.2573   0.2918     0.1621
weak_up           0.0000     0.0053  0.0302   0.3808     0.5836
strong_up         0.0309     0.0329  0.0399   0.3164     0.5798
```

### Ranking / Precision@Top-K
```
Base event rate: 0.06481356455315292

Top- 100 | Event Precision=0.000
Top- 500 | Event Precision=0.000
Top-1000 | Event Precision=0.067
```

---

## 一、先下總結（很重要）

**不是模型沒學到東西，而是：**

- **Outcome head** 在「避免誤報」
- **Confidence head** 沒有學會「事件排序」

這會導致一個典型現象：
- ✅ overall accuracy 很高
- ✅ confusion matrix 看起來「有結構」
- ❌ 但 **Precision@Top-K 幾乎為 0**

> **這不是 bug，是目標函數與評估目標不對齊。**

---

## 二、逐項解讀你的結果

### 1️⃣ Outcome accuracy = 0.91（⚠️ 不能被騙）
```
Outcome accuracy: 0.9116
```

這個數字幾乎沒有意義，原因你其實已經知道：
- Test event rate ≈ **6.48%**
- 只要模型偏向預測 `no_event`
- accuracy 就會自然很高

👉 **這裡不該再關注 accuracy**，而是：
- Event precision / recall
- Ranking 能不能抓到 event

---

### 2️⃣ Event detection：偏保守
```
Event precision: 0.379
Event recall   : 0.538
```

**解讀：**
- 模型抓到一半左右的事件（recall 尚可）
- 但只要一預測 event，約 **62% 是誤報**

👉 這其實是一個「**合理但保守**」的 outcome head  
（class weight + no_event dominance 的結果）

**Outcome head 本身不是主要問題。**

---

### 3️⃣ Direction acc | true event ≈ 0.52
```
Direction acc | true event: 0.516
```

這代表：
- 在真的發生事件時
- up / down 幾乎是**擲硬幣**

👉 這其實是符合 **Triple Barrier 的本質**：

> 在 hit 發生前，方向常常是高噪音的  
> TB 本來就不是 direction forecasting 問題

**這一點我不會拿來否定模型。**

---

## 三、真正的問題在這裡（關鍵）

### ❌ Precision@Top-K 幾乎失效
```
Base event rate: 0.0648

Top-100  | Event Precision = 0.000
Top-500  | Event Precision = 0.000
Top-1000 | Event Precision = 0.067
```

這是非常明確的訊號：

> **pred_conf 的排序，和「事件是否真的發生」幾乎無關**

也就是說：
- confidence head **沒有學會「哪個樣本更可能 hit」**
- ranking 只是在排序 **noise**

---

## 四、但 5×5 Confusion Matrix 告訴你一個好消息

你這個 5×5，其實不是爛，是「**半成功**」。

### ✅ 好的地方（請注意）

- **strong_down / strong_up**：
  - 對角線占比 ~58–67%
  - 幾乎不會跳到反方向 👉 **方向結構是對的**
- **weak ↔ strong 的錯誤是「鄰近桶」**  
  👉 表示模型學到「**強度連續性**」

### ⚠️ 真正的問題

看這一行：
```
flat → strong_up / strong_down 比例不低
flat 行分布: 0.0613, 0.2275, 0.2573, 0.2918, 0.1621
```

而且：
- **flat row 幾乎平均灑到各個 bucket**
- 代表 pred_conf 在 **no_event 上也會給很大的 |value|**

👉 這直接解釋了為什麼：
- Top-K 會選到一堆 **no_event**
- Precision@Top-K 會接近 **0**

---

## 五、根本原因（這句話很重要）

> 你現在的 **confidence head 是在「回歸 TB 值」**，  
> 而不是在「**為事件排序**」。

即使你是 hit-only loss：
```python
loss_conf = SmoothL1(pred_conf[hit_mask], y_conf[hit_mask])
```

但模型仍然可以：
- 在 **no_event 樣本上亂輸出大數值**
- 反正這些樣本**不進 loss**

👉 **排序目標沒有被約束**

---

## 六、這不是你做錯，而是「少一個必要的約束」

你現在的設計缺了這個概念：

> **no_event 的 confidence 應該被壓到 0 附近**

**沒有這個約束，ranking 一定爛。**

---

## 七、Loss 曲線的補充解讀

### 觀察：

1. **outcome_loss: 0.448 → 0.397**
   - 緩慢下降 → ✅ 正常學習
   - 沒有 collapse → ✅ 沒有作弊

2. **conf_loss: 0.072 → 0.056**
   - 快速收斂後穩定 → ⚠️ 表示：
     - 在 hit 樣本上，模型已經學會「回歸 TB 值」
     - 但這不等於「學會排序」
     - 而且 no_event 樣本**完全不受約束**

---

## 八、診斷結論

| 項目 | 狀態 | 評價 |
|------|------|------|
| Outcome head | ✅ 保守但合理 | 不是主要問題 |
| Direction structure | ✅ 方向不亂 | 學到鄰近結構 |
| Conf 回歸能力 | ✅ hit 樣本上可回歸 | loss 已收斂 |
| **Conf 排序能力** | ❌ **完全失效** | **no_event 未約束** |
| Precision@Top-K | ❌ 接近 0 | **無法用於交易** |

---

## 九、問題根源的一句話總結

> **你的 confidence head 在「回歸」，但市場需要的是「排序」。**  
> **而排序的前提是：no_event 必須被壓到低 confidence。**

這不是調參能解決的，是**目標函數設計的結構性問題**。

---

## 十、下一步建議（核心）

### 必須做的事：

1. **加入 no_event 約束**
```python
   # 方案 1: 對 no_event 也算 loss
   loss_no_event = pred_conf[no_event_mask]^2  # 壓向 0
   
   # 方案 2: Ranking loss (pairwise)
   # 保證 event 的 |conf| > no_event 的 |conf|
```

2. **改用 Ranking-aware 的 loss**
   - 例如：ListNet, LambdaRank, Contrastive loss
   - 或至少加入「event vs no_event」的對比項

### 不要做的事：

- ❌ 繼續調 class_weight
- ❌ 增加 conf_loss 權重
- ❌ 換更大的模型

**這些都不會解決「no_event confidence 亂飛」的問題。**

---

## 總結

這一版的本質：

- ✅ **模型有學習能力**（loss 下降、結構合理）
- ✅ **Outcome head 可用**（保守但不致命）
- ❌ **Conf head 目標錯位**（回歸 ≠ 排序）
- ❌ **無法用於交易**（Top-K 全是 noise）

**一句話：**

> 這是一個「訓練成功、但目標錯誤」的模型。  
> 不是沒學到，而是**學到了你沒要它學的東西**。

**修正方向：**

不是「調」，而是「改目標函數」—— 必須讓 no_event 的 confidence 被明確約束。