# 客戶流失預測 (Customer Churn Prediction) - 結果評估
# Customer Churn Prediction - Result Evaluation

本專案使用了隨機森林（Random Forest Classifier）來預測客戶是否會流失（目標變數 `Exited`）。以下是模型在驗證集（Validation Set）上的表現與特徵分析。

This project utilizes a **Random Forest Classifier** to predict whether a customer will churn (target variable: `Exited`). The following sections present the model's performance on the **Validation Set** along with feature analysis.

---

## 1. 混淆矩陣 (Confusion Matrix)

混淆矩陣幫助我們具體了解模型「猜對」與「猜錯」的實際數量分佈。在此驗證集中，總資料筆數為 33,007 筆。

The Confusion Matrix helps us understand the exact distribution of correct and incorrect predictions made by the model. The validation set contains a total of **33,007 records**.

根據產出的熱力圖，四個象限的數值意義如下：

Based on the generated heatmap, the four quadrants are interpreted as follows:

| 象限 Quadrant |
|---|---|---|
| **True Negative (TN) — 左上 / Top-Left: 24,775** | 實際**未流失 (0)**，模型正確預測**未流失 (0)** | Actual **Not Churned (0)**, correctly predicted as **Not Churned (0)** |
| **False Positive (FP) — 右上 / Top-Right: 1,248** | 實際**未流失 (0)**，但被誤判為**流失 (1)** | Actual **Not Churned (0)**, incorrectly predicted as **Churned (1)** |
| **False Negative (FN) — 左下 / Bottom-Left: 3,389** | 實際**流失 (1)**，但被漏判為**未流失 (0)** | Actual **Churned (1)**, incorrectly predicted as **Not Churned (0)** |
| **True Positive (TP) — 右下 / Bottom-Right: 3,595** | 實際**流失 (1)**，模型正確辨識**流失 (1)** | Actual **Churned (1)**, correctly predicted as **Churned (1)** |

### 模型評估指標計算 / Evaluation Metrics

利用上述數值，可計算出以下模型評估指標：

The following metrics are derived from the values above:

$$
\text{準確率 (Accuracy)} = \frac{TP + TN}{Total} = \frac{3595 + 24775}{33007} \approx 85.95\%
$$

$$
\text{精確率 (Precision)} = \frac{TP}{TP + FP} = \frac{3595}{3595 + 1248} \approx 74.23\%
$$

$$
\text{召回率 (Recall)} = \frac{TP}{TP + FN} = \frac{3595}{3595 + 3389} \approx 51.47\%
$$

####  分析總結 / Insight Summary

模型的整體準確率相當高（約 86%），特別是對於「不會流失的客戶」辨識能力極強。然而，在「真正會流失的客戶」中，模型只成功抓出約一半（召回率 51.47%），漏掉了 3,389 人。不過，一旦模型發出警告，有高達 74% 的機率是準確的（精確率 74.23%），意味著行銷資源可以精準地投放在模型標記的危險名單上。
>
The model achieves a high overall accuracy (\~86%), with particularly strong capability in identifying customers who will **not** churn. However, among customers who **actually do churn**, the model only captures roughly half of them (Recall: 51.47%), missing 3,389 individuals. On the flip side, whenever the model flags a customer as at-risk, it is correct **74% of the time** (Precision: 74.23%) — meaning marketing resources can be deployed precisely and efficiently toward the flagged high-risk list.

---

## 2. 接收者操作特徵曲線 (ROC Curve)

ROC 曲線是評估二元分類模型整體效能的重要視覺化工具，展示了在不同判定閾值下，**真陽性率 (True Positive Rate)** 與 **偽陽性率 (False Positive Rate)** 之間的權衡關係。

The ROC Curve is a key visualization tool for evaluating the overall performance of a binary classification model. It illustrates the trade-off between the **True Positive Rate (Recall)** and the **False Positive Rate** across different decision thresholds.

| 圖例 Legend |
|---|---|
|  **藍色虛線 / Blue Dashed Line** | 隨機猜測的基準線（AUC = 0.5）/ Random guessing baseline (AUC = 0.5) |
|  **橘色實線 / Orange Solid Line** | 本模型的實際表現 / Actual performance of our trained model |
|  **AUC 分數 / AUC Score** | **0.88** |

####  總結 / Summary

AUC 分數 0.88 屬於**非常優異**的成績（通常大於 0.8 即代表良好模型）。這意味著當隨機挑選一名「會流失」與一名「不會流失」的客戶時，模型有 **88% 的機率**能正確排序其流失風險，證明模型具備強大的區分能力。
>
An AUC score of **0.88** is considered **excellent** (a score above 0.8 generally indicates a strong model). This means that if we randomly pick one churned customer and one non-churned customer, the model has an **88% chance** of correctly ranking the churned customer as higher risk — demonstrating robust discriminative power rather than mere guessing.

---

## 3. 前 10 大特徵重要性 (Top 10 Feature Importances)

此長條圖顯示了隨機森林在決策時最依賴的 10 個變數。數值越高，代表該變數對預測客戶是否流失的影響力越大。

This bar chart displays the **10 most influential variables** used by the Random Forest during decision-making. A higher value indicates greater importance in predicting customer churn.

### 排行榜解析 / Feature Ranking Breakdown

| 排名 Rank | 特徵 Feature | 說明 | Description |
|:---:|---|---|---|
| 1 | `Age` 年齡 | 最關鍵特徵，不同年齡層客戶的流失傾向差異極大 | Most critical feature; churn tendency varies significantly across age groups |
| 2 | `NumOfProducts` 使用產品數量 | 持有產品數量是決定去留的第二大關鍵 | Number of products held is the second most decisive factor |
| 3 | `EstimatedSalary` 預估薪資 | 客戶的經濟能力對流失行為有顯著影響 | Customer's financial capacity significantly influences churn behavior |
| 4 | `id` 客戶編號 | 不應具備預測力，可能暗含時序或地區規律 | Should not carry predictive power; may encode hidden temporal/regional patterns (see diagnostic note below) |
| 5 | `CreditScore` 信用分數 | 信用狀況對流失風險有相當的貢獻 | Credit status contributes considerably to churn risk |
| 6 | `Balance` 帳戶餘額 | 帳戶資金多寡同樣影響客戶忠誠度 | Account balance level also affects customer loyalty |
| 7 | `Tenure` 往來年資 | 與公司關係的時間長短具顯著影響力 | Duration of the customer relationship with the company is notably impactful |
| 8 | `IsActiveMember` 活躍會員 | 是否為活躍會員反映客戶黏著度 | Active membership status reflects customer engagement |
| 9 | `Geography_Germany` 地區（德國） | 德國客戶有顯著不同的流失模式 | Customers in Germany show a notably different churn pattern |
| 10 | `Gender_Female` 性別（女性） | 性別因素在流失預測中亦有一定角色 | Gender plays a modest yet measurable role in churn prediction |

**模型診斷 / Model Diagnostic Note**
>
`id` 理應是純流水號，不具備任何商業預測意義。它出現在重要性排行中，可能是因為資料集的 `id` 分配隱含了時間順序或地區規律。建議在未來的模型優化中**剔除此欄位**，以避免**過度擬合 (Overfitting)** 或**資料洩漏 (Data Leakage)**。
>
 `id` should theoretically be a meaningless sequential identifier with no business predictive value. Its appearance in the importance ranking suggests it may encode hidden patterns such as temporal ordering or regional groupings within the dataset. It is strongly recommended to **drop this column** in future model iterations to prevent **Overfitting** or **Data Leakage**.

####  商業意涵總結 / Business Insight Summary

基於此特徵報告，企業應優先針對「特定年齡層」且「產品持有數較低」的客戶群體，設計挽留行銷活動（例如發送專屬優惠或提升會員活躍度），這將是降低客戶流失率最有效的切入點。
>
Based on this feature report, businesses should prioritize designing **retention campaigns** targeting customers of **specific age groups** with a **low number of held products** — for example, sending personalized offers or incentivizing higher membership engagement. This represents the most effective lever for reducing overall customer churn.

---
