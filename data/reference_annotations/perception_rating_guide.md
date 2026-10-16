# Perception Rating 評分指南

給填寫本資料夾內 `perception_rating_sheet_<你的 annotator_id>_combined_<mode>.csv`（共 4 份，
`grounded` / `ablate_channel` / `ablate_translation` / `ablate_both`）的兩位評分者使用。

**這是 Tier 2（perception adequacy），跟 `annotator_guide.md` 講的 D1/D2/semiotic modes 標註
（Tier 3 的參考標註）是兩個不同的任務**，不要混在一起：

- `annotator_guide.md`：你自己看影片、自己判斷目標動作的 D1/D2/三個 semiotic modes——這是「參考
  標註」，之後拿去跟模型的判斷比較。
- 這份指南：你不用自己判斷 D1/D2，只需要**看模型寫的形式描述（`model_hands` 等七欄），評它準不準**
  ——也就是「模型有沒有看對它自己要評分的那個動作」。這個分數同時是 Tier 3 的門檻：模型描述不合格
  的項目，不會拿去比 D1/D2（見下方「這個分數的用途」）。

---

## 1. 你要看什麼影片

每一列是**某一支 clip 的某一個目標動作（event1 或 event2），在某個 cue 條件下的一次模型輸出**。
要判斷模型的描述準不準，去 `data/clips_c/` 資料夾裡找對應的影片，**用不帶 `_model` 後綴的版本**
（`_model.mp4` 是燒錄字幕模糊化後給模型看的版本，你不需要看那個模糊化的版本；沒有 `_model` 的是
原始未模糊化版本，給人類看的）：

| 這一列的 `event_id` | 主要看這個檔案（目標動作） | 需要判斷 `temporal_alignment`（動作跟前後手語的時間關係）時，可以再看這個檔案（前後文） |
|---|---|---|
| `event1` | `data/clips_c/<clip_id>_ev1.mp4` | `data/clips_c/<clip_id>_window.mp4` |
| `event2` | `data/clips_c/<clip_id>_ev2.mp4` | `data/clips_c/<clip_id>_window.mp4` |

`ev1.mp4` / `ev2.mp4` 已經是剪好、只留少量緩衝的目標動作片段，**整支片段基本上就是目標動作本身**，
不用再自己找時間點。`window.mp4` 是完整分析窗口，只在需要判斷「這個動作跟周圍手語的時間關係」時
才需要參考。

如果想同時確認翻譯是什麼（例如懷疑模型看錯人、看錯動作），可以對照 `clip_manifest.csv` 或你自己的
`annotation_sheet_<你的 annotator_id>.csv`（用 `clip_id` + `event_id` 對照），裡面有
`translation` / `target_start_sec` 等欄位。目標 channel 已經直接列在這份評分表的 `target_channel`
欄位裡，不用再去查（見第 2 節）。

本資料夾所有評分表都只針對 **`clip_kind = combined`** 這個條件（研究者已決定優先評這個條件，
`window`/`ev1`/`ev2` 單獨送的條件這輪不評），所以你看到的每一列，模型當時拿到的輸入就是
「`window.mp4` 當作前後文 + `ev1.mp4`/`ev2.mp4` 當作目標動作片段」一起送進去的那次推論結果。

---

## 2. 評分欄位

CSV 裡研究者已經填好、**不要改**的欄位：
`clip_id, event_id, clip_kind, run_index, target_channel, channel_judged, model_hands,
model_handshape, model_movement, model_location, model_nonmanual, model_non_dominant_hand,
model_temporal_alignment, rater_id`

你要填的兩欄：

- **`perception_score`（必填，0–2 整數）**：見下方判準。
- **`notes`（選填）**：想補充哪裡對／哪裡錯可以寫在這裡，尤其是給 0 或 1 分時，簡短寫一句原因方便
  之後查對（例如「描述的是聽者的點頭，不是手語者」）。

### Step 0（先做這步）：看 `target_channel`，決定這一列的「核心欄位」是哪個

`target_channel` 是 `clip_manifest.csv` 裡研究者指定的目標 articulator，四選一：
`manual`（手部）／`face`（臉部）／`head_torso`（頭與軀幹）／`non_dominant_hand`（非慣用手）。
**七個 `model_*` 欄位不是每一欄都同等重要——真正的內容應該集中在跟 `target_channel` 對應的那個
欄位，其餘欄位次要、甚至合理為 N/A。** 這是這份指南最重要的一條規則，先查這一列的
`target_channel`，再決定要嚴格檢查哪些欄位：

| `target_channel` | 核心欄位（必須準確，決定 0/1 vs 2 的分界） | 次要欄位（不夠精確不擋 2 分） |
|---|---|---|
| `manual` | `model_hands`、`model_handshape`、`model_movement`、`model_location` | `model_nonmanual`、`model_non_dominant_hand`、`model_temporal_alignment` |
| `non_dominant_hand` | `model_non_dominant_hand`（非慣用手在做的動作本身，不是慣用手） | 其餘欄位 |
| `face` | `model_nonmanual`（且內容應該是**臉部**表情/動作，不是頭或軀幹） | 其餘欄位（`model_hands`/`model_handshape` 此時常合理為 N/A 或「持續打手語」之類） |
| `head_torso` | `model_nonmanual`（且內容應該是**頭或軀幹**的動作，不是臉部表情） | 其餘欄位 |

**注意 `model_nonmanual` 這一欄同時承載 `face` 跟 `head_torso` 兩種 target_channel 的內容**（schema
只有一個欄位，沒有分開）。所以 `target_channel = face` 時，即使 `model_nonmanual` 寫得很詳細，
如果內容其實是在講點頭、聳肩這種頭/軀幹動作而不是表情，那還是**沒答對核心欄位**，不能給 2 分；
`target_channel = head_torso` 時同理反過來檢查。

### Step 1：核對 `channel_judged` 跟 `target_channel` 是否一致

`channel_judged` 是模型自己回報「我判斷這個動作是由哪個 articulator 承擔的」，跟 `target_channel`
（研究者指定的正確答案）用同一組四個值。**先比對這兩欄再看七個描述欄位**，因為這決定了模型的
形式描述整體上是不是在講對的東西：

- **`grounded` / `ablate_translation`（這兩個 mode 有把 target channel 告訴模型）**：這種情況下
  `channel_judged` 應該要等於 `target_channel`，因為答案已經給了。若兩者**不一致**，代表模型沒有
  遵照指定的 channel 作答（跟有沒有看懂畫面是兩回事，比較像是沒有照指示做），這種情況下七個描述
  欄位幾乎必然是在描述錯的動作/部位，**直接給 0 或 1，不要因為其中某幾欄湊巧寫得像那麼回事就給
  2 分**。在 `notes` 註明「channel_judged 與 target_channel 不符」方便之後篩選。
- **`ablate_channel` / `ablate_both`（這兩個 mode 沒有把 target channel 告訴模型，要模型自己判斷）**：
  這裡 `channel_judged` 本身就是模型的猜測，猜錯是預期中可能發生的失敗模式，不是「沒照指示做」。
  但對這一列的評分來說，**若 `channel_judged` 跟 `target_channel` 不符，七個描述欄位描述的多半也是
  模型自己選的那個動作，而不是研究者要問的那個目標動作**，所以同樣應該給低分（0 或 1）——差別只在
  `notes` 怎麼寫比較準確，例如寫「模型自己判斷成 face，但目標是 non_dominant_hand，屬於自行判斷
  channel 判斷錯誤，非指示不遵循」。

兩種情況下的**給分結果一樣（低分）**，差別只在 `notes` 怎麼歸因，因為兩者對應到論文裡不同的討論
（指示遵循 vs. 模型自行定位 channel 的能力）。

### 判準

**先想清楚這個分數在問什麼：不是「這份描述寫得完不完美」，而是「這份描述準到我敢不敢拿它去信任
模型接下來給的 D1/D2 跟三個 semiotic modes 分數」。** 2 分不代表七欄全對、無可挑剔，只代表**核心
欄位（依上表，隨 `target_channel` 而定）**都對，剩下的分數才有意義去比較；如果核心欄位就錯了，
後面的功能判斷等於是在評論一個他沒看對的動作，這種項目就不該讓 D1/D2 進入比較。

用**核心欄位對不對、加上其餘欄位大致答對幾成**當作給分的具體依據，不要純憑印象：

| Score | 判準 |
|---|---|
| 0 | 核心欄位錯誤或與畫面無關，且其餘欄位也大多不對；看不出模型抓到了哪個動作 |
| 1 | 核心欄位部分正確、或核心欄位對但其餘欄位大多錯，整體不足以放心採信 |
| 2 | 核心欄位都對，足以支撐後續的功能歸屬（D1/D2、semiotic modes）判斷；次要欄位的用詞差異或不夠精確不影響給 2 分 |

換句話說：**核心欄位是否正確，決定 0/1 vs 2 的分界；次要欄位寫得不夠細只會讓你在 0 跟 1 之間猶豫，
不會擋掉 2 分。** 如果核心欄位有明顯錯的（例如寫錯人、寫錯方向、或 channel_judged 跟 target_channel
不符——見下方「常見扣分點」），即使其他欄位寫得還不錯，也不要因此放水給 2 分，因為那類錯誤代表
模型根本沒看對目標動作本身。

一列只給**一個**總分，不用逐欄各給一個分數。

### 逐欄檢查表

給分前，對照影片逐項檢查模型這七欄寫得準不準（先看 Step 0 的表，知道哪幾欄是這一列的核心欄位）：

| 欄位 | 要檢查什麼 |
|---|---|
| `model_hands` | 哪隻手做目標動作；若 `target_channel` 不是 `manual`，這欄常合理為 N/A 或簡短帶過 |
| `model_handshape` | 手指／手掌的具體形狀是否吻合；非手部動作寫 N/A 是否合理 |
| `model_movement` | 移動軌跡與方式（直/弧、單次/重複、快/慢）是否吻合 |
| `model_location` | 動作發生在身體/空間哪個位置是否吻合；臉部動作是否指對臉部區域 |
| `model_nonmanual` | 手部以外的臉／頭／軀幹活動描述是否吻合；`target_channel = face` 或 `head_torso` 時這欄是核心，且要對到正確的部位（臉 vs. 頭/軀幹，見 Step 0） |
| `model_non_dominant_hand` | 非慣用手在做什麼（放鬆／維持手形／與慣用手對稱／或本身就是目標動作）是否吻合 |
| `model_temporal_alignment` | 動作跟周圍手語的時間關係描述是否吻合（需要時參考 `window.mp4`） |

### 常見扣分點，特別留意

- **`channel_judged` 跟 `target_channel` 不符**：見上方 Step 1，這種情況幾乎必然拖垮核心欄位的
  準確度，直接視為描述錯誤。
- **描述的是另一個人**：訪談片段中，模型可能把聽者的點頭/表情寫進描述，而不是主要手語者的動作
  （誰是主要手語者，見 `annotator_guide.md` §1.2：目標時間範圍內實際在產出這段話語的那位）。
- **方向以鏡頭為準而非手語者為準**：模型 prompt 有要求一律用手語者自己的左右（見
  `annotator_guide.md` §1.3），若描述明顯搞反，算描述錯誤。

三者都算描述錯誤，不要因為「其他欄位寫得不錯」就給高分。

---

## 3. 獨立評分（務必遵守）

**兩位評分者要各自獨立評分，填完存檔之前不可以先討論分數或互看對方的表**，之後才會對這份表算
inter-rater agreement（`12_compute_annotator_agreement.py`）。填完之後再互相比較、討論分歧，分歧
本身也是有價值的資訊。

四份檔案（`grounded` / `ablate_channel` / `ablate_translation` / `ablate_both`）可以照任何順序評，
但同一個 `clip_id` + `event_id` 會在四份檔案裡各出現一次、對應模型在四種不同 cue 條件下寫的描述
（可能很像、也可能差很多）。**每一列都只看那一列寫的內容單獨評分**，不用刻意記住你在其他檔案裡
給過的分數、也不用刻意求一致——如果模型在不同條件下寫得品質本來就不一樣，分數本來就該不一樣。

---

## 4. 這個分數的用途

`perception_score` 除了算 inter-rater agreement，也是 Tier 3（`10b_score_attribution_events.py`）
比較模型 D1/D2/semiotic modes 準確度時的**門檻**：只有 `perception_score`（兩位評分者平均）達到
`pipeline_config.yaml` 裡 `attribution_scoring.perception_adequate_min_score`（目前設定為 2）的
項目，才會拿去跟人類參考標註比 D1/D2——這樣「模型根本沒看對動作」跟「模型看對了但功能判斷錯」才
不會混在一起算成同一種錯誤。所以**評分要對照影片實際核對，不要用猜的或大致掃過**，這個分數的品質
直接決定 Tier 3 結果可不可信。
