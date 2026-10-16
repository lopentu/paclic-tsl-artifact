# Uncapped 重評指南（補充，只有 45 列）

給填寫 `uncapped_rating_sheet_A01.csv` / `uncapped_rating_sheet_A02.csv` 的兩位評分者使用。

**這是 `perception_rating_guide.md` 的補充，不是取代**。影片怎麼找、七個 `model_*` 欄位各是什麼
意思、0/1/2 的整體判準，全部沿用原本那份指南。這份只講**這一輪跟原本規則不一樣的地方**，共三點。

## 為什麼要重評這 45 列

原本的評分規則有一條 Step 1：模型自報的 articulator（`channel_judged`）跟研究者指定的
（`target_channel`）不符時，直接給 0 或 1 分。這條規則在「有告訴模型目標 articulator」的兩個
條件下沒問題，但在「沒告訴模型、要模型自己判斷」的 `ablate_channel` / `ablate_both` 兩個條件下，
它讓我們的一個結論變成**循環論證**：我們想說「模型表現差是因為它選錯 articulator」，但選錯就自動
被扣分，所以這句話有一部分是規則自己造成的，不是資料告訴我們的。

要拆開這兩件事，需要知道：**在那些選錯 articulator 的列裡，模型的描述本身到底準不準？** 也就是
不管它描述的是哪個 articulator，那段描述作為「對畫面的描述」是否正確。這 45 列就是所有選錯的列
（`ablate_channel` 22 列 + `ablate_both` 23 列），需要在**暫停 Step 1** 的情況下重新評一次。

這是這一輪唯一需要新增人工評分的項目。

## 三個改變

### 1. Step 1 停用——不要因為 articulator 選錯就給低分

原本「`channel_judged` 跟 `target_channel` 不符 → 直接 0 或 1 分」這條，**這一輪完全不適用**。
這 45 列**全部都是**不符的列，如果照原規則評，45 列會全部是低分，等於什麼都沒問到。

這一輪要問的是：**這段描述，作為對它自己所描述的那個 articulator 的描述，準不準？** 描述得準就
可以給 2 分，即使它描述的不是研究者原本想問的那個 articulator。

### 2. 「核心欄位」改看 `channel_judged`，不再看 `target_channel`

原本指南 Step 0 的表是用 `target_channel` 決定哪幾個 `model_*` 欄位是核心欄位。這一輪改用
**`channel_judged`** 查同一張表（兩欄的四個值完全相同：`manual` / `face` / `head_torso` /
`non_dominant_hand`）。

這 45 列裡 43 列的 `channel_judged` 是 `manual`、2 列是 `face`，所以實際上多數列的核心欄位是
`model_hands` / `model_handshape` / `model_movement` / `model_location`。

### 3. 評分表上沒有 `target_channel` 這一欄——這是故意的，請不要去查

原本指南說「目標 channel 已經直接列在評分表的 `target_channel` 欄位裡」，這一輪**刻意移除**，
因為看到它就很難不順手套回 Step 1，那樣重評就白做了。

`clip_manifest.csv` 和你自己的 `annotation_sheet_<id>.csv` 裡都還有這個欄位，**這一輪請不要去對照
它**。要確認翻譯內容、或想看模型是不是看錯人，一樣可以查 `clip_manifest.csv` 的 `translation`
欄位，只是不要順便看目標 channel。研究者事後會自己把 `target_channel` 接回來做分析，你不需要知道。

## 要填的欄位

| 欄位 | 填法 |
|---|---|
| `uncapped_score` | **必填**，0 / 1 / 2 整數。判準沿用原指南，只是照上面三點調整 |
| `notes` | 選填。給 0 或 1 分時簡短寫一句原因會很有幫助 |

其他欄位（含已經幫你填好的 `rater_id`）都不要改。

**欄位名字是 `uncapped_score` 不是 `perception_score`**，這樣才不會跟你上一輪填的分數混在一起，
兩輪的分數之後要拿來互相比較。

## 兩件跟上一輪相同的事

- **兩位評分者各自獨立評分**，填完存檔之前不可以先討論或互看，之後要算 inter-rater agreement。
- **同一個 `clip_id` + `event_id` 會出現兩次**（`mode` 欄位一個是 `ablate_channel`、一個是
  `ablate_both`），對應模型在兩種不同 cue 條件下寫的描述，內容不一樣。**每一列只看那一列的內容
  單獨評分**，不用刻意跟另一列一致。檔案裡兩個 mode 是分開成兩區塊的，所以同一個 event 的兩列
  不會相鄰。

不用回頭去翻你上一輪給這些列打了幾分，也不要刻意避開——就當成新的一輪來評。
