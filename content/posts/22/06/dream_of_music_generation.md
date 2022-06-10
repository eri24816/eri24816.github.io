---
title: "生成音樂的夢"
date: 2022-06-10T23:34:05+08:00
draft: false
image: "https://i.imgur.com/64Dk0jO.png"
categories: music
---
![Image](https://i.imgur.com/64Dk0jO.png#center)

夢的意思是我根本不知道做不做得出來。

## 音樂像文字，還是圖像?

在這個 transformer 當道的時代，大家紛紛拿 transformer-like 的 language model 來生成音樂，所以音樂被視為字詞序列，所有音符、時間和其他資訊攤都被平成一串 token。例如 [Pop Music Transformer](https://arxiv.org/abs/2002.00212)。

巨觀來看，音樂像自然語言沒錯，情緒會在語句間、章節間漂浮；但若從微觀來看，音樂比較像圖像，有和弦、上下行、琶音等 pattern，這點從樂譜，甚至頻譜上能很明顯地看出來。

所以我給的答案是，音樂在低階 feature 處應該當圖像看待，而高階 feature 處應該當文字。

我想以和弦進行為分界，和弦進行以上 (1 bar ~ 整首歌) 的尺度以 transformer (或其他 auto-regressive model) 處理，而和弦以下 (note/frame ~ 1 bar) 的尺度主要以 CNN 處理。而整個神經網路模型結構就像圖像處理模型一樣，漸進式地擴大能感知到的 feature 尺度:

1. 音符
2. 和聲
3. 旋律片段/上下行等結構/和弦在時間上的 pattern
4. 和弦進行
5. 情緒變化
6. 整首音樂的 embedding

這樣做相對於純 transformer 有什麼好處?
1. CNN 比較會乖乖學固定的幾種 pattern，而音樂的低階 pattern 滿固定的 (因為樂理)
2. Transformer 為了不讓計算量太高，限制了視野距離。針對於此，TransformerXL 在常數級別增加了 transformer 視野距離。而我使用的時間尺度漸進式架構，則能以指數級別增加視野距離。

## 我想的作法

高三的時候我因為考上大學很閒，就在想這件事了。我當時想出了模型架構:

![Image](https://i.imgur.com/qeJNR2c.png#center)

整個模型 由 3 層 VAE[^1] 配上 4 個層次的 feature 組成。
[^1]: 3 層 VAE 這件事剛好有點像 [OpenAI Jukebox](https://openai.com/blog/jukebox/)

![Image](https://i.imgur.com/FKMW1YZ.png#center)

先訓練最底下的 E1、D1，訓練完後把 dataset 用 E1 轉上來，變成較抽象的 z1，用來 train E2、D2。然後再用同樣的方法 train E3、D3。(我是沒這樣 train 過東西啦，說不定根本 train 不起來，but who knows)


![Image](https://i.imgur.com/KZxxhKn.png#center)

Inference 時，把三個 encoder 或 decoder 接在一起就可以對一首音樂 encode 或 decode 了。

如果想生成音樂，就用 decoder。

那 encoder 可以幹嘛?我最期待的應用就是把它當 denoising AE 用。我彈電子琴總是彈得零零落落，如果這個模型訓練好了，就可以把我彈的東西 encode 進去再 decode 出來，變成美妙的音樂 UwU

---
高三的時候雖然想好怎做了，但因為 dataset 蒐集不起來而沒有繼續做。

事實上當時我為了投稿旺宏科學獎，把內部細節都設計了一遍。那時我還不懂 transformer，所以裡面都是 RNN 和 CNN。

![Image](https://i.imgur.com/Qt36gbr.png#center)

現在趁著能到中研院實習，正要重拾這項計畫。我會把 transformer 加進去，transformer 一定會很有幫助。感謝至斌學長教我 transformer。