---
title: "Reverb plugin"
date: 2022-03-16T17:33:37+08:00
draft: false

image: "https://i.imgur.com/v9uOdti.jpg"
categories: "music"
---

一直想做的東西後來真的有課教我怎麼做，就會覺得那個課很棒。這門課也是我這一年遇過最好玩的。

修課的十幾個學生分成6組，每組做一種效果器。我們這組做的是 reverb。Reverb 的功用是把音樂加上迴音，像是在大教堂或音樂廳的感覺。

這張圖是我們一開始參考的架構。

![Image](https://i.imgur.com/gLgWwXH.jpg#center)
https://ccrma.stanford.edu/~jos/pasp/Zita_Rev1.html

首先，輸入的 2 個 channel 被一個 2\*8 matrix 分成 8 個 channel，接著進入迴圈。迴圈裡都是 8 個 channel 在跑，訊號會依序經過 allpass、feedback matrix、lowpass、delay line 四個基礎 filter，然後重複。在 feedback matrix 後面有一條脫離迴圈的輸出路徑，會經過一個 8\*2 的 matrix，輸出到 2 個 channel。

這看似簡單(我們只要把每個基本模組實做出來即可)，但困難點在於