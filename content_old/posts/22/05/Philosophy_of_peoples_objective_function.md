---
title: "關於人們的 objective function 的哲學"
date: 2022-05-24T00:04:40+08:00
draft: true
categories: "self"
mathjax: true
---

在做決策(還有評判別人的決策)時，我會把每件好或壞的結果用實數表示，好事是正的，壞事是負的。最佳解即為讓總和值最大。看了《正義 一場思辨之旅》後，我知道這種想法被稱作功利主義。學過 machine learning 後，我知道那個要最大化的東西叫 objective function。

$$a^*=\underset{a}{\mathrm{argmax}} O(a)$$

$$O(a)=\sum_{e\in{consequence(a)}} v(e)$$

Where:
$O$ 為 objective，$a$ 為決策，$e$ 為決策後造成的結果，$v$ 為事情的價值


假設所有結果都是可預估的，如果有


對，我就是在試圖量化(或至少用代數表示)這些抽象的價值。

## 公平
為何人們喜歡捐錢給窮人而不是捐給富人?為何劫富濟貧的故事往往令人讚賞?

因為對於一般的資源，其最佳分配方式接近平均分配。

人類的感覺是邊際效應遞減的，且常常是對數。例如一天
