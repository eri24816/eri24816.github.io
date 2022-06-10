---
title: "Reverb plugin"
date: 2022-03-16T17:33:37+08:00
draft: false

image: "https://i.imgur.com/v9uOdti.jpg"
categories: "music"
tags: ["c++"]
MathJax : true
---
一直想做的東西後來真的有課教我怎麼做，就會覺得那個課很棒。

在「數位音樂訊號分析」這門課中，十幾個學生分成6組，每組使用 JUCE(一個製作 VST 的 C++ 框架)來做一種 VST 效果器。我們這組做的是 reverb。Reverb 的功用是把音樂加上迴音，像是在大教堂或音樂廳的感覺。

這門課是我這一年遇過最好玩的。我學到非常多好玩的觀念，像是 z transform、怎麼看 zero pole plot、如何用 C++ 來 OOP (踩各種指標的坑XD)。


## 整體架構
下圖是我們一開始參考的架構。

![Image](https://i.imgur.com/gLgWwXH.jpg#center)
https://ccrma.stanford.edu/~jos/pasp/Zita_Rev1.html

首先，輸入的 2 個 channel 被一個 2\*8 matrix 分成 8 個 channel，接著進入迴圈。迴圈裡都是 8 個 channel 在跑，訊號會依序經過 allpass、feedback matrix、lowpass、delay line 四個基礎 filter，然後重複。在 feedback matrix 後面有一條脫離迴圈的輸出路徑，會經過一個 8\*2 的 matrix，輸出到 2 個 channel。

這看似簡單(我們只要把每個基本模組實做出來即可)，但困難點在於這兩項性質難以同時兼顧:

1. 產生的聲音好聽
2. IIR的穩定性

這是因為我們使用的是帶有迴圈的 IIR，所以參數沒調好的話會不穩定 (unstable)。如果使用的是 FIR 則能免除這項困難，但是如果用 FIR 做 reverb，計算量會太大。所以在建好架構後，我們花了大半的時間在嘗試找出在 IIR 穩定的情況下，聲音最好聽的參數。

## 各模組的實作

在我們的實作中有很多種模組 (例如 DelayLine，LowPass)，每種模組都是一個多 channel 的 causal filter。一個大模組裡面可能包含數個小模組，而完整的 Reverb 本身就是一個最大的模組。

每種模組皆有實作方法 

```c++
float* update(float* input)
```
在 VST 的每一個 time step，這個方法都會收到一個 float 陣列作為輸入，計算完後輸出一個 float 陣列。大部分的模組都具有 memory，所以 update() 方法不是 time independent 的。

### DelayLine
DelayLine 的功能單純就是在收到訊號後延遲數個 sample 的時間再輸出，在頻域上的作用為$z^{-n}$。用 std::queue 就可以簡單地把它實做出來。
```c++
float* update(float* input) override{
    for (int i = 0; i < inputDim; i++) {
        queues[i].push(input[i]);
        outputBuffer[i] = queues[i].front();
        queues[i].pop();
    }
    return outputBuffer;
}
```
後來為了減少耗時，我又自己刻了一個沒有用到 std::queue 的版本。
```c++
float* update(float* input) override {
    for (int i = 0; i < inputDim; i++) {
        outputBuffer[i] = arr[i][pos[i]];
        arr[i][pos[i]] = input[i];
        pos[i]++;
        if (pos[i] == len[i])pos[i] = 0;
    }
    return outputBuffer;
}
```
### Lowpass
這個模組是一階 low pass filter，頻域上的作用為$(1-a)z+az^{-1}$。

實作的方法是把前一個輸出以某個權重加回當前的輸入，以此作為輸出。也就是在波型上做 smoothing。
```c++
float* update(float* input) override {
    mult(inputDim, input, 1 - a);
    mult(inputDim, feedback, a);
    add(inputDim, input, feedback);
    copy(inputDim, feedback, input);
    return mult(inputDim, input,1);
}
// 這些 code 長得很像 assembly 是因為我沒有定義 vector 這個 class 和它的 operators，
// 只有草率定義了幾個對 float 陣列的運算。我當時太懶了XD
```

a 是調控 cutoff frequency 的參數，它們的關係是: 
$$a=e^{-2\pi \frac{ \mathit{Cutoff}} {\mathit{SampleRate}}}$$

### Allpass

它是二階的 all pass filter。這是這個 project 最難做的 filter，我們必須由該 filter 應具有的性質，推導出實作的方法。

二階 all pass filter 的 pole-zero plot 有兩個 poles 和兩個 zeros，且此 filter 的性質受到以下限制:
1. 根據 complex conjugate root theorem，兩個 poles(zeros) 必共軛
2. 為了讓各頻率的 amplitude response 皆維持在 1，每個 pole 對單位圓的反演處必須有一個 zero，反之亦然

結合這兩項限制，代表我們只需要一組參數 $(r,θ)$ 來控制某一個 pole 的位置，即可以決定所有 pole 和 zero 的位置，也決定了這個二階 filter。

![Image](https://i.imgur.com/4HVI7Xu.png#centers)

接下來就是用 $(r,θ)$ 來推出 IIR 的結構。

設此 filter 的 response 為 $\frac{P(z)}{Q(z)}$，$P$ 的兩根為 zero，$Q$ 的兩根為 pole。那麼:[^1]

$$\begin{aligned}
P(z) &=(z-r^{-1}e^{iθ}) (z-r^{-1}e^{-iθ})  \\\
&=z^2-r^{-1}(e^{iθ}+e^{-iθ})z+r^{-2}\\\
&=z^2-2r^{-1}\cos(θ)z+r^{-2}
\end{aligned}$$


$$\begin{aligned}
Q(z)&=(z-re^{iθ})(z-re^{-iθ})\\\
&=z^2-r(e^{iθ}+e^{-iθ})z+r^2
\end{aligned}$$

### Comb

## 調整參數


[^1]: 國中學了但從未用過的根與係數終於在這裡用到了，覺得國中很浪費時間的心情稍稍下降了一點。