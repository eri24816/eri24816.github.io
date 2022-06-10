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
這個模組是一階 low pass filter，頻域上的作用為$(1-a)+az^{-1}$。

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

設此 filter 的 response 為 $H(z)=\frac{P(z)}{Q(z)}$，$P$ 的兩根為 zero，$Q$ 的兩根為 pole。以此為出發點，P 和 Q 為:[^1][^2]
[^1]: 國中學了但從未用過的根與係數終於在這裡用到了，覺得國中很浪費時間的心情稍稍下降了一點。
[^2]:第一行的 $r^2$ 用於 normalize

$$\begin{aligned}
P(z) &= r^2(z-r^{-1}e^{iθ}) (z-r^{-1}e^{-iθ})  \\\\
&=r^2z^2-r(e^{iθ}+e^{-iθ})z+1\\\\
&=r^2z^2-2r\cos(θ)z+1
\end{aligned}$$

$$\begin{aligned}
Q(z)&=(z-re^{iθ})(z-re^{-iθ})\\\\
&=z^2-r(e^{iθ}+e^{-iθ})z+r^2\\\\
&=z^2-2r\cos(θ)z+r^2
\end{aligned}$$

令$X(z)$為輸入，$Y(z)$為輸出:
$$\begin{aligned}
Y(z)&=H(z)X(z)\\\\
&=\frac{P(z)}{Q(z)}X(z)\\\\
\end{aligned}$$

$$\begin{aligned}
Q(z)Y(z)&=P(z)X(z)\\\\
(z^2-2r\cos(θ)z+r^2)Y(z)&=(r^2z^2-2r\cos(θ)z+1)X(z)\\\\
(1-2r\cos(θ)z^{-1}+r^2z^{-2})Y(z)&=(r^2-2r\cos(θ)z^{-1}+z^{-2})X(z)\\\\
\end{aligned}$$

經過移項，最後可以得到以下 difference equation:
$$y[n] = r^2 * x[n]-2r\cos(θ) * x[n-1]+x[n-2]+2r \cos(θ) * y[n-1]-r^2* y[n-2]$$

其中，$x[n]$為目前輸入的 sample，$y[n]$為目前欲輸出的 sample。因為計算 $y[n]$ 會用到 4 個以前的值，所以此 filter 需要 4 條 delay line(或 4 個 memory)。實作如下:
```c++
float* update(float* input)override {
    for (int i =0;i<inputDim;i++) {
        output[i] = R2[i] * input[i] - twoRCosTheta[i] * x1[i] + x2[i] + twoRCosTheta[i] * y1[i] - R2[i]*y2[i];

        x2[i] = x1[i];
        x1[i] = input[i];

        y2[i] = y1[i];
        y1[i] = output[i];
    }
    return output;
}
```

### Comb
Comb 超簡單，因為它是 FIR，沒有 feedback:
```c++
float* update(float* input)override {
    add(inputDim, input, delay.update(input));
    return input;
}
```
(但其實這個沒有用到，我們用 all pass 代替它了

### Reverb

Reverb 這個最大的 filter 就是把所有小 filter 組裝起來。

```c++
float* update(float* input) override{
    
    float dry[2];
    copy(inputDim, dry, input);

    input = distrib * inDelay.update(input);

    delayFilters.update(feedBack);

    add(NCH,input, fbDelayLine.update(feedBack));

    input = allpass.update(input);

    input = mult(inputDim, input, _decay);

    float* output = feedbackmatrix * input;

    copy(NCH,feedBack, output);

    return dcBlocker.update(add(inputDim,mult(inputDim,outDistrib*output,wetAmount), mult(inputDim, dry,dryAmount)));
}
```


## 調整參數

把所有 filter 組裝起來之後，我們遇到的第一個問題是跑了一段時間後數值很容易爆炸。這是因為在主迴圈中，如果有某個頻率的 amplitude response 超過 1，經過數次遞迴，那個頻率的強度就會指數發散。

我們隨即調低迴圈的 feedback matrix 的值，使得 amplitude response 下降。這時又出現另一個問題:殘響時間不夠長。當迴圈的 amplitude response 小於 1 太多，經過數次遞迴，聲音就會快速衰減並消失。

也就是說，必須讓每種頻率的 amplitude response 都小於 1，但只能小一點點。

後來我們找到的作法是:

1. low pass 的 amplitude response 小於等於 1
2. all pass 的 amplitude response 等於 1 (當然)
3. feedback matrix 的每個 row 絕對值總和稍微小於 1 

這樣的話就可以保證不會爆炸了。原因如下:
以最嚴格的情況來看，假設 low pass 和 all pass 的 amplitude response 都是 1，訊號從 feedback matrix 的輸出繞一圈回到 feedback matrix 前的 amplitude 增益就是 1，強度不變。而 feedback matrix 會將 8 個 channel 重新混合，feedback matrix 的每個 row 絕對值總和小於 1 這項限制保證了混合後的訊號不會因疊加而增強。

不過因為訊號繞了一圈後，會發生複雜的項位改變，就算 feedback matrix 每個 row 絕對值總和非常接近 1，訊號卻很容易因為破壞性疊加而有很大的衰減率。而且 VST 的 sample rate 非常高(例如44100Hz)，所以還是會有不到 1 秒聲音就幾乎不見的情況。