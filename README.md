## Table of contents

1. [Instroduction](#introduction)
2. [How to use `August`](#how_to_use)
   - [Installation](#installation)
   - [Data structrue](#data_structure)
   - [Example usage](#usage)
4. [Reference](#reference)

# <a name='introduction'></a> August

Keyword-Transformer is an Transformer architecture for Keyword spotting problem. Keywords spotting (in other words, Voice Trigger or Wake-up Words Detection) is a very important research problem, is used to detect specific-words from a stream of audio, typically in a low-power always-on setting such as smart speakers and mobile phones or detect profanity-words in live-streaming.

# <a name='how_to_use'></a> How to use

## <a name='installation'></a> Installation

```js

```

## <a name='data_structure'></a> Data structure

```
data
    gsc_v2.1
        train
            active
                right_1.wav
                right_2.wav
                ...
            
            non_active
                on_1.wav
                on_2.wav
                ...

            ...

        valid

        test

```

## <a name='usage'></a> Example usage

### Training

```py

```

### Evaluation

```py

```

### Inference

```py

```


# <a name='reference'></a> Reference

1. Axel Berg, Mark O'Connor and Miguel Tairum Cruz: “[Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/pdf/2104.00769v2.pdf)”, in arXiv:2104.00769, 2021.


# License

      MIT License

      Copyright (c) 2021 Phuc Phan

      Permission is hereby granted, free of charge, to any person obtaining a copy
      of this software and associated documentation files (the "Software"), to deal
      in the Software without restriction, including without limitation the rights
      to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
      copies of the Software, and to permit persons to whom the Software is
      furnished to do so, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all
      copies or substantial portions of the Software.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
      IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
      FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
      LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
      OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
      SOFTWARE.

  
# Author

August was developed by Phuc Phan © Copyright 2021.

For any questions or comments, please contact the following email: phanxuanphucnd@gmail.com