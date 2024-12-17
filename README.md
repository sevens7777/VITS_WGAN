# VITS WGAN

相较于原VITS改动：

1.判别器最后一层去掉激活函数
models.py 第331行 第359行

2.生成器和判别器的loss不取log
losses.py 第40行 第26行 第28行

3.每次更新判别器的参数之后把它们的绝对值截断到不超过固定常数0.01
train.py 第122-123行
train.py 第128-129行

4.不用基于动量的优化算法（包括momentum和Adam），采用RMSProp
train_ms.py 第93行 第99行
优化器从AdamW换成RMSprop，删除betas参数
train.py 第89行 第94行
优化器从AdamW换成RMSprop，删除betas参数


以下为原作者们的README：

# VITS 原神语音合成V2

本repo包含了我用于训练原神VITS模型对源代码做出的修改，以及新的config文件。

由于各种原因，模型和数据集暂无法公布，感兴趣可以自行提取，自行训练。

此外，也可以尝试使用公开的api：[http://245671.proxy.nscc-gz.cn:8888/ ](http://245671.proxy.nscc-gz.cn:8888/?text=%E4%BD%A0%E5%A5%BD%E5%91%80&speaker=枫原万叶)来进行尝试，此API可用于二创等用途，但禁止用于任何商业用途。
感谢星尘以及国家超级计算广州中心提供的算力支持，感谢VITS模型作者Jaehyeon Kim, Jungil Kong, and Juhee Son，感谢ContentVEC作者 Kaizhi Qian.
本模型训练时使用的所有音频文件版权属于米哈游科技（上海）有限公司。

Query String 参数：

| 参数 | 类型 | 值 |
| ------------- | ------------- | ------------  |
| text | 字符串 |  生成的文本，支持常见标点符号。英文可能无法正常生成，数字请转换为对应的汉字再进行生成。 |
| speaker | 字符串 |  说话者名称。必须是上面的名称之一。 |
| noise | 浮点数 |  生成时使用的 noise_factor，可用于控制感情等变化程度。默认为0.667。 |
| format | 字符串 |  生成语音的格式，必须为mp3或者wav。默认为mp3。 |


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)
