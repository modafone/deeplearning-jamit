# 日本医用画像工学会 Medical Imaging Technology (MIT) 誌第36巻2号特集「Kerasによるディープラーニング」関連ファイル

## 2D画像2クラス分類ソースコード

keras_cnn_imageclassify.py

## サンプルデータ

用意するかもしれない

## WindowsへのKerasインストール方法

Windows10 64bitへインストールしたときの方法です．Windowsの他のバージョンでも同様だと思います．

Visual Studio 2013 or 2010をインストールする．（2015では動かないかも）Visual studioが無い場合は，無料のVisual Studio 2013 communityをインストールしても良いです．

CUDAをダウンロードしインストールする  
<https://developer.nvidia.com/cuda-toolkit-archive>
* バックエンドにTensorFlowを使う場合CUDA 8.0選択
* バックエンドにTheanoを使う場合CUDA 7.5を選択

Git for Windowsをダウンロードしインストールする  
<http://msysgit.github.io/>

Anaconda（Python3.6 version，64-bit installer）をダウンロードしインストールする  
<https://www.anaconda.com/download/>

スタートメニューのAnaconda3グループにあるAnaconda Promptを管理者権限で起動する．以下ではこのプロンプトに入力する．  
アップデートする．
```bash
conda update conda
```

Python3.6が含まれたAnacondaをインストールしたが，後で使うopencv3は現在Python3.5までしか対応しないのでPyton3.5の仮想環境を作る．
```bash
conda create -n py35 python=3.5
```
作成した仮想環境を有効にする．このコマンドはAnaconda Promptを起動するたびに入力する．（起動するたびに入力するのを忘れずに！）
```bash
activate py35
```

必要なパッケージのインストールを行う
```bash
conda install numpy scipy mingw libpython spyder
```

*バックエンドにTensorFlowを使う場合は* インストールする
```bash
pip install tensorflow-gpu
```
*バックエンドにTensorFlowを使う場合は* NVIDIAのサイトからcuDNN v6.0 for CUDA 8.0をダウンロードし，bin,include,lib内のファイルをCUDAインストール先ディレクトリ（おそらくC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0）にコピーする．cuDNNのダウンロードには会員登録が必要．  
<https://developer.nvidia.com/cudnn>

*バックエンドにTheanoを使う場合は* インストールする
```bash
conda install theano
```
*バックエンドにTheanoを使う場合は* Theanoの設定ファイルを作る．C:\Users\（ユーザ名）\に「.theanorc.txt」というファイル名のテキストファイルを作成し，以下の内容を書いてください．  
「#」で始まる行はコメントアウトの意味です．  
PCのGPU搭載有無によってdeviceの値を変更してください．  
cuDNNを入れた場合はoptimizer_includingのコメントアウトを外してください．現在TheanoがサポートするcuDNNはv5.1までなのでこれより新しいものは入れないでください．  
Visual Studioのバージョンによってcompiler_bindirのディレクトリを変更してください．
```bash
[global]
device = cpu
#device = gpu
floatX = float32
#optimizer_including = cudnn
optimizer=fast_run
#optimizer=fast_compile

[nvcc]
compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
fastmath = True
```

Kerasのインストールを行う
```bash
pip install keras
```

ソースコード実行に必要なものをインストールする
```bash
pip install scikit-learn matplotlib pyyaml h5py pillow
conda install --channel https://conda.anaconda.org/menpo opencv3
```

Kerasを使ったコードを何か1回実行する
```bash
python keras_cnn_imageclassify.py
```
実行後にC:\Users\（ユーザ名）\.keras\keras.jsonをテキストエディタで開き，バックエンドの指定を書く．
```bash
"backend": "tensorflow"
```
または
```bash
"backend": "theano"
```
