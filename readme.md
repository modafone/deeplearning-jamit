# MIT誌第36巻2号特集「Kerasによるディープラーニング」関連ファイル

## 2D画像2クラス分類ソースコード

keras_cnn_imageclassify.py

## サンプルデータ

用意するかもしれない

## WindowsへのKerasインストール方法

Windows10 64bitへインストールしたときの方法です．Windowsの他のバージョンでも同様だと思います．

Visual Studio 2013 or 2010をインストールする．（2015では多分動きません）Visual studioが無い場合は，無料のVisual Studio 2013 communityをインストールしても良いです．

CUDAインストール
* バックエンドにTensorFlowを使う場合
CUDA 8.0をダウンロードしインストールする  
<https://developer.nvidia.com/cuda-toolkit-archive>

* バックエンドにTheanoを使う場合
CUDA 7.5をダウンロードしインストールする  
<https://developer.nvidia.com/cuda-toolkit-archive>

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
