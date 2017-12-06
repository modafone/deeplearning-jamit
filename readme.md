# MIT����36��2�����W�uKeras�ɂ��f�B�[�v���[�j���O�v�֘A�t�@�C��

## 2D�摜2�N���X���ރ\�[�X�R�[�h

keras_cnn_imageclassify.py

## �T���v���f�[�^

�p�ӂ��邩������Ȃ�

## Windows�ւ�Keras�C���X�g�[�����@

Windows10 64bit�փC���X�g�[�������Ƃ��̕��@�ł��DWindows�̑��̃o�[�W�����ł����l���Ǝv���܂��D

Visual Studio 2013 or 2010���C���X�g�[������D�i2015�ł͑��������܂���jVisual studio�������ꍇ�́C������Visual Studio 2013 community���C���X�g�[�����Ă��ǂ��ł��D

CUDA���_�E�����[�h���C���X�g�[������  
<https://developer.nvidia.com/cuda-toolkit-archive>
* �o�b�N�G���h��TensorFlow���g���ꍇCUDA 8.0�I��
* �o�b�N�G���h��Theano���g���ꍇCUDA 7.5��I��

Git for Windows���_�E�����[�h���C���X�g�[������  
<http://msysgit.github.io/>

Anaconda�iPython3.6 version�C64-bit installer�j���_�E�����[�h���C���X�g�[������  
<https://www.anaconda.com/download/>

�X�^�[�g���j���[��Anaconda3�O���[�v�ɂ���Anaconda Prompt���Ǘ��Ҍ����ŋN������D�ȉ��ł͂��̃v�����v�g�ɓ��͂���D  
�A�b�v�f�[�g����D
```bash
conda update conda
```

Python3.6���܂܂ꂽAnaconda���C���X�g�[���������C��Ŏg��opencv3�͌���Python3.5�܂ł����Ή����Ȃ��̂�Pyton3.5�̉��z�������D
```bash
conda create -n py35 python=3.5
```
�쐬�������z����L���ɂ���D���̃R�}���h��Anaconda Prompt���N�����邽�тɓ��͂���D�i�N�����邽�тɓ��͂���̂�Y�ꂸ�ɁI�j
```bash
activate py35
```

�K�v�ȃp�b�P�[�W�̃C���X�g�[�����s��
```bash
conda install numpy scipy mingw libpython spyder
```

*�o�b�N�G���h��TensorFlow���g���ꍇ��* �C���X�g�[������
```bash
pip install tensorflow-gpu
```
*�o�b�N�G���h��TensorFlow���g���ꍇ��* NVIDIA�̃T�C�g����cuDNN v6.0 for CUDA 8.0���_�E�����[�h���Cbin,include,lib���̃t�@�C����CUDA�C���X�g�[����f�B���N�g���i�����炭C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0�j�ɃR�s�[����DcuDNN�̃_�E�����[�h�ɂ͉���o�^���K�v�D  
<https://developer.nvidia.com/cudnn>

*�o�b�N�G���h��Theano���g���ꍇ��* �C���X�g�[������
```bash
conda install theano
```
*�o�b�N�G���h��Theano���g���ꍇ��* Theano�̐ݒ�t�@�C�������DC:\Users\�i���[�U���j\�Ɂu.theanorc.txt�v�Ƃ����t�@�C�����̃e�L�X�g�t�@�C�����쐬���C�ȉ��̓��e�������Ă��������D  
�u#�v�Ŏn�܂�s�̓R�����g�A�E�g�̈Ӗ��ł��D  
PC��GPU���ڗL���ɂ����device�̒l��ύX���Ă��������D  
cuDNN����ꂽ�ꍇ��optimizer_including�̃R�����g�A�E�g���O���Ă��������D����Theano���T�|�[�g����cuDNN��v5.1�܂łȂ̂ł�����V�������͓̂���Ȃ��ł��������D  
Visual Studio�̃o�[�W�����ɂ����compiler_bindir�̃f�B���N�g����ύX���Ă��������D
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

Keras�̃C���X�g�[�����s��
```bash
pip install keras
```

�\�[�X�R�[�h���s�ɕK�v�Ȃ��̂��C���X�g�[������
```bash
pip install scikit-learn matplotlib pyyaml h5py pillow
conda install --channel https://conda.anaconda.org/menpo opencv3
```

Keras���g�����R�[�h������1����s����
```bash
python keras_cnn_imageclassify.py
```
���s���C:\Users\�i���[�U���j\.keras\keras.json���e�L�X�g�G�f�B�^�ŊJ���C�o�b�N�G���h�̎w��������D
```bash
"backend": "tensorflow"
```
�܂���
```bash
"backend": "theano"
```
