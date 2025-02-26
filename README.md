# Moonlight by ipex-llm
Run moonlight-16B-A3B-instruct on intel core ultra or A770/B580 by intel ipex-llm-transformers python

intel core ultra recommand 32GB memory.

# step 1: create env

Install miniforge on windows: https://conda-forge.org/miniforge/

```
conda create -n ipex-llm python=3.11 libuv
conda activate ipex-llm
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tiktoken blobfile transformers==4.45 trl==0.11 accelerate==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --pre --upgrade ipex-llm==2.2.0b20250224
```

pip list
```
Package            Version
------------------ --------------
accelerate         0.26.0
aiohappyeyeballs   2.4.6
aiohttp            3.11.13
aiosignal          1.3.2
attrs              25.1.0
bigdl-core-xe-all  2.6.0b20250223
blobfile           3.0.0
certifi            2025.1.31
charset-normalizer 3.4.1
colorama           0.4.6
datasets           3.3.2
dill               0.3.8
docstring_parser   0.16
dpcpp-cpp-rt       2025.0.2
filelock           3.17.0
frozenlist         1.5.0
fsspec             2024.12.0
huggingface-hub    0.29.1
idna               3.10
intel-cmplr-lib-rt 2025.0.2
intel-cmplr-lib-ur 2025.0.2
intel-cmplr-lic-rt 2025.0.2
intel-opencl-rt    2025.0.2
intel-openmp       2025.0.2
intel-sycl-rt      2025.0.2
ipex-llm           2.2.0b20250224
Jinja2             3.1.5
lxml               5.3.1
markdown-it-py     3.0.0
MarkupSafe         3.0.2
mdurl              0.1.2
mpmath             1.3.0
multidict          6.1.0
multiprocess       0.70.16
networkx           3.4.2
numpy              1.26.4
onednn             2025.0.1
onednn-devel       2025.0.1
packaging          24.2
pandas             2.2.3
pillow             11.1.0
pip                25.0.1
propcache          0.3.0
protobuf           6.30.0rc1
psutil             7.0.0
py-cpuinfo         9.0.0
pyarrow            19.0.1
pycryptodomex      3.21.0
Pygments           2.19.1
python-dateutil    2.9.0.post0
pytz               2025.1
PyYAML             6.0.2
regex              2024.11.6
requests           2.32.3
rich               13.9.4
safetensors        0.5.2
sentencepiece      0.2.0
setuptools         75.8.0
shtab              1.7.1
six                1.17.0
sympy              1.13.1
tabulate           0.9.0
tbb                2022.0.0
tcmlib             1.2.0
tiktoken           0.9.0
tokenizers         0.20.3
torch              2.6.0+xpu
torchaudio         2.6.0+xpu
torchvision        0.21.0+xpu
tqdm               4.67.1
transformers       4.45.0
trl                0.11.0
typeguard          4.4.2
typing_extensions  4.12.2
tyro               0.9.16
tzdata             2025.1
umf                0.9.1
urllib3            2.3.0
wheel              0.45.1
xxhash             3.5.0
yarl               1.18.3
```

# step 2: download model

Download Moonlight-16B-A3B-Instruct model from https://modelscope.cn/models/moonshotai/Moonlight-16B-A3B-Instruct

For example:
```
pip install modelscope
modelscope download --model moonshotai/Moonlight-16B-A3B-Instruct --local_dir ./Moonlight-16B-A3B-Instruct
```
or dowload model from huggingface : https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct

# step 3: convert model

modify src_path and dst_path in Line 5 and Line 6 on convert.py

```
python convert.py
```

and then copy the following files from /path/to/Moonlight-16B-A3B-Instruct to /path/to/Moonlight-16B-A3B-Instruct-converted

![image](https://github.com/user-attachments/assets/11973b63-c708-47d0-8bcb-bc0f30eae0eb)
![image](https://github.com/user-attachments/assets/6d007d5a-15b9-4701-bc88-c8a29663178d)


# step 4: run Moonlight-16B-A3B-Instruct model

modify model_path in Line 34 on run_moonlight-16B-A3B-instruct.py

```
set SYCL_CACHE_PERSISTENT=1 
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
python run_moonlight-16B-A3B-instruct.py
```
![image](https://github.com/user-attachments/assets/ed54cd46-8b00-4d90-85df-d3362c0d9dcf)

因式分解：x^3-19x+30

以AIPC为主题，创作一首现代诗，赞美时代的美好，人人都能在笔记本电脑上体验人工智能带来的新功能。
