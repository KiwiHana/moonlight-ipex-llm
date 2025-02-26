# moonlight by ipex-llm
Run moonlight-16B-A3B-instruct on intel core ultra or A770/B580 by intel ipex-llm-transformers python

# step 1: create env
```
conda create -n ipex-llm python=3.11 libuv
conda activate ipex-llm
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tiktoken blobfile transformers==4.45 trl==0.11 accelerate==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --pre --upgrade ipex-llm==2.2.0b20250224
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

run convert.py

```
python convert.py
```

and then copy the following files from /path/to/Moonlight-16B-A3B-Instruct to /path/to/Moonlight-16B-A3B-Instruct-converted

![image](https://github.com/user-attachments/assets/11973b63-c708-47d0-8bcb-bc0f30eae0eb)
![image](https://github.com/user-attachments/assets/6d007d5a-15b9-4701-bc88-c8a29663178d)


# step 4: run Moonlight-16B-A3B-Instruct model
```
set SYCL_CACHE_PERSISTENT=1 
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
python run_moonlight-16B-A3B-instruct.py
```
![image](https://github.com/user-attachments/assets/ed54cd46-8b00-4d90-85df-d3362c0d9dcf)

