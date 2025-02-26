
import time
import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer,TextStreamer
 
class PerformanceTextStreamer(TextStreamer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.first_token_time = None
        self.start_time = time.time()
        self.token_count = 0

    def on_finalized_text(self, text, stream_end: bool = False):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.token_count += len(self.tokenizer.tokenize(text))
        super().on_finalized_text(text, stream_end)
        if stream_end:
            end_time = time.time()
            # 计算首字延迟时间（毫秒）
            first_token_latency = (self.first_token_time - self.start_time) * 1000
            # 计算后续 token 的生成速度（token/s）
            if self.token_count > 1:
                next_token_speed = (self.token_count - 1) / (end_time - self.first_token_time)
            else:
                next_token_speed = 0
            print('\n---------------------------')
            print(f"output token count",self.token_count + 1)
            print(f"First token latency: {first_token_latency:.2f} ms")
            print(f"Next token generation speed: {next_token_speed:.2f} token/s")

    
model_path = r"C:\Users\Lengda\Documents\Moonlight-16B-A3B-Instruct-converted"
 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    load_in_low_bit='sym_int4',
)
model = model.to('xpu')

prompt = "你好，你是谁？"

messages = [
    {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('xpu')
generated_ids = model.generate(inputs=input_ids,max_new_tokens=100)
response = tokenizer.batch_decode(generated_ids)[0]
print("response:\n",response)

with torch.inference_mode():
    while 1:
        input_text = input("请输入提示词（输入 'q' 退出）: \n")
        if input_text.lower() == 'q':
            break

        # 将输入文本转换为模型可接受的输入格式
        messages = [
        {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
        {"role": "user", "content": input_text}
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('xpu')
        streamer = PerformanceTextStreamer(tokenizer)
        generated_ids = model.generate(inputs=input_ids,streamer=streamer)
        torch.xpu.synchronize()
        print('-' * 80)
