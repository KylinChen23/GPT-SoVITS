from fastapi.applications import FastAPI
import logging
from fastapi import FastAPI, Query
import torch
from TTS_infer_pack.TTS import TTS, TTS_Config
from pydantic import BaseModel
from contextlib import asynccontextmanager
import subprocess
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

### sys ###
# now_dir = os.getcwd()
# sys.path.append("%s/GPT_weights_v4" % (now_dir))
# sys.path.append("%s/SoVITS_weights_v4" % (now_dir))

### logger ###
logger = logging.getLogger('uvicorn')
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

### check device ###
if torch.cuda.is_available():
    device = "cuda"
    logger.warning("device: %s" % (device))
else:
    device = "cpu"
    logger.warning("device: %s" % (device))

### TTS config ### 
tts_config_dict: dict[str, object] = {
    "device": "cuda",
    "is_half": False,
    "version": "v4",
    "vits_weights_path": "SoVITS_weights_v4/Star_Railway_e10_s12730.pth", # ~~~
    "t2s_weights_path": "GPT_weights_v4/Star_Railway-e10.ckpt", # ~~~
    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
}

### 
class dict_language:
    all_zh: str = "all_zh" # 全部按中文识别
    all_ja: str = "all_ja" # 全部按日文识别
    all_en: str = "en" # 全部按英文识别
    zh_en: str = "zh" # 中英混合
    auto: str = "auto" # 多语种启动切分识别语种

class cut_method:
    cut0: str = "cut0" # 不切
    cut1: str = "cut1" # 凑四句一切
    cut2: str = "cut2" # 凑50字一切
    cut3: str = "cut3" # 按中文句号。切
    cut4: str = "cut4" # 按英文句号.切
    cut5: str = "cut5" # 按标点符号切

### input_item ### 
class TTSInput(BaseModel):
    content: str

class inputItem(BaseModel):
    text: str # 要合成的文本内容
    text_lang: str # 文本的语言
    # 参考音频路径
    ref_audio_path: str # 参考音频路径，用于语音风格迁移
    prompt_text: str # 参考音频的提示文本
    prompt_lang: str # 参考音频的提示文本的语言
    aux_ref_audio_paths: list[str] # 辅助参考音频路径列表，用于多说话人音色融合
    # 模型推理控制参数
    top_k: int = 5 # 从预测概率中，选出前 k 个概率最高的词，然后在它们中 随机选择一个
    top_p: float = 1 # 选择一组累积概率 ≥ p 的词（从高到低加起来），然后在这组词中 随机选择一个。
    temperature: float = 1 # 控制 softmax 输出的“平滑度”或“随机性”。
    repetition_penalty: float = 1.35 # 惩罚那些在先前已经生成过的 token，从而减少模型“复读机”行为。
    # 文本处理参数
    text_split_method: str = cut_method.cut1 # 文本切分方法
    batch_size: int = 1 # 一次处理的文本数量
    batch_threshold: float = 0.75
    split_bucket: bool = True
    # 音频输出控制
    return_fragment: bool = False # 是否返回音频片段
    speed_factor: float = 1.0 # 控制音频的速度
    fragment_interval: float = 0.3 # 控制音频片段的间隔
    super_sampling: bool = False # 是否使用超采样
    # 其他参数
    seed: int = -1
    parallel_infer: bool = True # 是否使用并行推理(Text2SemanticLightningModule)

### fastAPI ###
class MyApp:
    def __init__(self):
        self.app = FastAPI(lifespan=self.lifespan)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("启动")
        self.tts_config: TTS_Config = TTS_Config(tts_config_dict)
        self.tts_config.device = tts_config_dict["device"]
        self.tts_config.is_half = tts_config_dict["is_half"]
        self.tts_config.t2s_weights_path = tts_config_dict["t2s_weights_path"]
        self.tts_config.vits_weights_path = tts_config_dict["vits_weights_path"]
        logger.info(self.tts_config)
        self.tts_pipeline: TTS = TTS(self.tts_config)
        yield
        logger.info("关闭")

my_app = MyApp()
app = my_app.app

@app.post("/tts_inference")
async def tts_inference(user_input: TTSInput) -> dict:
    inputs: inputItem = inputItem(
        text=user_input.content,
        text_lang=dict_language.zh_en,
        ref_audio_path="/mnt/d/GPT-SoVITS/Star_Railway/reference_audios/emotions/流萤/中文/【中立_neutral】家族从银河招揽了许多艺术家、建筑师、学者…组成「筑梦师」团队，编织匹诺康尼的美梦。.wav",
        prompt_text="家族从银河招揽了许多艺术家、建筑师、学者…组成「筑梦师」团队，编织匹诺康尼的美梦。",
        prompt_lang=dict_language.all_zh,
        aux_ref_audio_paths=[
            "/mnt/d/GPT-SoVITS/Star_Railway/reference_audios/emotions/流萤/中文/【开心_happy】嗯…不必说什么，你的眼睛已经给了我答案。.wav"
            "/mnt/d/GPT-SoVITS/Star_Railway/reference_audios/emotions/流萤/中文/【生气_angry】既然是「枪火」的试炼…应该能用战斗解决问题吧？希望不会浪费我们太多时间。.wav"
            "/mnt/d/GPT-SoVITS/Star_Railway/reference_audios/emotions/流萤/中文/【难过_sad】怎么了吗？你这样的表情…我从没见过。.wav"
        ],
    )
    for item in my_app.tts_pipeline.run(inputs.model_dump(mode="python")):
        if isinstance(item, tuple) and len(item) == 2:
            sampling_rate, audio = item
            play_np_audio_with_pydub(audio, sampling_rate)
        else:
            logger.info("非音频数据，跳过")

    return {"status": "ok"}

def play_np_audio_with_pydub(audio: np.ndarray, sampling_rate: int, normalize: bool = True):
    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    audio_int16 = np.int16(audio * 32767)
    audio_bytes = audio_int16.tobytes()

    # 构造 pydub 的 AudioSegment
    audio_seg = AudioSegment(
        data=audio_bytes,
        sample_width=2,  # 16-bit audio = 2 bytes
        frame_rate=sampling_rate,
        channels=1
    )

    play(audio_seg)
