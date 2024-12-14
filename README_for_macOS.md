# 安装
双击[`for_macOS_install.command`](for_macOS_install.command)。
# 运行
- 双击[`for_macOS_start.command`](for_macOS_start.command)，则打开WebUI。
- 双击[`for_macOS_start_api.command`](for_macOS_start_api.command)，则供外部（比如 pyvideotrans ）调用。
# 自定义配置
## 使用bigvgan
```command
cd F5-TTS
git submodule update --init --recursive  # (optional, if need bigvgan)
```
If initialize submodule, you should add the following code at the beginning of src/third_party/BigVGAN/bigvgan.py.
```py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```