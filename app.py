# -*- coding: utf-8 -*-
"""
校园 AI 助教 —— 课程答疑 + 自动出题
基于智谱 GLM-4.7-Flash 免费模型，Streamlit 单文件网页应用
"""

import os
import time
import streamlit as st
from openai import OpenAI

# ============================================
# 页面配置（必须在所有 Streamlit 命令之前）
# ============================================
st.set_page_config(
    page_title="校园AI助教 · 智能答疑与出题",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# 自定义样式（注入 CSS）
# ============================================
st.markdown(
    """
    <style>
    /* 全局字体 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', -apple-system, sans-serif;
    }

    /* 隐藏默认 Streamlit 元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* 渐变标题栏 */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(118, 75, 162, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.92);
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* 统计卡片 */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        margin-top: 0.2rem;
    }
    .stat-card.blue { border-left-color: #4facfe; }
    .stat-card.green { border-left-color: #43e97b; }
    .stat-card.orange { border-left-color: #fa709a; }

    /* 按钮美化 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.25s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(118, 75, 162, 0.35);
        filter: brightness(1.08);
    }
    .stButton > button p { color: white !important; }

    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f7fa 0%, #e8ecf3 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #4a4a6a;
    }

    /* 聊天消息圆角 */
    .stChatMessage {
        border-radius: 12px;
    }

    /* 输入框美化 */
    .stChatInput > div {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* 页脚 */
    .app-footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }

    /* 主内容区宽度限制 */
    .stApp > section.main > div {
        max-width: 900px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================
# 从本地 secrets 读取 API Key（不会泄露到代码仓库）
# ============================================
API_KEY = st.secrets.get("API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 模型降级列表：依次尝试，前一个限流就切下一个（同 Key，都免费）
FALLBACK_MODELS = ["GLM-4.7-Flash", "GLM-4-Flash"]

# 课件最大字符数（截断防止超长，免费模型上下文足够）
MAX_KNOWLEDGE_CHARS = 8000

# 内置题库文件路径（自动加载，无需手动上传）
QUESTION_BANK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "题库.txt")


def load_default_question_bank():
    """启动时自动加载内置题库，无需手动上传。"""
    try:
        with open(QUESTION_BANK_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:MAX_KNOWLEDGE_CHARS]
    except Exception:
        return ""

# ============================================
# 课程模式预设（不同人设的系统提示词）
# ============================================
COURSE_MODES = {
    "高等数学": "你是一位耐心、严谨的高等数学助教。回答时步骤清晰、推理完整，适当举例说明，用语专业但易懂。",
    "编程开发": "你是一位耐心的编程助教。回答时注重代码示例和原理解释，语言简洁，代码用代码块包裹。",
    "英语学习": "你是一位友好的英语学习助教。回答时注重语法解释和例句，必要时对比中英表达差异。",
    "通用模式": "你是一位耐心、专业的大学助教。请用中文回答学生的问题，回答需清晰、有条理。",
}

# ============================================
# 初始化会话状态
# ============================================
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "course_mode" not in st.session_state:
        st.session_state.course_mode = "通用模式"
    if "knowledge_base" not in st.session_state:
        # 启动时自动加载内置题库，无需手动上传
        st.session_state.knowledge_base = load_default_question_bank()
    if "knowledge_source" not in st.session_state:
        st.session_state.knowledge_source = "内置题库"


init_state()


# ============================================
# 获取系统提示词（根据课程模式）
# ============================================
def get_system_prompt():
    mode = st.session_state.course_mode
    base = COURSE_MODES.get(mode, COURSE_MODES["通用模式"])
    if st.session_state.knowledge_base:
        source = st.session_state.get("knowledge_source", "题库")
        base += f"\n\n回答时请优先基于以下{source}内容。如果题库中有相关内容，请在回答末尾标注【出自题库】；如果是你补充的通用知识，标注【AI补充】。"
    return base


# ============================================
# 调用智谱 API（流式输出）
# ============================================
def call_api(messages, temperature=0.7, max_tokens=2048):
    """调用智谱 API 并流式返回文本。
    策略：依次尝试 FALLBACK_MODELS 中的模型，每个模型遇 429 重试 2 次，
    仍失败则切下一个模型；全部失败返回 None。"""
    if not API_KEY:
        st.error("未检测到 API Key，请在 .streamlit/secrets.toml 中配置 API_KEY。")
        return None

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    is_last_model = False
    last_error = ""
    for model_idx, model_name in enumerate(FALLBACK_MODELS):
        is_last_model = model_idx == len(FALLBACK_MODELS) - 1
        max_retries = 2  # 每个模型重试 2 次
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response
            except Exception as e:
                msg = str(e)
                last_error = msg
                # 429 限流：等待后重试，或切下一个模型
                if "429" in msg or "1305" in msg or "访问量过大" in msg:
                    if attempt < max_retries - 1:
                        wait = 3 * (attempt + 1)
                        with st.spinner(f"{model_name} 限流，{wait}秒后重试..."):
                            time.sleep(wait)
                        continue
                    elif not is_last_model:
                        with st.spinner(f"{model_name} 限流，切换到备用模型 {FALLBACK_MODELS[model_idx+1]}..."):
                            time.sleep(1)
                        break  # 切下一个模型
                    else:
                        st.warning("所有模型当前访问量过大，已重试多次仍失败。请稍等 1-2 分钟后再次提问。")
                        return None
                elif "auth" in msg.lower() or "401" in msg or "api key" in msg.lower():
                    st.error("API Key 无效，请检查 .streamlit/secrets.toml 中的配置。")
                    return None
                elif "connect" in msg.lower() or "timeout" in msg.lower():
                    st.error("网络连接失败，请检查网络后重试。")
                    return None
                else:
                    # 其他错误：若还有备用模型则切换，否则报错
                    if not is_last_model:
                        break
                    st.error(f"调用出错：{msg}")
                    return None
    st.error(f"调用出错：{last_error}")
    return None


def stream_response(messages, temperature=0.7, max_tokens=2048):
    """流式输出 API 回答，返回完整文本。"""
    response = call_api(messages, temperature=temperature, max_tokens=max_tokens)
    if response is None:
        return None

    full_text = ""
    placeholder = st.empty()
    try:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                placeholder.markdown(full_text + "▌")
    except Exception as e:
        st.error(f"生成中断：{e}")
        if full_text:
            return full_text
        return None

    placeholder.markdown(full_text)
    return full_text if full_text else None


# ============================================
# 左侧边栏：课程模式 + 上传课件 + 使用说明
# ============================================
with st.sidebar:
    st.header("📚 控制面板")

    # 课程模式切换
    st.subheader("课程模式")
    new_mode = st.selectbox(
        "选择助教人设",
        list(COURSE_MODES.keys()),
        index=list(COURSE_MODES.keys()).index(st.session_state.course_mode),
    )
    if new_mode != st.session_state.course_mode:
        st.session_state.course_mode = new_mode
        st.rerun()
    st.caption(f"当前模式：**{st.session_state.course_mode}**")

    st.divider()

    # 题库状态 + 可选上传替换
    st.subheader("📚 题库")
    kb_len = len(st.session_state.knowledge_base)
    src = st.session_state.get("knowledge_source", "内置题库")
    st.success(f"当前知识来源：**{src}**（{kb_len} 字符）")

    with st.expander("📁 上传自定义资料替换题库（可选）"):
        uploaded_file = st.file_uploader(
            "上传 TXT 文件", type=["txt"], key="courseware_upload"
        )
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            st.session_state.knowledge_base = content[:MAX_KNOWLEDGE_CHARS]
            st.session_state.knowledge_source = uploaded_file.name
            st.success(f"已替换为：{uploaded_file.name}（{len(content)} 字符）")

    if st.button("🔄 恢复内置题库", use_container_width=True):
        st.session_state.knowledge_base = load_default_question_bank()
        st.session_state.knowledge_source = "内置题库"
        st.rerun()

    st.divider()

    # 使用说明
    st.subheader("📖 使用说明")
    st.markdown(
        """
        1. **直接提问**：内置题库已加载，直接在下方输入框提问\n
        2. **出题**：点击「📝 自动出题」基于题库生成新练习题\n
        3. **切换模式**：上方选择不同课程人设\n
        4. **替换资料**：可选上传自己的 TXT 替换题库\n
        5. **清空对话**：开始新的对话话题
        """
    )

    st.divider()
    st.caption("模型：智谱 GLM-4.7-Flash（免费）")


# ============================================
# 主聊天区
# ============================================

# 渐变标题栏
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">🎓 校园 AI 助教</h1>
        <p class="hero-subtitle">内置多学科题库 · AI 智能答疑与自动出题 · 智谱 GLM-4.7-Flash</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 统计卡片
kb_len = len(st.session_state.knowledge_base)
src = st.session_state.get("knowledge_source", "内置题库")
mode = st.session_state.course_mode
col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
with col_s1:
    st.markdown(
        f"""<div class="stat-card blue"><div class="stat-label">当前学科</div><div class="stat-value">{mode}</div></div>""",
        unsafe_allow_html=True,
    )
with col_s2:
    st.markdown(
        f"""<div class="stat-card green"><div class="stat-label">题库容量</div><div class="stat-value">{kb_len} 字</div></div>""",
        unsafe_allow_html=True,
    )
with col_s3:
    st.markdown(
        f"""<div class="stat-card orange"><div class="stat-label">知识来源</div><div class="stat-value" style="font-size:1rem">{src}</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("")

# 顶部操作按钮
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("📝 自动出题", use_container_width=True, type="primary"):
        if not st.session_state.knowledge_base:
            st.warning("题库为空，请上传资料后再出题。")
        else:
            with st.chat_message("user"):
                st.markdown("📝 请根据题库内容出 3 道练习题。")
            st.session_state.messages.append(
                {"role": "user", "content": "📝 请根据题库内容出 3 道练习题。"}
            )

            quiz_prompt = f"""请根据以下题库资料，出 3 道练习题。要求：
1. 题目应覆盖题库中的不同知识点
2. 每道题附上【答案】和【解析】
3. 难度适中，适合学生自测
4. 用以下格式输出：

### 第1题
【题目】...
【答案】...
【解析】...

### 第2题
...

### 第3题
...

【题库资料】
{st.session_state.knowledge_base}"""

            with st.chat_message("assistant"):
                with st.spinner("正在出题中..."):
                    result = stream_response(
                        [{"role": "system", "content": get_system_prompt()}]
                        + [
                            m
                            for m in st.session_state.messages
                            if m["role"] != "system"
                        ]
                        + [{"role": "user", "content": quiz_prompt}]
                    )
            if result:
                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )

with col2:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col3:
    if st.button("➕ 扩充题库", use_container_width=True):
        mode = st.session_state.course_mode
        with st.chat_message("user"):
            st.markdown(f"➕ 请为「{mode}」生成 5 道新练习题，扩充题库。")
        st.session_state.messages.append(
            {"role": "user", "content": f"➕ 请为「{mode}」生成 5 道新练习题，扩充题库。"}
        )
        expand_prompt = f"""请围绕「{mode}」主题，生成 5 道练习题。要求：
1. 题目要新颖，尽量与题库中已有题目不重复
2. 每道题附上【答案】和【解析】
3. 难度从易到难递进
4. 用与题库相同的格式输出：

第N题
【题目】...
【答案】...
【解析】...

现有题库已包含的主题请避免重复，直接输出 5 道新题。"""

        with st.chat_message("assistant"):
            with st.spinner("正在生成新题目扩充题库..."):
                result = stream_response(
                    [{"role": "system", "content": get_system_prompt()}]
                    + [
                        m for m in st.session_state.messages if m["role"] != "system"
                    ]
                    + [{"role": "user", "content": expand_prompt}],
                    temperature=0.9,
                    max_tokens=3000,
                )
        if result:
            # 将新生成的题目追加到题库中，实现实时扩充
            st.session_state.knowledge_base += "\n\n" + result
            st.session_state.knowledge_source = f"内置题库（已扩充）"
            st.session_state.messages.append(
                {"role": "assistant", "content": result}
            )
            st.success(f"✅ 题库已扩充！新增 5 道题，当前题库共 {len(st.session_state.knowledge_base)} 字符。")

st.divider()

# ============================================
# 显示历史对话
# ============================================
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ============================================
# 聊天输入框
# ============================================
if prompt := st.chat_input("直接提问，AI 基于内置题库回答..."):
    # 构造带题库上下文的提问
    if st.session_state.knowledge_base:
        enhanced_prompt = f"""请基于以下题库资料回答问题。如果题库中有相关内容，在回答末尾标注【出自题库】；如果是你补充的通用知识，标注【AI补充】。

【题库资料】
{st.session_state.knowledge_base}

【学生问题】
{prompt}"""
    else:
        enhanced_prompt = prompt

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 存入历史（存增强后的，便于 AI 上下文连贯）
    st.session_state.messages.append({"role": "user", "content": enhanced_prompt})

    # 获取 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            messages_for_api = (
                [{"role": "system", "content": get_system_prompt()}]
                + st.session_state.messages
            )
            result = stream_response(messages_for_api)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
    else:
        # 出错时把刚才的用户消息撤回，避免历史污染
        st.session_state.messages.pop()

# ============================================
# 页脚
# ============================================
st.markdown(
    """
    <div class="app-footer">
        🎓 校园 AI 助教 · 基于 Streamlit + 智谱 GLM-4.7-Flash<br>
        智能答疑 · 自动出题 · 题库扩充 · 多学科切换
    </div>
    """,
    unsafe_allow_html=True,
)
