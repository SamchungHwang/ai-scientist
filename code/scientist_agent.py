import json
import configparser
import http.client
import streamlit as st
import os
import sys
from dotenv import load_dotenv

sys.path.append(".")
import os.path as osp
import time
from typing import List, Dict, Union

# 기존 코드를 가져옴
from llm import get_response_from_llm, extract_json_between_markers
import requests
import backoff
from generate_ideas import generate_ideas, check_idea_novelty, stop_generation

st.title("Welcome to Scientist Agent:blue[cool] :sunglasses:")


# CompletionExecutor 클래스 정의
class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request(
            "POST",
            "/testapp/v1/completions/LK-D",
            json.dumps(completion_request),
            headers,
        )
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding="utf-8"))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res["status"]["code"] == "20000":
            return res["result"]["text"]
        else:
            return "Error"


if "config_loaded" not in st.session_state:
    # 여기에서 config 로드 등 초기 설정 수행
    stop_generation = False
    S2_API_KEY = os.getenv("S2_API_KEY")
    load_dotenv()
    st.session_state.config_loaded = True

with st.sidebar:
    experiment = st.selectbox(
        "실험 유형을 선택하세요",
        ("nanoGPT", "다른 실험 옵션"),  # 필요에 따라 실험 옵션을 추가
    )
    # 모델 선택
    model = st.selectbox(
        "모델을 선택하세요",
        [
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
        ],
        index=1,
    )
    # Create client
    if model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif model.startswith("bedrock") and "claude" in model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock()
    elif model == "gpt-4o-2024-05-13" or model == "hybrid":
        import openai

        print(f"Using OpenAI API with model {model}.")
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()
    elif model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {model} not supported.")

    base_dir = st.text_input("기본 디렉토리 경로 입력", "../templates/nanoGPT")

    # 아이디어 생성을 건너뛸지 여부
    skip_idea_generation = st.checkbox("기존 아이디어 사용", value=False)

    # 아이디어 독창성 검사 여부
    check_novelty = st.checkbox("아이디어 독창성 검사", value=False)


if st.button("아이디어 생성"):
    print("skip_idea_generation=", skip_idea_generation)
    print("check_novelty=", check_novelty)
    print("base_dir=", base_dir)

    # "Stop" 버튼이 눌리면 생성을 종료
    if st.button("Stop"):
        stop_generation

    if skip_idea_generation == False:
        st.write("아이디어 생성을 시작합니다...")
        ideas = generate_ideas(
            base_dir=base_dir,
            client=client,
            model="gpt-4o-2024-05-13",
            skip_generation=False,
            max_num_generations=10,
            num_reflections=3,
            outftn=st.write,
        )
        st.write("아이디어 생성 완료:")
        for idea in ideas:
            st.json(idea)
    elif check_novelty == True:
        st.write("아이디어 독창성 검사를 시작합니다...")
        # `ideas`를 불러오거나 생성된 아이디어를 사용하여 독창성 검사 수행
        ideas = []  # 이미 생성된 아이디어 리스트를 불러와야 함
        if ideas:
            checked_ideas = check_idea_novelty(
                ideas=ideas,
                base_dir=base_dir,
                client=client,
                model="gpt-4o-2024-05-13",
                max_num_iterations=10,
            )
            st.write("독창성 검사 완료:")
            for idea in checked_ideas:
                st.json(idea)
        else:
            st.write("아이디어가 없습니다. 먼저 아이디어를 생성해주세요.")


# Streamlit을 이용한 질문 입력
question = st.text_area("질문 입력", placeholder="질문을 입력해 주세요")
