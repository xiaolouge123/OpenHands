import argparse
import asyncio
import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from hashlib import md5
from typing import Any, Optional, Union

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


def get_tool_call_id():
    return 'tool-call-' + str(uuid.uuid4())


def add_tool_call_id(tool_calls):
    for tool_call in tool_calls:
        if tool_call.get('id', '') == '':
            tool_call['id'] = get_tool_call_id()
    return tool_calls


# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--enable_cache', action='store_true', default=False, help='是否启用请求缓存'
    )
    parser.add_argument(
        '--enable_ollama',
        action='store_true',
        default=False,
        help='是否启用本地ollama API',
    )
    parser.add_argument(
        '--enable_volcengine',
        action='store_true',
        default=False,
        help='是否启用火山引擎 API',
    )
    return parser.parse_args()


# 修改日志配置
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/openai_proxy.log'

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        # 文件处理器 - 按天滚动日志文件
        logging.handlers.TimedRotatingFileHandler(
            LOG_FILE,
            when='midnight',
            interval=1,
            backupCount=7,  # 保留7天的日志
            encoding='utf-8',
        ),
        # 控制台处理器
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Function(BaseModel):
    name: str
    description: str
    parameters: Optional[dict[str, Any]] = None


class Tool(BaseModel):
    type: str
    function: Function


class ATool(BaseModel):
    name: str
    description: str
    input_schema: Optional[dict[str, Any]]


class OpenAIRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[list[dict]] = None
    response_format: Optional[dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    stream: Optional[bool] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    tools: Optional[list[Union[Tool, ATool]]] = None

    # @model_validator(mode="after")
    # def set_model(self):
    #     self.model = "claude-3-5-sonnet-20241022"
    #     self.max_tokens = 8192
    #     return self


class OpenAIAPI:
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        self.proxy = proxy
        self.url = 'https://api.openai.com/v1/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        logger.info(f'Initializing OpenAIAPI with proxy: {proxy}')
        if not api_key:
            logger.error('API key is not provided!')

    def _get_headers(self):
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    async def request_openai(self, request_data: OpenAIRequest):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logger.info(
                    f'Making streaming request to OpenAI (attempt {retry_count + 1}/{max_retries})'
                )
                logger.debug(
                    f'Request data: {request_data.model_dump(exclude_none=True)}'
                )

                # 设置120秒超时
                timeout = aiohttp.ClientTimeout(total=120)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.url,
                        headers=self._get_headers(),
                        json=request_data.model_dump(exclude_none=True),
                        proxy=self.proxy,
                    ) as response:
                        logger.info(f'Received response with status: {response.status}')

                        if response.status != 200:
                            response_text = await response.text()
                            logger.error(
                                f'OpenAI API error: {response.status} - {response_text}'
                            )
                            if retry_count < max_retries - 1:
                                retry_count += 1
                                continue
                            error_response = json.dumps(
                                {
                                    'data': '',
                                    'code': response.status,
                                    'message': f'Error while calling OpenAI API: {response_text}',
                                    'meta': {},
                                }
                            )
                            yield error_response
                            return

                        response.raise_for_status()
                        async for line in response.content:
                            yield line
                            time.sleep(0.001)
                        return  # 成功完成，退出重试循环

            except asyncio.TimeoutError:
                logger.warning(
                    f'Request timed out after 120 seconds (attempt {retry_count + 1}/{max_retries})'
                )
                if retry_count < max_retries - 1:
                    retry_count += 1
                    continue
                error_response = json.dumps(
                    {
                        'data': '',
                        'code': 'timeout_error',
                        'message': 'Request timed out after 120 seconds',
                        'meta': {},
                    }
                )
                yield error_response
                return

            except Exception as e:
                logger.exception(
                    f'Error in request_openai (attempt {retry_count + 1}/{max_retries}):'
                )
                if retry_count < max_retries - 1:
                    retry_count += 1
                    continue
                error_response = json.dumps(
                    {
                        'data': '',
                        'code': 'internal_error',
                        'message': str(e),
                        'meta': {},
                    }
                )
                yield error_response
                return

    async def request_openai_non_stream(self, request_data: OpenAIRequest):
        async def make_request():
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.url,
                    headers=self._get_headers(),
                    json=request_data.model_dump(exclude_none=True),
                    proxy=self.proxy,
                ) as response:
                    logger.info(f'Received response with status: {response.status}')

                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(
                            f'OpenAI API error: {response.status} - {response_text}'
                        )
                        return None, response.status, response_text

                    response_data = await response.json()
                    # logger.debug(f"Response data: {response_data}")
                    return response_data, 200, None

        try:
            logger.info(
                f'Making non-streaming request to OpenAI with model: {request_data.model}'
            )
            logger.debug(f'Request data: {request_data.model_dump(exclude_none=True)}')

            timeout = aiohttp.ClientTimeout(total=300)

            # 第一次尝试
            max_retries = 3
            retry_count = 0

            try:
                while retry_count < max_retries:
                    response_data, status, error_text = await make_request()
                    if status == 200:
                        return response_data

                    # 重试前记录日志
                    logger.info(
                        f'Attempt {retry_count + 1} failed, retrying in 10 seconds...'
                    )
                    if retry_count < max_retries - 1:
                        await asyncio.sleep(10)
                        retry_count += 1
                        continue

                    # 所有重试都失败后返回错误响应
                    return JSONResponse(
                        content={
                            'data': None,
                            'code': status,
                            'message': error_text,
                            'meta': {},
                        },
                        status_code=status,
                    )

            except asyncio.TimeoutError:
                logger.error('Request timed out after 120 seconds')
                return JSONResponse(
                    content={
                        'data': None,
                        'code': 'timeout_error',
                        'message': 'Request timed out after 120 seconds',
                        'meta': {},
                    },
                    status_code=504,
                )

        except Exception as e:
            logger.exception('Error in request_openai_non_stream:')
            return JSONResponse(
                content={
                    'data': '',
                    'code': 'internal_error',
                    'message': str(e),
                    'meta': {},
                },
                status_code=500,
            )

    async def list_models(self):
        url = 'https://api.openai.com/v1/models'
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=self.headers, proxy=self.proxy
            ) as response:
                return await response.json()


class WokaAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        self.proxy = proxy
        self.url = 'https://4.0.wokaai.com/v1/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        logger.info(f'Initializing Woka OpenAIAPI with proxy: {proxy}')
        if not api_key:
            logger.error('API key is not provided!')

    async def list_models(self):
        url = 'https://4.0.wokaai.com/v1/models'
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.json()


class DeepSeekAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        self.proxy = proxy
        self.url = 'https://api.deepseek.com/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        logger.info(f'Initializing DeepSeek OpenAIAPI with proxy: {proxy}')
        if not api_key:
            logger.error('API key is not provided!')

    async def list_models(self):
        url = 'https://api.deepseek.com/models'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                return await response.json()


class DashscopeAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        self.proxy = proxy
        self.url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        logger.info(f'Initializing Dashscope OpenAIAPI with proxy: {proxy}')
        if not api_key:
            logger.error('API key is not provided!')

    async def list_models(self):
        url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/models'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                return await response.json()


class LocalOllamaAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = None
        self.proxy = proxy
        self.url = 'http://localhost:11434/api/chat'
        self.headers = {
            'Content-Type': 'application/json',
        }

    async def list_models(self):
        list_models_url = 'http://localhost:11434/api/tags'
        async with aiohttp.ClientSession() as session:
            async with session.get(list_models_url) as response:
                return await response.json()


class VolcEngineAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        self.proxy = proxy
        self.url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        self.id2model_name = {
            'ep-20250310165044-pbqpf': 'deepseek-chat',
            'ep-20250205221026-wtfnn': 'deepseek-reasoner',
            'doubao-1-5-vision-pro-32k-250115': 'doubao-1-5-vision-pro-32k-250115',
        }
        self.model_name2id = {v: k for k, v in self.id2model_name.items()}

        self.support_models = [k for k in self.id2model_name.keys()] + [
            v for v in self.id2model_name.values()
        ]

    async def list_models(self):
        return {
            'data': [
                {
                    'id': model,
                    'object': 'model',
                    'created': 0,
                    'owned_by': 'volcengine',
                    'model': model,
                }
                for model in self.support_models
            ]
        }

    def cast_model_name(self, model_name):
        if model_name in self.support_models:
            if model_name in self.model_name2id:
                return self.model_name2id[model_name]
            else:
                return model_name
        else:
            raise HTTPException(
                status_code=400,
                detail=f'Model {model_name} not supported with VolcEngine API',
            )


class GeminiAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        if ',' in api_key:
            self.api_key_pool = api_key.split(',')
        else:
            self.api_key_pool = [api_key]
        print(f'Initializing Gemini API with api_key_pool: {self.api_key_pool}')
        self.proxy = proxy
        logger.info(f'Initializing Gemini API with proxy: {proxy}')
        self.url = (
            'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
        )

    def _get_headers(self):
        api_key = random.choice(self.api_key_pool)
        return {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

    async def list_models(self):
        url = 'https://generativelanguage.googleapis.com/v1beta/openai/models'
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=self._get_headers(), proxy=self.proxy
            ) as response:
                return await response.json()


class RTEngineAPI(OpenAIAPI):
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        self.proxy = proxy
        self.url = 'http://10.100.0.2:8500/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
        }

    async def list_models(self):
        url = 'http://10.100.0.2:8500/v1/chat/models'
        headers = {
            'Accept': 'application/json',
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                models = await response.json()
        return {
            'data': [
                {
                    'id': model,
                    'object': 'model',
                    'created': 0,
                    'owned_by': 'rtengine',
                    'model': model,
                }
                for model in models
            ]
        }


async def check_internal_network():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://10.100.0.2:8500/v1/chat/models') as response:
                return response.status == 200
    except Exception:
        return False


args = parse_args()
model_to_api = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_to_api

    if os.getenv('OPENAI_API_KEY'):
        logger.info('Using OpenAI API')
        api_key = os.getenv('OPENAI_API_KEY')
        api = OpenAIAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        logger.info(
            f'OpenAI Available models: {[model["id"] for model in models["data"]]}'
        )
        models = [model['id'] for model in models['data']]
        for model in models:
            if model not in model_to_api:
                model_to_api[model] = api

    if os.getenv('WOKA_API_KEY'):
        logger.info('Using Woka API')
        api_key = os.getenv('WOKA_API_KEY')
        api = WokaAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        logger.info(
            f'WOKA Available models: {[model["id"] for model in models["data"]]}'
        )
        models = [model['id'] for model in models['data']]
        for model in models:
            if model not in model_to_api:
                model_to_api[model] = api

    if os.getenv('DEEPSEEK_API_KEY'):
        logger.info('Using DeepSeek API')
        api_key = os.getenv('DEEPSEEK_API_KEY')
        api = DeepSeekAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        logger.info(
            f'DeepSeek Available models: {[model["id"] for model in models["data"]]}'
        )
        models = [model['id'] for model in models['data']]
        for model in models:
            if model not in model_to_api:
                model_to_api[model] = api

    if os.getenv('DASHSCOPE_API_KEY'):
        logger.info('Using Dashscope API')
        api_key = os.getenv('DASHSCOPE_API_KEY')
        api = DashscopeAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        logger.info(
            f'Dashscope Available models: {[model["id"] for model in models["data"]]}'
        )
        models = [model['id'] for model in models['data']]
        for model in models:
            if model not in model_to_api:
                model_to_api[model] = api

    if os.getenv('GEMINI_API_KEY'):
        logger.info('Using Gemini API')
        api_key = os.getenv('GEMINI_API_KEY')
        api = GeminiAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        # print(models)

        models = [model['id'] for model in models['data']]
        models = [m.lstrip('models/') for m in models]
        logger.info(f'Gemini Available models: {models}')
        for model in models:
            if model not in model_to_api:
                model_to_api[model] = api

    if args.enable_ollama:
        logger.info('Using local Ollama API')
        local_api = LocalOllamaAPI(api_key=None, proxy=None)
        models = await local_api.list_models()
        models = [model['name'] for model in models['models']]
        logger.info(f'Ollama Available models: {models}')
        for model in models:
            if model not in model_to_api:
                model_to_api[model] = local_api

    if args.enable_volcengine:
        logger.info('Using VolcEngine API')
        api_key = os.getenv('VOLCENGINE_API_KEY')
        api = VolcEngineAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        logger.info(
            f'VolcEngine Available models: {[model["id"] for model in models["data"]]}'
        )
        models = [model['id'] for model in models['data']]
        logger.info(f'Replacing following models: {models} with VolcEngine API')
        for model in models:
            model_to_api[model] = api

    if await check_internal_network():
        logger.info('Using RTEngine API')
        api_key = os.getenv('RTE_API_KEY')
        api = RTEngineAPI(api_key=api_key, proxy=None)
        models = await api.list_models()
        logger.info(
            f'RTEngine Available models: {[model["id"] for model in models["data"]]}'
        )
        models = [model['id'] for model in models['data']]
        logger.info(f'Replacing following models: {models} with RTEngine API')
        for model in models:
            model_to_api[model] = api

    logger.info(f'ALL Available models: {model_to_api.keys()}')
    yield
    logger.info('Closing application')
    del api, local_api


# FastAPI app
app = FastAPI(lifespan=lifespan)


# 创建一个缓存类
class ResponseCache:
    def __init__(self, expire_minutes=30):
        self.cache = {}
        self.expire_minutes = expire_minutes

    def get_cache_key(self, request_data: OpenAIRequest) -> str:
        # 将请求数据转换为可哈希的字符串，并生成MD5作为缓存键
        cache_data = request_data.model_dump(exclude_none=True)
        cache_str = json.dumps(cache_data, sort_keys=True)
        return md5(cache_str.encode()).hexdigest()

    def get(self, key: str):
        if key in self.cache:
            cached_item = self.cache[key]
            # 检查是否过期
            if datetime.now() < cached_item['expire_time']:
                return cached_item['data']
            else:
                # 删除过期数据
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = {
            'data': value,
            'expire_time': datetime.now() + timedelta(minutes=self.expire_minutes),
        }


def response_postprocess_top1_tool_call(response: dict):
    if (
        isinstance(response, dict)
        and 'choices' in response
        and response['choices']
        and response['choices'][0]['finish_reason'] == 'tool_calls'
    ):
        if (
            'tool_calls' in response['choices'][0]['message']
            and response['choices'][0]['message']['tool_calls']
        ):
            tool_calls = response['choices'][0]['message']['tool_calls']
            if len(tool_calls) > 1:
                logger.info(f'Only keep the first tool call: {tool_calls[0]}')
                response['choices'][0]['message']['tool_calls'] = [tool_calls[0]]
            elif len(tool_calls) == 1:
                logger.info(
                    f'Only one tool call found in the response: {tool_calls[0]}'
                )
            else:
                logger.info('No tool calls found in the response')
    return response


def request_preprocess_for_gemini_tool_call_with_image(request_data: OpenAIRequest):
    if request_data.model == 'gemini-2.5-pro-preview-03-25':
        for message in request_data.messages:
            if message['role'] == 'tool' and message['content']:
                if isinstance(message['content'], list) and len(message['content']) > 0:
                    role_revert_mark = False
                    for content in message['content']:
                        if isinstance(content, dict) and 'image_url' in content:
                            role_revert_mark = True
                            break
                    if role_revert_mark:
                        message['role'] = 'user'
    return request_data


# 创建缓存实例
response_cache = ResponseCache()


@app.post('/v1/chat/completions')
# async def create_chat_completion(request: Request):
#     # 添加请求体的调试日志
#     request_data = request
#     body = await request.json()
#     print("收到的请求内容:", json.dumps(body, ensure_ascii=False))

#     # 如果需要更详细的请求信息，还可以打印headers
#     print("请求头:", request.headers)
async def get_openai_response(request_data: OpenAIRequest):
    global model_to_api
    logger.info('Received chat completion request')
    api = model_to_api[request_data.model]
    request_data = request_preprocess_for_gemini_tool_call_with_image(request_data)
    casting = False
    if isinstance(api, VolcEngineAPI):
        request_data.model = api.cast_model_name(request_data.model)
        casting = True
    try:
        # 检查是否启用缓存且有缓存数据
        if args.enable_cache:
            cache_key = response_cache.get_cache_key(request_data)
            cached_response = response_cache.get(cache_key)

            if cached_response:
                logger.info('Returning cached response')
                if request_data.stream:

                    async def stream_cached_response():
                        yield json.dumps(cached_response).encode()

                    return StreamingResponse(
                        stream_cached_response(), media_type='text/event-stream'
                    )
                return cached_response

        # 没有缓存时，调用API并缓存结果
        if request_data.stream:
            logger.info('Processing streaming request')
            response_chunks = []

            async def cached_stream():
                async for chunk in api.request_openai(request_data):
                    response_chunks.append(chunk)
                    yield chunk
                # 流式请求完成后，缓存完整响应
                if args.enable_cache:
                    full_response = b''.join(response_chunks)
                    try:
                        response_json = json.loads(full_response)
                        response_cache.set(cache_key, response_json)
                    except json.JSONDecodeError:
                        logger.warning(
                            'Failed to cache streaming response: Invalid JSON'
                        )

            return StreamingResponse(cached_stream(), media_type='text/event-stream')
        else:
            logger.info('Processing non-streaming request')
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                response = await api.request_openai_non_stream(request_data)
                # 添加 usage 结构标准化处理
                if isinstance(response, dict) and 'usage' in response:
                    if 'completion_tokens' not in response['usage']:
                        response['usage']['completion_tokens'] = 0
                    if 'prompt_tokens' not in response['usage']:
                        response['usage']['prompt_tokens'] = 0
                    if 'total_tokens' not in response['usage']:
                        response['usage']['total_tokens'] = (
                            response['usage']['prompt_tokens']
                            + response['usage']['completion_tokens']
                        )
                    if 'usage' not in response:
                        response['usage'] = {
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0,
                        }

                # 检查响应内容
                if (
                    isinstance(response, dict)
                    and 'choices' in response
                    and response['choices']
                    and 'tool_calls' in response['choices'][0]['message']
                    and response['choices'][0]['message']['tool_calls']
                ):
                    # 检查 tool calls 回复
                    logger.info('OAI Tool call response')
                    response['choices'][0]['message']['tool_calls'] = add_tool_call_id(
                        response['choices'][0]['message']['tool_calls']
                    )
                    if casting:
                        response = response_postprocess_top1_tool_call(response)

                elif (
                    isinstance(response, dict)
                    and 'choices' in response
                    and response['choices']
                    and 'message' in response['choices'][0]
                    and 'content' in response['choices'][0]['message']
                    and (
                        not response['choices'][0]['message']['content'].strip()
                        or "Sorry, I didn't understand your query"
                        in response['choices'][0]['message']['content']
                    )
                ):
                    retry_count += 1
                    logger.warning(
                        f'Empty or error response received, retrying... (attempt {retry_count}/{max_retries})'
                    )
                    if retry_count == max_retries:
                        logger.error('Max retries reached with invalid responses')
                        break
                    continue

                if (
                    isinstance(response, dict)
                    and 'error' in response
                    and response['error']
                    == {'message': '', 'type': '', 'param': '', 'code': None}
                ):
                    del response['error']

                if args.enable_cache:
                    response_cache.set(cache_key, response)

                logger.info(f'Final Response: {response}')
                return response

    except Exception as e:
        logger.exception('Error in get_openai_response endpoint:')
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=23323)
