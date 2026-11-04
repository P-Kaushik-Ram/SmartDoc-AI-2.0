"""Optional local language model provider for the document assistant.

No model is downloaded or contacted unless a recruiter configures Ollama.
"""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class AssistantProviderError(Exception):
    """Safe-to-display assistant provider failure."""


class AssistantUnavailable(AssistantProviderError):
    pass


class AnswerProvider:
    def answer(self, messages):
        raise NotImplementedError


class OllamaProvider(AnswerProvider):
    def __init__(self, endpoint=None, model=None, timeout=None):
        self.endpoint = (endpoint or os.environ.get('SMARTDOC_OLLAMA_URL', 'http://127.0.0.1:11434')).rstrip('/')
        self.model = model or os.environ.get('SMARTDOC_OLLAMA_MODEL', 'qwen2.5:1.5b-instruct-q4_K_M')
        self.timeout = timeout or int(os.environ.get('SMARTDOC_OLLAMA_TIMEOUT', '90'))
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', '::1'}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise AssistantUnavailable('The local model endpoint must use HTTP on this computer (localhost).')
        if not self.model or len(self.model) > 120:
            raise AssistantUnavailable('The local model name is not configured correctly.')

    def answer(self, messages):
        payload = json.dumps({
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {'temperature': 0.1, 'num_ctx': 8192, 'num_predict': 600},
        }).encode('utf-8')
        request = Request(
            f'{self.endpoint}/api/chat', data=payload,
            headers={'Content-Type': 'application/json'}, method='POST',
        )
        try:
            class NoRedirect(HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None

            # Bypass ambient HTTP(S)_PROXY settings: document text must go
            # directly to the explicitly configured loopback service.
            with build_opener(ProxyHandler({}), NoRedirect).open(request, timeout=self.timeout) as response:
                data = json.loads(response.read(1024 * 1024).decode('utf-8'))
            answer = data.get('message', {}).get('content', '')
            if not isinstance(answer, str) or not answer.strip():
                raise AssistantUnavailable('The local model returned an empty answer. Try again.')
            return answer.strip()
        except HTTPError as exc:
            if exc.code == 404:
                raise AssistantUnavailable(
                    'The configured model is not available in Ollama. Follow the local model setup instructions.'
                ) from exc
            raise AssistantUnavailable('The local model service returned an error. Check Ollama and retry.') from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AssistantUnavailable(
                'The local model service is unavailable or timed out. Check Ollama and retry.'
            ) from exc


def get_answer_provider():
    provider = os.environ.get('SMARTDOC_ASSISTANT_PROVIDER', 'disabled').strip().lower()
    if provider == 'ollama':
        return OllamaProvider()
    if provider in {'', 'disabled', 'none'}:
        raise AssistantUnavailable(
            'The document assistant is not configured. Set up the optional local Ollama model; '
            'no document text has been sent anywhere.'
        )
    raise AssistantUnavailable('The configured assistant provider is unsupported.')
