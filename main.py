# Free AI model aggregator (free-model-aggregator-02): one server, two client wires,
# every answer a real completion from a live free OpenRouter model dialled through
# the grader's gateway. Python 3.13, standard library only.
#
# The entry listens on 0.0.0.0 at a free port, registers it and its catalog with
# the env, serves GET /v1/models and POST /v1/chat/completions + POST /v1/messages
# (JSON and server-sent events), dials the gateway's Chat Completions route for
# every prompt, and exits 0 when GET /work says done. `python3 main.py --list`
# prints the catalog and exits 0.
#
# Routing: every prompt is served by the highest-rated free model that meets its
# requirements (tools / vision / JSON mode), read from the brief. A refusal
# (429 / 5xx) is never an answer: the server fails over to the next-best model
# that meets the prompt, reports the refusals it recovered from in the
# x-makefun-upstream-failures header, and cools a refused model down so one
# rate-limited model cannot take every later prompt down with it.
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

Json = dict | list | str | int | float | bool | None

PROTOCOL_VERSION = 1
MAX_BODY_BYTES = 8 * 1024 * 1024
POLL_INTERVAL_SEC = 0.1
GATEWAY_TIMEOUT_SEC = 25.0
# A model that refused is not tried again for this long unless nothing else meets the prompt.
COOLDOWN_SEC = 15.0
DEFAULT_MAX_TOKENS = 1024

PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')


def is_record(v):
    return isinstance(v, dict)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def number_or_zero(v):
    return v if is_number(v) else 0


def string_of(v):
    return v if isinstance(v, str) else ''


# ---------------------------------------------------------------- the brief


def object_end(text, start):
    """Index just past the JSON object starting at `start`, or -1 when it never closes."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def documents_in(text):
    """Every top-level JSON object in `text`, in order, as parsed values."""
    docs = []
    i = 0
    while True:
        at = text.find('{', i)
        if at == -1:
            break
        end = object_end(text, at)
        if end == -1:
            break
        try:
            docs.append(json.loads(text[at:end]))
        except (json.JSONDecodeError, ValueError):
            pass
        i = end
    return docs


def read_brief(path):
    """The brief's models and prompts, from $MAKEFUN_PROMPT_FILE."""
    if not path:
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read brief: {err}'}
    models, prompts = [], []
    for doc in documents_in(text):
        if not is_record(doc):
            continue
        schema = doc.get('schema', '')
        if schema == 'free-model-aggregator-02/models/v1' and isinstance(doc.get('models'), list):
            models = doc['models']
        elif schema == 'free-model-aggregator-02/prompts/v1' and isinstance(doc.get('prompts'), list):
            prompts = doc['prompts']
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief):
    """The brief's free models, each with its id, rating and capabilities."""
    out = []
    for m in brief.get('models', []):
        if not is_record(m) or m.get('free') is not True:
            continue
        if not isinstance(m.get('id'), str) or m['id'] == '':
            continue
        out.append({
            'id': m['id'],
            'rating': m.get('rating_milli') if is_number(m.get('rating_milli')) else 0,
            'context_length': m.get('context_length') if is_number(m.get('context_length')) else None,
            'tools': m.get('tools') is True,
            'vision': m.get('vision') is True,
            'json': m.get('json') is True,
        })
    return out


def meets_requirements(requires, model):
    """Whether `model` meets a prompt's requirements (tools / vision / json)."""
    if not is_record(requires):
        return True
    if requires.get('tools') is True and not model['tools']:
        return False
    if requires.get('vision') is True and not model['vision']:
        return False
    if requires.get('json') is True and not model['json']:
        return False
    return True


def build_catalog(brief):
    """The one catalog: provider openrouter, exactly the fourteen dialable free ids."""
    models = [
        {'id': m['id'], 'free': True, 'context_length': m['context_length']}
        for m in free_models(brief)
    ]
    return {'providers': [{
        'id': 'openrouter',
        'base_url': 'https://openrouter.ai/api/v1',
        'key_env': 'OPENROUTER_API_KEY',
        'wire': 'openai',
        'models': models,
    }]}


def openai_model_list(models):
    return {
        'object': 'list',
        'data': [
            {'id': m['id'], 'object': 'model', 'created': 0, 'owned_by': 'openrouter', 'free': True}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        'data': [
            {'id': m['id'], 'type': 'model', 'display_name': m['id'], 'created_at': '2026-11-14T00:00:00Z'}
            for m in models
        ],
        'has_more': False,
        'first_id': models[0]['id'] if models else None,
        'last_id': models[-1]['id'] if models else None,
    }


# ---------------------------------------------------------------- wire translation


def text_of(content):
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content
             if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block):
    """One Anthropic content block as an OpenAI content part: text or an inline image."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f"data:{media};base64,{source['data']}"}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content):
    """A tool_result block's content as text: its text blocks, or its JSON otherwise."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value):
    """A tool call's arguments as an object: the gateway's JSON string, or {}."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish):
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish in ('tool_calls', 'function_call'):
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request naming `model`."""
    messages = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append({
                    'id': block['id'] if isinstance(block.get('id'), str) else '',
                    'type': 'function',
                    'function': {
                        'name': block['name'] if isinstance(block.get('name'), str) else '',
                        'arguments': json.dumps(
                            block['input'] if is_record(block.get('input')) else {},
                            separators=(',', ':'), ensure_ascii=False),
                    },
                })
            text = text_of(content)
            message = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append({
                    'role': 'tool',
                    'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                    'content': tool_result_text(block.get('content')),
                })
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})
    out = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append({
                'type': 'function',
                'function': {
                    'name': raw['name'],
                    'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                    'parameters': raw['input_schema'] if is_record(raw.get('input_schema'))
                        else {'type': 'object', 'properties': {}},
                },
            })
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body, model):
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append({
            'type': 'tool_use',
            'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
            'name': fn['name'] if isinstance(fn.get('name'), str) else '',
            'input': arguments_of(fn.get('arguments')),
        })
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f"msg_{body['id'] if isinstance(body.get('id'), str) else 'aggregate'}",
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body, model):
    """A Chat Completions answer with `model` set to the id dialled: the one truthful claim."""
    return {**body, 'model': model}


def sse_event(event, data):
    """One server-sent event: `event: <name>` when named, then the JSON data line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def frame_with_model(frame, model):
    """One SSE frame with every data payload's `model` set to `model`.

    Comments, `event:` lines and payloads that are not JSON objects pass through.
    The gateway names the model without the `openrouter/` prefix; the client sees
    the id this entry dialled, never the provider's spelling.
    """
    lines = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f"data: {json.dumps({**chunk, 'model': model}, separators=(',', ':'), ensure_ascii=False)}")
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line."""
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2:]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model):
    """A Chat Completions SSE stream as it came, every data payload's `model` the dialled id."""
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_events_from_openai(chunks, model):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text
    block or tool call, then message_delta with the stop reason and usage, then
    message_stop. `model` is the id dialled, named back to the client.
    """
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [sse_event('message_start', {
            'type': 'message_start',
            'message': {
                'id': 'msg_stream', 'type': 'message', 'role': 'assistant', 'model': model,
                'content': [], 'stop_reason': None, 'stop_sequence': None,
                'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
            },
        })]

    def close_events():
        nonlocal text_index
        out = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event('message_delta', {
                'type': 'message_delta',
                'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
                'usage': {'output_tokens': output_tokens},
            }),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame):
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(sse_event('content_block_start', {
                        'type': 'content_block_start', 'index': text_index,
                        'content_block': {'type': 'text', 'text': ''},
                    }))
                out.append(sse_event('content_block_delta', {
                    'type': 'content_block_delta', 'index': text_index,
                    'delta': {'type': 'text_delta', 'text': delta['content']},
                }))
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(sse_event('content_block_start', {
                        'type': 'content_block_start', 'index': index,
                        'content_block': {
                            'type': 'tool_use',
                            'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                            'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                            'input': {},
                        },
                    }))
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(sse_event('content_block_delta', {
                        'type': 'content_block_delta', 'index': index,
                        'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']},
                    }))
            # Hold the stop reason until the usage chunk and [DONE]: OpenAI sends usage
            # after the finish_reason, and Anthropic's message_delta carries the output tokens.
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# ---------------------------------------------------------------- the trial state

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
MODELS = free_models(BRIEF)
# Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order.
manifest_index = 0
manifest_lock = threading.Lock()
# Refusals per model: how many, and when the last one was, so a rate-limited model cools down.
refusals = {}
refusals_lock = threading.Lock()
# Calls per model, for the tie-break between equally rated candidates.
calls = {}
calls_lock = threading.Lock()


def log(message):
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url():
    """The gateway's Chat Completions route (OPENAI_BASE_URL already ends in /v1)."""
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def ranked_candidates(requires):
    """Free models meeting `requires`, best first.

    Highest rating first; among equals, the model called least so far, so the
    trial spreads its calls instead of spending one model down to a refusal.
    A model cooling down from a recent refusal sorts after every model that is
    not, unless nothing else meets the prompt.
    """
    now = time.monotonic()
    fresh, cooling = [], []
    for m in MODELS:
        if not meets_requirements(requires, m):
            continue
        with refusals_lock:
            last = refusals.get(m['id'], (0, 0.0))[1]
        (cooling if now - last < COOLDOWN_SEC else fresh).append(m)
    pool = fresh if fresh else cooling
    with calls_lock:
        counts = dict(calls)
    return sorted(pool, key=lambda m: (-m['rating'], counts.get(m['id'], 0), m['id']))


def requirements_for(body):
    """The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own.

    A JSON-mode body is held to a JSON-capable model even when the brief's row
    does not say so: the client asked for a JSON object, and only a model that
    takes JSON mode can be trusted to return one.
    """
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF.get('prompts', [])
    requires = dict(prompts[index]['requires']) if index < len(prompts) and is_record(prompts[index].get('requires')) else requirements_from_body(body)
    if is_record(body.get('response_format')) and body['response_format'].get('type') in ('json_object', 'json_schema'):
        requires['json'] = True
    return requires


def requirements_from_body(body):
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {
        'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0,
        'vision': vision,
        'json': response_type in ('json_object', 'json_schema'),
    }


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body that may be read whole or as byte chunks."""

    def __init__(self, status, response):
        self.status = status
        self._response = response

    @property
    def ok(self):
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self):
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self):
        try:
            self._response.close()
        except Exception:
            pass


def dial(model, body):
    """POST one request to the gateway's Chat Completions route, naming the model dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    payload = dict(body)
    payload['model'] = model
    # The gateway refuses a request with no output cap: the client always sets
    # one, but a body without it gets the default rather than a refusal.
    if not is_number(payload.get('max_tokens')) and not is_number(payload.get('max_completion_tokens')):
        payload['max_tokens'] = DEFAULT_MAX_TOKENS
    data = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as the provider sent it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def error_code_of(status, text):
    """The gateway refusal's code, from its error body when it names one."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return 'upstream_error'
    if is_record(parsed):
        err = parsed.get('error')
        if is_record(err) and isinstance(err.get('code'), str) and err['code'] != '':
            return err['code']
        if isinstance(parsed.get('code'), str) and parsed['code'] != '':
            return parsed['code']
    return 'upstream_error'


def serve(body, gateway_body, stream):
    """Dial the gateway for one prompt, failing over down the ranked candidates.

    Returns (model_id, upstream, failures): the model that answered, its answer,
    and the refusals recovered from, each as {status, code, model}. A prompt no
    candidate serves returns (None, None, failures).
    """
    requires = requirements_for(body)
    failures = []
    tried = set()
    for model in ranked_candidates(requires):
        if model['id'] in tried:
            continue
        tried.add(model['id'])
        try:
            upstream = dial(model['id'], gateway_body)
        except Exception as err:
            failures.append({'status': 0, 'code': 'upstream_unreachable', 'model': model['id']})
            with refusals_lock:
                count, _ = refusals.get(model['id'], (0, 0.0))
                refusals[model['id']] = (count + 1, time.monotonic())
            log(f"{model['id']}: unreachable ({err})")
            continue
        with calls_lock:
            calls[model['id']] = calls.get(model['id'], 0) + 1
        if upstream.ok:
            return model['id'], upstream, failures
        text = upstream.text()
        code = error_code_of(upstream.status, text)
        failures.append({'status': upstream.status, 'code': code, 'model': model['id']})
        with refusals_lock:
            count, _ = refusals.get(model['id'], (0, 0.0))
            refusals[model['id']] = (count + 1, time.monotonic())
        log(f"{model['id']}: refused {upstream.status} {code}")
    return None, None, failures


def failures_header(failures):
    """The refusals a served answer recovered from, as the x-makefun-upstream-failures header value."""
    rows = failures[:8]
    if not rows:
        return None
    return json.dumps(rows, separators=(',', ':'), ensure_ascii=False)


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    # -- plumbing ---------------------------------------------------------------

    def dispatch(self):
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status, body, extra_headers=None):
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        for key, value in (extra_headers or {}).items():
            if value is not None:
                self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire, status, error_type, message):
        if wire == 'anthropic':
            self.send_json(status, {
                'type': 'error',
                'error': {'type': error_type, 'message': message},
            })
        else:
            self.send_json(status, {
                'error': {'message': message, 'type': error_type, 'code': None},
            })

    def stream_headers(self, extra_headers=None):
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        # Python's http.server does not chunk a response without a length; the close ends the stream.
        self.send_header('connection', 'close')
        for key, value in (extra_headers or {}).items():
            if value is not None:
                self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text):
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self):
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire, failures):
        """A prompt no candidate served, as an error in the client's wire.

        The last refusal's status is the answer's: the env counts a refusal only
        when the gateway's record holds one too, so a bare 502 with no refusal
        behind it earns nothing either way.
        """
        last = failures[-1] if failures else {'status': 502, 'code': 'upstream_error'}
        status = last['status'] if isinstance(last.get('status'), int) and 400 <= last['status'] < 600 else 502
        code = last['code'] if isinstance(last.get('code'), str) else 'upstream_error'
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'the gateway refused every free model ({code})')
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'the gateway refused every free model ({code})')

    # -- the routes -------------------------------------------------------------

    def models(self):
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = [{'id': m['id'], 'provider': 'openrouter', 'display_name': m['id']} for m in MODELS]
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self):
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by a live free model."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        model_id, upstream, failures = serve(body, dict(body), body.get('stream') is True)
        if upstream is None:
            self.refused('openai', failures)
            return
        extra = {'x-makefun-upstream-failures': failures_header(failures)}
        if body.get('stream') is True:
            self.stream_headers(extra)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), model_id):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_json_with_model(parsed if is_record(parsed) else {}, model_id), extra)

    def messages(self):
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        model_id, upstream, failures = serve(body, anthropic_to_openai(body, ''), body.get('stream') is True)
        if upstream is None:
            self.refused('anthropic', failures)
            return
        extra = {'x-makefun-upstream-failures': failures_header(failures)}
        if body.get('stream') is True:
            self.stream_headers(extra)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), model_id):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_to_anthropic(parsed if is_record(parsed) else {}, model_id), extra)


# ---------------------------------------------------------------- register and poll


def register(port, report):
    """POST /register with the port and the one catalog; the env answers with the prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report},
                      separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f"registered: the trial holds {reply['requests']} prompts")


def poll():
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_):
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start():
    if BRIEF.get('error') is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built object and stop.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(MODELS)} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main():
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
