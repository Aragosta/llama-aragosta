#!/usr/bin/env python3
"""Short persistent-prefix A/B trace; server needs --slot-save-path."""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from agent_load import CHARS_PER_TOKEN, stream_chat, summarize, synth_tools

url = sys.argv[1]
system = 'You are a coding agent. Tools:\n' + synth_tools(int(4000 * CHARS_PER_TOKEN))

def request(sid, prompt, tokens, system_text=system):
    messages = [{'role': 'system', 'content': system_text}, {'role': 'user', 'content': prompt}]
    r = stream_chat(url, messages, tokens, {'cache_prompt': True})
    st = r['stamps']
    assert st, 'server returned no content tokens'
    tm = r['timings'] or {}
    return {'session': sid, 'ttft': st[0] - r['t0'], 'n_tok': len(st),
            'gaps': [b-a for a,b in zip(st, st[1:])], 't0': r['t0'],
            't_end': st[-1], 'prompt_n': tm.get('prompt_n'),
            'cache_n': tm.get('cache_n'), 'text': r['text']}

def erase_slots():
    for sid in range(3):
        req = urllib.request.Request(url + f'/slots/{sid}?action=erase', data=b'', method='POST')
        with urllib.request.urlopen(req, timeout=20):
            pass

if len(sys.argv) > 2 and sys.argv[2] in ('--retention', '--pressure'):
    pressure = sys.argv[2] == '--pressure'
    systems = ['You are a coding agent. Tools:\n' + synth_tools(int((4000 if pressure else 1000) * CHARS_PER_TOKEN), seed=7+i)
               for i in range(3)]
    for sid, text in enumerate(systems):
        request(sid, f'Inspect repository {sid}.', 8, text)
        erase_slots()
    if pressure:
        print(json.dumps({'parents': 3, 'errors': 0}))
    else:
        child = request(2, 'Inspect another module.', 16, systems[2])
        assert (child['cache_n'] or 0) > 500, child
        print(json.dumps({'cache_n': child['cache_n'], 'prompt_n': child['prompt_n']}))
else:
    request(-1, 'Inspect the repository and list its main modules.', 16)
    erase_slots()
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(lambda sid: request(sid, f'Inspect module {sid} and summarize its purpose.', 64), range(2)))
    print(json.dumps({'summary': summarize(rows), 'requests': rows}))
