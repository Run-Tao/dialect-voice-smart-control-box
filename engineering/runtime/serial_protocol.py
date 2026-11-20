# -*- coding: utf-8 -*-
"""PC-ESP32串口协议与模拟设备。"""
ALLOWED_INTENTS = {
    'LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE',
    'ALL_OFF', 'STATUS_ALL'
}

class ProtocolError(ValueError):
    def __init__(self, code, message=''):
        self.code = code
        super().__init__(message or code)

def build_command(seq, intent_id):
    if not isinstance(seq, int) or seq < 0:
        raise ProtocolError('BAD_SEQ', 'seq必须是非负整数')
    if intent_id not in ALLOWED_INTENTS:
        raise ProtocolError('UNKNOWN_CMD', '未知命令')
    return f'CMD,{seq},{intent_id}\n'

def parse_command(line):
    line = line.strip()
    parts = line.split(',')
    if len(parts) != 3 or parts[0] != 'CMD':
        raise ProtocolError('BAD_FORMAT', '命令格式错误')
    try:
        seq = int(parts[1])
    except ValueError:
        raise ProtocolError('BAD_SEQ', 'seq不是整数')
    intent = parts[2]
    if intent not in ALLOWED_INTENTS:
        raise ProtocolError('UNKNOWN_CMD', '未知命令')
    return {'seq': seq, 'intent': intent}

def build_ack(seq, status='OK'):
    return f'ACK,{seq},{status}\n'

def build_err(seq, code):
    return f'ERR,{seq},{code}\n'

def build_state(seq, state):
    return f'STATE,{seq},LIGHT={state["light"]},CURTAIN={state["curtain"]}\n'

class MockEsp32Controller:
    def __init__(self):
        self.state = {'light': 'off', 'curtain': 'closed'}
        self.seen_seq = set()

    def handle_line(self, line):
        try:
            cmd = parse_command(line)
        except ProtocolError as e:
            return build_err('?', e.code)
        seq = cmd['seq']
        if seq in self.seen_seq:
            return build_ack(seq, 'DUP')
        self.seen_seq.add(seq)
        intent = cmd['intent']
        if intent == 'LIGHT_ON':
            self.state['light'] = 'on'
        elif intent == 'LIGHT_OFF':
            self.state['light'] = 'off'
        elif intent == 'CURTAIN_OPEN':
            self.state['curtain'] = 'open'
        elif intent == 'CURTAIN_CLOSE':
            self.state['curtain'] = 'closed'
        elif intent == 'ALL_OFF':
            self.state['light'] = 'off'
            self.state['curtain'] = 'closed'
        if intent == 'STATUS_ALL':
            return build_ack(seq, 'OK') + build_state(seq, self.state)
        return build_ack(seq, 'OK')
