import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serial_protocol import (
    build_command, parse_command, ProtocolError, MockEsp32Controller
)

class TestSerialProtocol(unittest.TestCase):
    def test_valid_command(self):
        line = build_command(1, 'LIGHT_ON')
        self.assertEqual(line, 'CMD,1,LIGHT_ON\n')
        self.assertEqual(parse_command(line)['intent'], 'LIGHT_ON')

    def test_unknown_command(self):
        with self.assertRaises(ProtocolError):
            build_command(1, 'FAN_ON')
        with self.assertRaises(ProtocolError):
            parse_command('CMD,2,FAN_ON\n')

    def test_mock_ack_and_state(self):
        dev = MockEsp32Controller()
        self.assertEqual(dev.handle_line('CMD,1,LIGHT_ON\n'), 'ACK,1,OK\n')
        self.assertEqual(dev.state['light'], 'on')
        response = dev.handle_line('CMD,2,STATUS_ALL\n')
        self.assertIn('ACK,2,OK', response)
        self.assertIn('LIGHT=on', response)

    def test_duplicate_seq(self):
        dev = MockEsp32Controller()
        dev.handle_line('CMD,3,LIGHT_OFF\n')
        self.assertEqual(dev.handle_line('CMD,3,LIGHT_OFF\n'), 'ACK,3,DUP\n')

    def test_malformed(self):
        dev = MockEsp32Controller()
        self.assertEqual(dev.handle_line('BADLINE\n'), 'ERR,?,BAD_FORMAT\n')
        self.assertEqual(dev.handle_line('CMD,x,LIGHT_ON\n'), 'ERR,?,BAD_SEQ\n')

if __name__ == '__main__':
    unittest.main()
