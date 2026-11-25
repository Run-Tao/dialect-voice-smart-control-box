#include <Arduino.h>
#include "board_config.h"

namespace {

constexpr uint8_t kServoResolutionBits = 16;
constexpr uint32_t kServoFrequencyHz = 50;
constexpr uint32_t kServoPulseMinUs = 500;
constexpr uint32_t kServoPulseMaxUs = 2500;
constexpr uint32_t kServoPeriodUs = 20000;

struct DeviceState {
  bool light_on = false;
  bool curtain_open = false;
};

struct SeenSequences {
  uint32_t values[DV_SEEN_SEQ_SLOTS] = {};
  uint8_t count = 0;
  uint8_t next = 0;
};

DeviceState state;
SeenSequences seen_sequences;
String line_buffer;
bool servo_ready = false;
int servo_channel = 0;

bool isAllowedIntent(const String &intent) {
  return intent == "LIGHT_ON" || intent == "LIGHT_OFF" ||
         intent == "CURTAIN_OPEN" || intent == "CURTAIN_CLOSE" ||
         intent == "ALL_OFF" || intent == "STATUS_ALL";
}

bool seenSequence(uint32_t seq) {
  for (uint8_t i = 0; i < seen_sequences.count; ++i) {
    if (seen_sequences.values[i] == seq) {
      return true;
    }
  }
  return false;
}

void rememberSequence(uint32_t seq) {
  seen_sequences.values[seen_sequences.next] = seq;
  seen_sequences.next = (seen_sequences.next + 1) % DV_SEEN_SEQ_SLOTS;
  if (seen_sequences.count < DV_SEEN_SEQ_SLOTS) {
    ++seen_sequences.count;
  }
}

void setServoAngle(int angle) {
  angle = constrain(angle, 0, 180);
  const uint32_t pulse_us =
      kServoPulseMinUs + (kServoPulseMaxUs - kServoPulseMinUs) * angle / 180U;
  const uint32_t duty =
      (pulse_us * ((1UL << kServoResolutionBits) - 1UL) + kServoPeriodUs / 2U) /
      kServoPeriodUs;

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  if (!servo_ready) {
    servo_ready = ledcAttach(DV_CURTAIN_SERVO_PIN, kServoFrequencyHz,
                             kServoResolutionBits);
  }
  if (servo_ready) {
    ledcWrite(DV_CURTAIN_SERVO_PIN, duty);
  }
#else
  if (!servo_ready) {
    ledcSetup(servo_channel, kServoFrequencyHz, kServoResolutionBits);
    ledcAttachPin(DV_CURTAIN_SERVO_PIN, servo_channel);
    servo_ready = true;
  }
  ledcWrite(servo_channel, duty);
#endif
}

void setLight(bool on) {
  state.light_on = on;
  const uint8_t level =
      DV_LIGHT_ACTIVE_HIGH ? (on ? HIGH : LOW) : (on ? LOW : HIGH);
  digitalWrite(DV_LIGHT_PIN, level);
}

void blinkStatus() {
  digitalWrite(DV_STATUS_LED_PIN, HIGH);
  delay(20);
  digitalWrite(DV_STATUS_LED_PIN, LOW);
}

void sendAck(uint32_t seq, const char *status = "OK") {
  Serial.printf("ACK,%lu,%s\n", static_cast<unsigned long>(seq), status);
}

void sendError(uint32_t seq, const char *code) {
  Serial.printf("ERR,%lu,%s\n", static_cast<unsigned long>(seq), code);
}

void sendState(uint32_t seq) {
  Serial.printf("STATE,%lu,LIGHT=%s,CURTAIN=%s\n",
                static_cast<unsigned long>(seq),
                state.light_on ? "on" : "off",
                state.curtain_open ? "open" : "closed");
}

void applyIntent(const String &intent) {
  if (intent == "LIGHT_ON") {
    setLight(true);
  } else if (intent == "LIGHT_OFF") {
    setLight(false);
  } else if (intent == "CURTAIN_OPEN") {
    state.curtain_open = true;
    setServoAngle(DV_SERVO_OPEN_ANGLE);
  } else if (intent == "CURTAIN_CLOSE") {
    state.curtain_open = false;
    setServoAngle(DV_SERVO_CLOSE_ANGLE);
  } else if (intent == "ALL_OFF") {
    setLight(false);
    state.curtain_open = false;
    setServoAngle(DV_SERVO_CLOSE_ANGLE);
  }
}

bool parseUnsigned(const String &text, uint32_t &value) {
  if (text.length() == 0) {
    return false;
  }
  uint64_t parsed = 0;
  for (size_t i = 0; i < text.length(); ++i) {
    const char c = text[i];
    if (c < '0' || c > '9') {
      return false;
    }
    parsed = parsed * 10U + static_cast<uint8_t>(c - '0');
    if (parsed > 0xFFFFFFFFULL) {
      return false;
    }
  }
  value = static_cast<uint32_t>(parsed);
  return true;
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) {
    return;
  }

  const int first = line.indexOf(',');
  const int second = first < 0 ? -1 : line.indexOf(',', first + 1);
  if (first <= 0 || second <= first + 1 || line.indexOf(',', second + 1) >= 0) {
    sendError(0, "BAD_FORMAT");
    return;
  }

  const String head = line.substring(0, first);
  const String seq_text = line.substring(first + 1, second);
  const String intent = line.substring(second + 1);
  if (head != "CMD") {
    sendError(0, "BAD_FORMAT");
    return;
  }

  uint32_t seq = 0;
  if (!parseUnsigned(seq_text, seq)) {
    sendError(0, "BAD_SEQ");
    return;
  }
  if (!isAllowedIntent(intent)) {
    sendError(seq, "UNKNOWN_CMD");
    return;
  }
  if (seenSequence(seq)) {
    sendAck(seq, "DUP");
    return;
  }

  rememberSequence(seq);
  applyIntent(intent);
  blinkStatus();
  sendAck(seq);
  if (intent == "STATUS_ALL") {
    sendState(seq);
  }
}

void processSerial() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      handleLine(line_buffer);
      line_buffer = "";
      continue;
    }
    if (line_buffer.length() >= DV_MAX_LINE_LEN) {
      line_buffer = "";
      sendError(0, "LINE_TOO_LONG");
      continue;
    }
    line_buffer += c;
  }
}

}  // namespace

void setup() {
  pinMode(DV_LIGHT_PIN, OUTPUT);
  pinMode(DV_STATUS_LED_PIN, OUTPUT);
  digitalWrite(DV_STATUS_LED_PIN, LOW);
  setLight(false);
  setServoAngle(DV_SERVO_CLOSE_ANGLE);

  Serial.begin(DV_SERIAL_BAUD);
  const uint32_t started = millis();
  while (!Serial && millis() - started < 3000U) {
    delay(10);
  }
  line_buffer.reserve(DV_MAX_LINE_LEN);
  Serial.printf("\nBOOT,ESP32S3_DIALECTVOICE,READY,version=%s\n",
                DV_FIRMWARE_VERSION);
  sendState(0);
}

void loop() {
  processSerial();
  delay(2);
}
