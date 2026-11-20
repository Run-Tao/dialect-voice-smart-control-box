#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_RECORDER_MAX_SECONDS 10   /* 单次录音上限，到时自动停止 */

/* 录音落盘完成回调：在 SD 写任务上下文调用，禁止阻塞/耗时操作。
 * err != ESP_OK 时 wav_path/sample_id 仍给出（便于界面提示），文件可能不完整。 */
typedef void (*audio_recorder_saved_cb)(const char *wav_path, const char *sample_id,
                                        uint32_t duration_ms, esp_err_t err);

esp_err_t audio_recorder_init(audio_recorder_saved_cb cb);

/* 开始录音：I2S 采集在独立高优先级任务（core1）中进行；
 * SD 卡挂载失败 / 正在录音 / 上一次尚未落盘 时返回 ESP_ERR_INVALID_STATE。 */
esp_err_t audio_recorder_start(void);

/* 请求停止（幂等）；采集任务收尾后由 SD 写任务落盘并触发 saved 回调 */
esp_err_t audio_recorder_stop(void);

bool audio_recorder_is_recording(void);
bool audio_recorder_is_saving(void);
uint32_t audio_recorder_elapsed_ms(void);
uint8_t audio_recorder_level(void);   /* 0~100，供 UI 电平条 */

#ifdef __cplusplus
}
#endif