#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "network_client.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void (*talk_press)(void);     /* 触摸「按住说话」按下 */
    void (*talk_release)(void);   /* 触摸松开 / 按压丢失 */
    void (*feedback)(bool is_correct);
    void (*retry)(void);          /* 错误页「返回重试」 */
} ui_callbacks_t;

esp_err_t ui_manager_init(const ui_callbacks_t *cb);

void ui_manager_show_idle(void);
void ui_manager_show_recording(void);
void ui_manager_show_uploading(void);
void ui_manager_show_result(const understand_result_t *res);
void ui_manager_show_error(const char *msg);
void ui_manager_update_recording(uint8_t level, uint32_t elapsed_ms);
void ui_manager_set_wifi(bool connected);

#ifdef __cplusplus
}
#endif