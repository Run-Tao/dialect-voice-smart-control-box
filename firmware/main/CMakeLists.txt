# 第一阶段：仅 board_bsp + 自检 main，确保「能编译、能上电、能扫到器件」
# 后续模块（audio_recorder / network_client / ui_manager）就位后，
# 把对应 .c 加进 SRCS 即可，无需改其它构建配置。
idf_component_register(
    SRCS
        "main.c"
        "board_bsp.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        driver
        esp_driver_i2c
        esp_driver_i2s
        esp_driver_gpio
        esp_driver_spi
        esp_driver_sdmmc
        esp_driver_uart
        esp_timer
        nvs_flash
        fatfs
        sdmmc
        spi_flash
        esp_psram
        esp_netif
        esp_wifi
        esp_http_client
        json
        app_update
)
