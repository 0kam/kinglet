/* Written with Claude (Anthropic) under the author's direction; see paper/README.md. */
#ifndef SCENARIO_BIRD_PDM_COMMON_CONFIG_H_
#define SCENARIO_BIRD_PDM_COMMON_CONFIG_H_

#define DBG_APP_LOG 0

/* Model blob (Vela-compiled kinglet-xs_pub-jp-int8_vela.tflite) flash address.
 * Flashed with: xmodem_send.py --model "<model> 0xB7B000 0x00000"; the XIP window
 * maps flash at 0x3A000000, so the CPU reads it at 0x3AB7B000. */
#define BIRD_FLASH_ADDR 0x3AB7B000

#endif /* SCENARIO_BIRD_PDM_COMMON_CONFIG_H_ */
