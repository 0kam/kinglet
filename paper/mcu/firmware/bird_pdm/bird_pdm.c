/* Written with Claude (Anthropic) under the author's direction; see paper/README.md. */
/* Grove Vision AI V2 bird classifier: PDM mic (measured 31,250 Hz) -> 160,000-sample windows -> Kinglet XS.
 * Based on the SDK kws_pdm_record scenario app (PDM DMA ring + TFLM/Ethos-U55; MIT licence, Himax).
 * On boot it runs a golden check: the LCG golden window through the full on-device
 * pipeline must reproduce the host int8 probs (bird_golden_probs, mel_assets.c). */

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "WE2_device.h"
#include "board.h"
#include "xprintf.h"
#include "console_io.h"
#include "WE2_core.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#include "spi_eeprom_comm.h"
#include "hx_drv_spi.h"

#include "hx_drv_pdm_rx.h"
#include "hx_drv_uart.h"

#include "bird_pdm.h"
#include "cvapp_bird.h"
#include "common_config.h"
#include "mel_assets.h"

/* DMA block limit is <8192 bytes. One lli transfer = 2 blocks = 6400 samples = 0.2048 s at
 * the measured 31250 Hz; ring of 10 buffers = 2 s slack (inference takes ~0.47 s = 2 transfers). */
#define BLK_BYTES    6400 /* not 8000: the DMA rounds blocks up to 8192 B (measured), 6400 stays exact */
#define BLK_NUM      2
#define XFER_SAMPLES (BLK_NUM * BLK_BYTES / 2) /* 6400 samples = 0.2 s */
#define NUM_BUFF     10
#define WIN_SAMPLES  160000
#define XFERS_PER_WIN (WIN_SAMPLES / XFER_SAMPLES) /* 25 */

static int16_t audio_buf[NUM_BUFF][XFER_SAMPLES];
static int16_t window_buf[WIN_SAMPLES];

static PDM_DEV_INFO pdm_dev_info;
/* Completed DMA transfers since measure_pdm_rate started the chain. Transfer k lands in
 * audio_buf[k % NUM_BUFF]; the DMA is always writing audio_buf[xfers_done % NUM_BUFF]. */
static volatile uint32_t xfers_done = 0;

static void app_pdm_dma_rx_cb(void)
{
    xfers_done++;
    hx_drv_pdm_dma_lli_transfer((void *)audio_buf[xfers_done % NUM_BUFF], BLK_NUM, BLK_BYTES, 0);
}

static void app_pdm_setting(void)
{
    pdm_dev_info.pdm_cfg.reg_addr = HIMAX_PDM_BASE_ADDR;
    pdm_dev_info.pdm_cfg.rx_fifo_threshold = 5;
    /* Measured on hardware: the enum rate is doubled per -1 OSR step from 8, i.e. the
     * SDK comment "6: 32KHz" assumes sample_rate stays FREQ_16K. FREQ_32K + OSR 6 ran at
     * 64 kHz (everything pitched down an octave). FREQ_16K + OSR 6 measures ~32 kHz. */
    pdm_dev_info.pdm_cfg.oversampling_ratio = 6;
    pdm_dev_info.pdm_cfg.cic_stages = 0;
    pdm_dev_info.pdm_cfg.cic_delay = 0;
    pdm_dev_info.pdm_cfg.dc_removal = 6;
    pdm_dev_info.pdm_cfg.bit_range_shift = 5;
    pdm_dev_info.pdm_cfg.data_in_0_en = PDM_DATA_IN_ENABLE;
    pdm_dev_info.pdm_cfg.data_in_1_en = PDM_DATA_IN_DISABLE;
    pdm_dev_info.pdm_cfg.data_in_2_en = PDM_DATA_IN_DISABLE;
    pdm_dev_info.pdm_cfg.data_in_3_en = PDM_DATA_IN_DISABLE;
    pdm_dev_info.pdm_cfg.capture_channel = PDM_CAPTURE_CHANNEL_LEFT_ONLY;
    pdm_dev_info.pdm_cfg.dma_ch = PDM_USE_DMA2_CH_0;
    pdm_dev_info.pdm_cfg.pdm_clk_src = PDM_CLKSRC_LSCREF;
    pdm_dev_info.pdm_cfg.sample_rate = PDM_PCM_FREQ_16K;
    pdm_dev_info.pdm_rx_cb = app_pdm_dma_rx_cb;
    pdm_dev_info.pdm_err_cb = NULL;
}

#define CPU_CLK (0xffffff + 1)

/* 1: dump each window's int8 embedding feature (1280 B as hex). */
#ifndef BIRD_DUMP_EMB
#define BIRD_DUMP_EMB 0
#endif

/* 1: host PCM test mode (mcuboard_v1). After the golden checks the board does not record:
 * it receives 160000-sample int16 windows over the console UART in place of the PDM stream,
 * runs the unchanged cv_bird_run (log-mel + network) on them and prints the full outputs.
 * Build with `make BIRD_PCM_TEST=1` (bird_pdm.mk); the production build leaves it 0. */
#ifndef BIRD_PCM_TEST
#define BIRD_PCM_TEST 0
#endif

/* Boot-time rate check of the ACTIVE config: time 5 DMA transfers (32000 samples).
 * The driver's real output rate does not follow the SDK comments (FREQ_32K + OSR 6 measured
 * 64 kHz), and re-initing the driver per combo hangs, so measure once with the real config. */
static void measure_pdm_rate(uint32_t freq)
{
    xfers_done = 0;
    hx_drv_pdm_dma_lli_transfer((void *)audio_buf[0], BLK_NUM, BLK_BYTES, 0);
    while (xfers_done == 0) {}
    uint32_t t1, l1, t2, l2;
    SystemGetTick(&t1, &l1);
    while (xfers_done < 6) {}
    SystemGetTick(&t2, &l2);
    uint32_t cyc = (l2 - l1) * CPU_CLK + (t1 - t2);
    uint32_t us = cyc / (freq / 1000000);
    xprintf("[rate] %d samples in %u us -> %u Hz\n", 5 * XFER_SAMPLES, (unsigned)us,
            (unsigned)((uint64_t)5 * XFER_SAMPLES * 1000000 / us));
}

static void print_result(const char *tag, const bird_result_t *r, uint32_t freq)
{
    xprintf("%s mel %u cyc (%u ms) npu+cpu %u cyc (%u ms) | top5:", tag,
            (unsigned)r->mel_cycles, (unsigned)(r->mel_cycles / (freq / 1000)),
            (unsigned)r->npu_cycles, (unsigned)(r->npu_cycles / (freq / 1000)));
    for (int k = 0; k < 5; k++)
        xprintf(" %s:%d", bird_codes[r->top_idx[k]], (int)(r->top_prob[k] * 1000.0f));
    xprintf(" (milli-probs)\n");
}

/* Same LCG as gen_mel_assets.golden_reference: the golden audio is synthesized, not stored. */
static void bird_golden_fill(int16_t *buf)
{
    uint32_t x = 1;
    for (int i = 0; i < WIN_SAMPLES; i++) {
        x = (uint32_t)((1103515245u * x + 12345u) & 0x7FFFFFFFu);
        buf[i] = (int16_t)((int32_t)((x >> 16) & 0xFFFu) - 2048);
    }
}

static int golden_check(uint32_t freq)
{
    bird_result_t res;
    bird_golden_fill(window_buf);
    if (cv_bird_run(window_buf, &res) != 0)
        return -1;
    print_result("[golden]", &res, freq);
    const float *probs = cv_bird_probs();
    float max_d = 0.0f;
    int ref_top = 0, dev_top = 0;
    for (int c = 0; c < BIRD_N_CLASSES; c++) {
        float d = probs[c] - bird_golden_probs[c];
        if (d < 0) d = -d;
        if (d > max_d) max_d = d;
        if (bird_golden_probs[c] > bird_golden_probs[ref_top]) ref_top = c;
        if (probs[c] > probs[dev_top]) dev_top = c;
    }
    xprintf("[golden] host top1 %s:%d, device top1 %s:%d, max |dp| %d milli\n",
            bird_codes[ref_top], (int)(bird_golden_probs[ref_top] * 1000.0f),
            bird_codes[dev_top], (int)(probs[dev_top] * 1000.0f),
            (int)(max_d * 1000.0f));
    const int pass = (ref_top == dev_top) && (max_d < 0.05f);
    xprintf("[golden] %s\n", pass ? "PASS" : "FAIL");
    const int8_t *e;
    float es;
    int ez, n = cv_bird_emb(&e, &es, &ez);
    if (n) { /* compare by eye against the host int8 embedding of the LCG window */
        xprintf("[golden emb] n %d scale_nano %d zp %d first16 ", n, (int)(es * 1e9f), ez);
        for (int i = 0; i < 16; i++)
            xprintf("%02x", (uint8_t)e[i]);
        xprintf("\n");
    }
    return pass ? 0 : -1;
}

#if BIRD_PCM_TEST
/* Protocol (host side: mcu/board_pcm_test.py). Per window the board prints
 * "[pcm] ready"; the host sends "PCM" + mode byte ('0', or '1' to also dump the int8 model
 * input) + 320000 bytes of little-endian int16. The board answers with the FNV-1a hash of the
 * bytes it received (the host resends on mismatch), the cycle counts, the 318 probabilities as
 * IEEE-754 bit patterns and the 1280-byte int8 feature, all in hex. Bytes before "PCM" are
 * skipped, so a host that pads after a lost byte resynchronizes on the next window. */
static void pcm_test_loop(void)
{
    DEV_UART *uart = hx_drv_uart_get_dev(USE_DW_UART_0);
    bird_result_t res;
    for (int win_no = 0;; win_no++) {
        xprintf("[pcm] ready\n");
        int matched = 0;
        while (matched < 3) {
            const int c = console_getchar();
            matched = (c == "PCM"[matched]) ? matched + 1 : (c == 'P');
        }
        const int mode = console_getchar();
        uart->uart_read(window_buf, sizeof(window_buf));
        uint32_t h = 2166136261u;
        const uint8_t *b = (const uint8_t *)window_buf;
        for (uint32_t i = 0; i < sizeof(window_buf); i++)
            h = (h ^ b[i]) * 16777619u;
        if (cv_bird_run(window_buf, &res) != 0) {
            xprintf("[pcmres %d] run failed\n", win_no);
            continue;
        }
        xprintf("[pcmres %d] fnv %08x mel_cyc %u npu_cyc %u\n", win_no, (unsigned)h,
                (unsigned)res.mel_cycles, (unsigned)res.npu_cycles);
        const float *probs = cv_bird_probs();
        xprintf("[probs] ");
        for (int c = 0; c < BIRD_N_CLASSES; c++) {
            uint32_t bits;
            memcpy(&bits, &probs[c], 4);
            xprintf("%08x", (unsigned)bits);
        }
        xprintf("\n");
        const int8_t *e;
        float es;
        int ez, n = cv_bird_emb(&e, &es, &ez);
        xprintf("[emb] ");
        for (int i = 0; i < n; i++)
            xprintf("%02x", (uint8_t)e[i]);
        xprintf("\n");
        if (mode == '1') {
            const int8_t *in = cv_bird_input();
            xprintf("[mel] ");
            for (int i = 0; i < BIRD_N_MELS * BIRD_N_FRAMES; i++)
                xprintf("%02x", (uint8_t)in[i]);
            xprintf("\n");
        }
    }
}
#endif

int bird_pdm_app(void)
{
    uint32_t freq = 0;

    /* PDM pins: PB9 CLK, PB10 DATA0 */
    hx_drv_scu_set_PB9_pinmux(SCU_PB9_PINMUX_PDM_CLK_13, 1);
    hx_drv_scu_set_PB10_pinmux(SCU_PB10_PINMUX_PDM_DATA0_13, 1);
    app_pdm_setting();

    if (hx_drv_pdm_init(&pdm_dev_info) != PDM_NO_ERROR)
        return -1;

    hx_drv_swreg_aon_get_pllfreq(&freq);
    xprintf("bird_pdm: pll %u Hz\n", (unsigned)freq);

    /* XIP so the model blob at BIRD_FLASH_ADDR is readable */
    hx_lib_spi_eeprom_open(USE_DW_SPI_MST_Q);
    hx_lib_spi_eeprom_enable_XIP(USE_DW_SPI_MST_Q, true, FLASH_QUAD, true);

#ifndef CPU_24MHZ_VERSION
    EPII_set_memory(0x56100030, 0x4037);
    EPII_set_memory(0x56100034, 0x0);
    EPII_set_memory(0x56100038, 0xc1b8);
#endif
    hx_drv_swreg_aon_get_pllfreq(&freq);

    measure_pdm_rate(freq);
    if (cv_bird_init(true, true, BIRD_FLASH_ADDR) != 0) {
        xprintf("cv_bird_init failed\n");
        return -1;
    }

    /* golden twice: the first run includes cold caches */
    golden_check(freq);
    if (golden_check(freq) != 0)
        xprintf("golden check FAILED - results below are not trustworthy\n");
#if BIRD_PCM_TEST
    pcm_test_loop(); /* does not return; the PDM chain keeps running but is not read */
#endif

    xprintf("bird_pdm: recording 5 s windows\n");
    /* the DMA chain is already running (started by measure_pdm_rate, self-chained in the rx cb) */

    bird_result_t res;
    /* consume every completed transfer in order; start from the one being written now so
     * the first window is fresh audio, not ring contents left over from the golden checks */
    uint32_t consumed = xfers_done;
    int filled = 0, win_no = 0;
    while (1) {
        while (xfers_done == consumed) {} /* wait for transfer `consumed` to complete */
        int16_t *src = audio_buf[consumed % NUM_BUFF];
        SCB_InvalidateDCache_by_Addr((uint32_t *)src, sizeof(audio_buf[0]));
        memcpy(&window_buf[filled * XFER_SAMPLES], src, sizeof(audio_buf[0]));
        /* checked after the copy: a full ring of lag means the DMA reached src (before or
         * during the memcpy), so the copy may be torn - drop the window and resync */
        uint32_t lag = xfers_done - consumed;
        if (lag >= NUM_BUFF) {
            xprintf("[overrun] %u xfers behind (ring %d), window dropped, resync\n",
                    (unsigned)lag, NUM_BUFF);
            consumed = xfers_done;
            filled = 0;
            continue;
        }
        consumed++;
        if (++filled == XFERS_PER_WIN) {
            filled = 0;
            if (cv_bird_run(window_buf, &res) == 0) {
                lag = xfers_done - consumed; /* transfers completed during inference */
                char tag[16];
                snprintf(tag, sizeof(tag), "[win %d]", win_no++);
                print_result(tag, &res, freq);
                xprintf("[lag %d] %u xfers\n", win_no - 1, (unsigned)lag);
#if BIRD_DUMP_EMB
                const int8_t *e;
                float es;
                int ez, n = cv_bird_emb(&e, &es, &ez);
                if (n) { /* xprintf has no %f: scale printed as nano units */
                    xprintf("[emb %d] n %d scale_nano %d zp %d ", win_no - 1, n,
                            (int)(es * 1e9f), ez);
                    for (int i = 0; i < n; i++)
                        xprintf("%02x", (uint8_t)e[i]);
                    xprintf("\n");
                }
#endif
            }
        }
    }
    return 0;
}
