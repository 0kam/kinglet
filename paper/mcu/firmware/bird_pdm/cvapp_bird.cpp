/* Written with Claude (Anthropic) under the author's direction; see paper/README.md. */
/* Bird classifier (Kinglet XS) on WE2: log-mel on CM55 + int8 trunk on Ethos-U55.
 * NPU initialisation follows the SDK scenario apps (MIT licence, Himax).
 *
 * Input contract (must match the host pipeline exactly, see mcu/export_int8.py):
 *   audio int16 @32 kHz, 160000 samples -> MelSpectrogram(n_fft 1024, hop 320, hann,
 *   power 2, htk mels 0..16k, center=True reflect pad) -> log(x+1e-6) -> (x-mu)/sd
 *   -> quantize to int8 input (1, 1, 128, 501) -> model -> float probs (318).
 */
#include <cstdint>
#include <cstring>
#include <cmath>

#include "WE2_device.h"
#include "WE2_core.h"
#include "board.h"
#include "xprintf.h"

#include "ethosu_driver.h"
#include "arm_math.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "mel_assets.h"
#include "cvapp_bird.h"

#define N_SAMPLES 160000
#define CPU_CLK (0xffffff + 1)

#ifdef TRUSTZONE_SEC
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE BASE_ADDR_APB_U55_CTRL
#endif
#endif

struct ethosu_driver ethosu_drv;

constexpr int tensor_arena_size = 0x118000; /* 1.1 MB: Vela NPU scratch 570 KiB + CPU ops */
static uint8_t tensor_arena[tensor_arena_size] __ALIGNED(32);

static tflite::MicroInterpreter *bird_int_ptr = nullptr;
static TfLiteTensor *bird_in = nullptr, *bird_out = nullptr, *bird_emb_t = nullptr;
static float in_scale = BIRD_IN_SCALE;
static int in_zp = BIRD_IN_ZP;

static arm_rfft_fast_instance_f32 rfft;
static float frame_f[BIRD_N_FFT];
static float spec[BIRD_N_FFT];
static float power[BIRD_N_FFT / 2 + 1];

static void _arm_npu_irq_handler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;
    _arm_npu_irq_init();
#if TFLM2209_U55TAG2205
    const void *ethosu_base_address = (void *)(U55_BASE);
#else
    void *const ethosu_base_address = (void *)(U55_BASE);
#endif
    if (0 != (err = ethosu_init(&ethosu_drv, ethosu_base_address, NULL, 0,
                                security_enable, privilege_enable))) {
        xprintf("failed to initialise Ethos-U device\n");
        return err;
    }
    xprintf("Ethos-U55 device initialised\n");
    return 0;
}

int cv_bird_init(bool security_enable, bool privilege_enable, uint32_t model_addr)
{
    xprintf("TA[%x] size %d\r\n", (unsigned)tensor_arena, tensor_arena_size);

    if (_arm_npu_init(security_enable, privilege_enable) != 0)
        return -1;
    if (model_addr == 0)
        return -1;

    static const tflite::Model *model = tflite::GetModel((const void *)model_addr);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("[ERROR] model schema %d != supported %d\n",
                (int)model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static tflite::MicroErrorReporter error_reporter;
    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSum();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    if (kTfLiteOk != resolver.AddEthosU()) {
        xprintf("Failed to add Arm NPU support to op resolver.\n");
        return -1;
    }

#if TFLM2209_U55TAG2205
    static tflite::MicroInterpreter interp(model, resolver, (uint8_t *)tensor_arena,
                                           tensor_arena_size, &error_reporter);
#else
    static tflite::MicroInterpreter interp(model, resolver, (uint8_t *)tensor_arena,
                                           tensor_arena_size);
#endif
    if (interp.AllocateTensors() != kTfLiteOk) {
        xprintf("AllocateTensors failed (arena %d)\n", tensor_arena_size);
        return -1;
    }
    xprintf("arena used: %d bytes\n", (int)interp.arena_used_bytes());

    bird_int_ptr = &interp;
    bird_in = interp.input(0);
    /* two outputs since the emb export: probs (318, float) + embedding (1280, int8) */
    for (size_t k = 0; k < interp.outputs_size(); k++) {
        TfLiteTensor *t = interp.output(k);
        if (t->dims->data[t->dims->size - 1] == BIRD_N_CLASSES)
            bird_out = t;
        else
            bird_emb_t = t;
    }
    if (!bird_out) {
        xprintf("no %d-class output tensor found\n", BIRD_N_CLASSES);
        return -1;
    }
    xprintf("in: type %d bytes %d  out: type %d bytes %d  emb: %d bytes\n",
            (int)bird_in->type, (int)bird_in->bytes, (int)bird_out->type, (int)bird_out->bytes,
            bird_emb_t ? (int)bird_emb_t->bytes : 0);
    if (bird_in->quantization.type == kTfLiteAffineQuantization) {
        auto *q = (TfLiteAffineQuantization *)bird_in->quantization.params;
        in_scale = q->scale->data[0];
        in_zp = q->zero_point->data[0];
    }

    arm_rfft_fast_init_f32(&rfft, BIRD_N_FFT);
    return 0;
}

/* One log-mel frame -> int8 input column f. Reflect padding like torch (edge not repeated). */
static void mel_frame(const int16_t *audio, int f, int8_t *in)
{
    const int base = f * BIRD_HOP - BIRD_N_FFT / 2;
    for (int i = 0; i < BIRD_N_FFT; i++) {
        int idx = base + i;
        if (idx < 0)
            idx = -idx;
        else if (idx >= N_SAMPLES)
            idx = 2 * (N_SAMPLES - 1) - idx;
        frame_f[i] = (float)audio[idx] * (1.0f / 32768.0f) * bird_hann[i];
    }
    arm_rfft_fast_f32(&rfft, frame_f, spec, 0);
    power[0] = spec[0] * spec[0];
    power[BIRD_N_FFT / 2] = spec[1] * spec[1];
    for (int k = 1; k < BIRD_N_FFT / 2; k++) {
        const float re = spec[2 * k], im = spec[2 * k + 1];
        power[k] = re * re + im * im;
    }
    for (int m = 0; m < BIRD_N_MELS; m++) {
        const float *w = &bird_mel_w[bird_mel_off[m]];
        const float *p = &power[bird_mel_start[m]];
        float acc = 0.0f;
        for (int k = 0; k < bird_mel_len[m]; k++)
            acc += w[k] * p[k];
        const float v = (logf(acc + 1e-6f) - BIRD_MEL_MU) / BIRD_MEL_SD;
        int q = (int)lrintf(v / in_scale) + in_zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        in[m * BIRD_N_FRAMES + f] = (int8_t)q;
    }
}

int cv_bird_run(const int16_t *audio_160k, bird_result_t *res)
{
    if (!bird_int_ptr)
        return -1;
    uint32_t t1, t2, l1, l2;

    int8_t *in = tflite::GetTensorData<int8_t>(bird_in);
    SystemGetTick(&t1, &l1);
    for (int f = 0; f < BIRD_N_FRAMES; f++)
        mel_frame(audio_160k, f, in);
    SystemGetTick(&t2, &l2);
    res->mel_cycles = (l2 - l1) * CPU_CLK + (t1 - t2);

    SystemGetTick(&t1, &l1);
    if (bird_int_ptr->Invoke() != kTfLiteOk) {
        xprintf("Invoke failed\n");
        return -1;
    }
    SystemGetTick(&t2, &l2);
    res->npu_cycles = (l2 - l1) * CPU_CLK + (t1 - t2);

    const float *probs = tflite::GetTensorData<float>(bird_out);
    for (int k = 0; k < 5; k++) {
        res->top_idx[k] = 0;
        res->top_prob[k] = -1.0f;
    }
    for (int c = 0; c < BIRD_N_CLASSES; c++) {
        float p = probs[c];
        for (int k = 0; k < 5; k++) {
            if (p > res->top_prob[k]) {
                for (int j = 4; j > k; j--) {
                    res->top_prob[j] = res->top_prob[j - 1];
                    res->top_idx[j] = res->top_idx[j - 1];
                }
                res->top_prob[k] = p;
                res->top_idx[k] = c;
                break;
            }
        }
    }
    return 0;
}

/* Full output vector of the last Invoke (for the golden check). */
extern "C" const float *cv_bird_probs(void)
{
    return bird_out ? tflite::GetTensorData<float>(bird_out) : nullptr;
}

/* int8 model input of the last run (quantized log-mel, 128 x 501, mel-major) - for the
 * BIRD_PCM_TEST mel dump that diagnoses board-vs-host spectrogram differences. */
extern "C" const int8_t *cv_bird_input(void)
{
    return bird_in ? tflite::GetTensorData<int8_t>(bird_in) : nullptr;
}

/* int8 embedding of the last Invoke; returns its length (0 if the model has none).
 * float value = (q - zp) * scale; 1,280 bytes per window. */
extern "C" int cv_bird_emb(const int8_t **data, float *scale, int *zp)
{
    if (!bird_emb_t || bird_emb_t->type != kTfLiteInt8)
        return 0;
    *data = tflite::GetTensorData<int8_t>(bird_emb_t);
    auto *q = (TfLiteAffineQuantization *)bird_emb_t->quantization.params;
    *scale = q->scale->data[0];
    *zp = q->zero_point->data[0];
    return bird_emb_t->dims->data[bird_emb_t->dims->size - 1];
}
