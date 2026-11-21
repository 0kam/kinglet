/* Written with Claude (Anthropic) under the author's direction; see paper/README.md. */
#ifndef CVAPP_BIRD_H_
#define CVAPP_BIRD_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int top_idx[5];
    float top_prob[5];
    uint32_t mel_cycles;
    uint32_t npu_cycles;
} bird_result_t;

int cv_bird_init(bool security_enable, bool privilege_enable, uint32_t model_addr);
int cv_bird_run(const int16_t *audio_160k, bird_result_t *res);
const float *cv_bird_probs(void);
/* int8 embedding of the last run; float = (q - zp) * scale. Returns length or 0. */
int cv_bird_emb(const int8_t **data, float *scale, int *zp);
/* int8 model input (quantized log-mel, BIRD_N_MELS x BIRD_N_FRAMES) of the last run. */
const int8_t *cv_bird_input(void);

#ifdef __cplusplus
}
#endif

#endif /* CVAPP_BIRD_H_ */
