# Microcontroller build: Kinglet XS on the Grove Vision AI V2

This folder holds everything behind section 3.7, Fig. 4 and Table S6 / Appendix S5: the int8 export of
Kinglet XS, the Vela compilation for the Ethos-U55, the firmware (a scenario app of the vendor SDK), the board
agreement test and the power measurement. Paths in capitals (`DISTILL_ROOT`, `DATA_ROOT`, `RELEASE_ASSETS`) are
the locations set in `paper/config.yaml` (see `paper/common/config.py`). All deployment files are written to
`DISTILL_ROOT/xs_pub_deploy/`.

Facts marked **[log]** were checked against the scripts and logs of the run (conversion log, Vela log and
summary, flashing log, UART log of the board, power logs); facts marked **[not verified]** come from the vendor
documentation or were not recorded in a log.

| File | Role |
|---|---|
| `export_int8.py` | compact output layer, deploy graph, calibration / parity windows, int8 TFLite conversion |
| `gen_mel_assets.py` | firmware constants `firmware/bird_pdm/mel_assets.{h,c}` (mel filterbank, normalisation, input quantisation, class codes, golden reference) |
| `firmware/bird_pdm/` | the scenario app (11 files) |
| `firmware/sdk.patch` | the two changes to the SDK itself (`app/main.c`, `makefile`) |
| `board_pcm_test.py` | board vs host on 512 WABAD windows sent over the UART (mcuboard_v1) |
| `mcurate.py` | host emulation of the board's 31,250 Hz capture (mcurate_v1b) |
| `measure_power.py` | power trace with a Nordic Power Profiler Kit II |

## 1. Environments

| Environment | Used for | Versions |
|---|---|---|
| main | `export_int8.py --stage pack/prep`, `gen_mel_assets.py`, `board_pcm_test.py prep/analyze`, `mcurate.py` | Python 3.12, PyTorch / torchaudio, timm, TensorFlow (for `tf.lite.Interpreter`), the `kinglet` package of this archive. **[not verified]** exact versions at run time; the environment as inspected afterwards had torch 2.10.0, torchaudio 2.10.0, timm 1.0.30, numpy 2.4.2, tensorflow 2.20.0 |
| converter | `export_int8.py --stage convert`, Vela | Python 3.12.3, torch 2.9.1+cpu, torchvision 0.24.1+cpu, torchao 0.14.1, litert-torch 0.8.0 (requires torch < 2.10), ai-edge-litert 2.1.2, ai-edge-tensorflow 2.21.0.dev20251110, timm 1.0.31, ethos-u-vela 5.2.0. torch 2.9.1 / torchao 0.14.1 are confirmed by the conversion log **[log]**; the other versions are those found in the environment afterwards **[not verified]** |
| board | flashing, `board_pcm_test.py run/log`, `measure_power.py` | pyserial 3.5, xmodem 0.4.7 (the SDK's `xmodem/requirements.txt`), ppk2-api 0.9.2, numpy |
| firmware | building the image | Arm GNU Toolchain 13.2.Rel1 (`arm-none-eabi`); the board reports "ARM GNU, 13.2.1 20231009" at boot **[log]**; GNU make |

## 2. Int8 model

The checkpoint is Kinglet XS (`DISTILL_ROOT/g2xcl_v1/mobilenetv4_conv_small_full/best.pt`). Its focal-only output
layer bundle `headpack_focal.npz` lies next to it (`evaluate/species_lists/export_headpack.py --focal-only`).

```bash
python mcu/export_int8.py --stage pack      # pack_jp_focal.npz: 317 species + ridge "other" row, no sound groups
python mcu/export_int8.py --stage prep      # deploy_state.pt, calib.npz (512 focal validation windows),
                                            # parity.npz (2,048 WABAD windows + fp32 reference)
<converter env>/bin/python mcu/export_int8.py --stage convert   # kinglet-xs_pub-jp-int8.tflite, parity_results.json
```

- The species without a Perch 2.0 class (bias -1e9) is clamped to a finite logit below the observed floor before
  quantisation (one row in the deployed layer: W = 0, b = -33.5 in `deploy_state.pt` **[log]**).
- Conversion result **[log]**: 3.33 MB TFLite; outputs = probabilities (1 x 318, float32, after an on-CPU
  dequantize + softmax) and the embedding feature (1 x 1280, int8, scale 0.0027061118744313717, zero point -128);
  input int8 (1, 1, 128, 501), scale 0.023585379123687744, zero point -33.
  Parity on the 2,048 WABAD windows: top-1 agreement 0.9946, window detection (p > 0.1) disagreement 0.0083.
- `--stage pack` reproduces the deployed `pack_jp_focal.npz` bit for bit (checked).

## 3. Vela (Ethos-U55-64)

```bash
<converter env>/bin/vela --accelerator-config ethos-u55-64 --output-dir DISTILL_ROOT/xs_pub_deploy \
    DISTILL_ROOT/xs_pub_deploy/kinglet-xs_pub-jp-int8.tflite > DISTILL_ROOT/xs_pub_deploy/xs_pub_vela.log
```

Writes `kinglet-xs_pub-jp-int8_vela.tflite` and `kinglet-xs_pub-jp-int8_summary_Ethos_U55_High_End_Embedded.csv`.
**[log]** accelerator Ethos_U55_64, Vela's default configuration file (system configuration
Ethos_U55_High_End_Embedded, memory mode Shared_Sram), 48 NPU and 38 CPU operators, 637.81 KiB SRAM,
2,699.42 KiB off-chip flash, 230,358,784 MACs per inference. **[not verified]** the exact command line (only the
accelerator and the default configuration are recorded in the log and summary).

## 4. Firmware assets

```bash
python mcu/gen_mel_assets.py --write-golden   # golden_{probs,emb}_int8.npy, then firmware/bird_pdm/mel_assets.{h,c}
```

The golden window is LCG noise that the firmware regenerates at boot (`bird_golden_fill` in `bird_pdm.c`); its
reference is the host int8 TFLite output. With the defaults (deployed XS model, input scale / zero point of §2)
the script regenerates the committed `mel_assets.h` / `mel_assets.c` and the deployed golden files byte for byte
(checked).

## 5. Building, signing and flashing the firmware

SDK: Himax `Seeed_Grove_Vision_AI_Module_V2` (github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) at
commit `d3265e20d75fe20faccbf185d24454ad08b2fda8`.

```bash
git clone https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2 sdk
cd sdk && git checkout d3265e20d75fe20faccbf185d24454ad08b2fda8
git apply <paper>/mcu/firmware/sdk.patch                 # BIRD_PDM main() + APP_TYPE = bird_pdm
cp -r <paper>/mcu/firmware/bird_pdm EPII_CM55M_APP_S/app/scenario_app/bird_pdm
mkdir -p model_zoo/bird_pdm && cp DISTILL_ROOT/xs_pub_deploy/kinglet-xs_pub-jp-int8_vela.tflite model_zoo/bird_pdm/

cd EPII_CM55M_APP_S && make clean && make                # production image
#                      make clean && make BIRD_PCM_TEST=1  # test image for board_pcm_test.py (section 6)

cd ../we2_image_gen_local                                 # secure-boot image (signing)
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json       # -> output_case1_sec_wlcsp/output.img

cd ..
python xmodem/xmodem_send.py --port=<board serial port> --baudrate=921600 --protocol=xmodem \
    --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
    --model="model_zoo/bird_pdm/kinglet-xs_pub-jp-int8_vela.tflite 0xB7B000 0x00000"
# press the board's reset button when asked; after "Firmware upgrade completed" press reset again
```

- **[log]** The flashing log shows the image `we2_image_gen_local/output_case1_sec_wlcsp/output.img` and the model
  `model_zoo/bird_pdm/kinglet-xs_pub-jp-int8_vela.tflite` being sent by `xmodem_send.py`; the model flash address
  0xB7B000 (read by the CPU at 0x3AB7B000 through the XIP window) is `BIRD_FLASH_ADDR` in `common_config.h`; the
  ELF name matches the build output of both images. **[not verified]** the `make` and `we2_local_image_gen`
  command lines (they follow the SDK README; the build itself was not logged).
- The SDK patch was produced from the local SDK clone, restricted to `app/main.c` and `makefile` and ignoring
  line-ending changes; it applies with `git apply` (or `patch -p1 --binary`) to the SDK commit above (checked).
  Other files that differed in the local clone (prebuilt libraries, secure-boot files) were not inspected
  **[not verified]** whether they matter for the build.
- Boot log of the production image (`fw_xs_pub_uart.log`) **[log]**: PDM rate measured 31,250 Hz (FREQ_16K + OSR 6;
  see `bird_pdm.c`), tensor arena 1,146,880 bytes of which 657,876 used, golden check PASS (host top-1 other 0.785,
  device 0.804, max |dp| 0.019), log-mel 108 ms and NPU + CPU 365 ms per window at 400 MHz, a new window every
  5.12 s (25 DMA transfers of 6,400 samples), no overruns.

## 6. Board agreement test (mcuboard_v1)

```bash
python mcu/board_pcm_test.py prep                          # main env: 512 windows, host int8 / fp32 references
<board env>/bin/python mcu/board_pcm_test.py run --port <board serial port>   # test image; resumable
python mcu/board_pcm_test.py analyze                       # -> results/mcuboard_v1.json
# after re-flashing the production image:
<board env>/bin/python mcu/board_pcm_test.py log --seconds 75 --port <board serial port> --log fw_prod_uart_after.log
```

The test image (`make BIRD_PCM_TEST=1`) receives each window as int16 PCM over the console UART (921,600 baud),
checks it with an FNV-1a hash and runs the unchanged log-mel + network. Windows are every 4th of the 2,048 parity
windows of §2, in two arms (`exact32k`: the 5-s window at 32 kHz; `mcu31k`: the window plus 0.12 s of the next one
resampled to 31,250 Hz, as `mcurate.py`). Known limitation: the `[mel]` dump of the first 8 windows is printed
after the inference, when TFLite Micro has reused the input tensor's memory, so `mel_input_diag_first_windows` in
the result is not the network input and is not used in the paper.

## 7. Power measurement

Wiring: Nordic Power Profiler Kit II (PPK2) in source-meter mode powers the board through the Grove connector
(PPK2 VOUT -> pin 3 GROVE_5V_IN, GND -> pin 4); the board's USB-C is unplugged. The production image runs
without a host.

```bash
<board env>/bin/python mcu/measure_power.py --seconds 60 --mv 5000 --out DISTILL_ROOT/xs_pub_deploy/power_xspub_5v_60s_r1.npz
# r2, r3 likewise
```

**[log]** three 60-s runs at 5.000 V, 100.0 kS/s each (no dropped samples): mean 110.3 / 110.4 / 110.4 mW over the
whole trace, steady state over 10 whole windows 110.1 / 110.2 / 110.2 mW (22.02-22.04 mA); one 353-354-ms burst per
window; window period 5.107 / 5.120 / 5.120 s. `figures/build_figure_data.py` reads the three traces.

## 8. Licences of the firmware files

The SDK is distributed under the MIT licence (Copyright (c) 2023 Himax Technologies, Inc.). None of the SDK files
used here carries a per-file licence header (the Arm-licensed files of the SDK's `kws_pdm_record` app, which do,
are not used), so there was no vendor header to keep. Each file now states its origin in its first line:

| File | Origin |
|---|---|
| `bird_pdm.ld`, `bird_pdm.sct`, `drv_user_defined.mk` | copied unchanged from the SDK's `kws_pdm_record` app (MIT, Himax) |
| `bird_pdm.mk` | adapted from `kws_pdm_record.mk` (MIT, Himax) |
| `bird_pdm.c` | new; PDM / DMA set-up follows `kws_pdm_record.c` (MIT, Himax) |
| `cvapp_bird.cpp` | new; Ethos-U55 initialisation follows the SDK's scenario apps (MIT, Himax) |
| `bird_pdm.h`, `cvapp_bird.h`, `common_config.h` | new |
| `mel_assets.h`, `mel_assets.c` | generated by `gen_mel_assets.py` |
| `sdk.patch` | changes to two SDK files (MIT, Himax) |

MIT licence text of the SDK, which applies to the SDK-derived parts:

```
MIT License

Copyright (c) 2023 Himax Technologies, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
