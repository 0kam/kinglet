# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Adapted from kws_pdm_record.mk of the Himax Seeed_Grove_Vision_AI_Module_V2 SDK (MIT licence; see mcu/BUILD.md).
override SCENARIO_APP_SUPPORT_LIST := $(APP_TYPE)

APPL_DEFINES += -DBIRD_PDM
APPL_DEFINES += -DTF_LITE_STATIC_MEMORY
APPL_DEFINES += -DETHOSU_ARCH=u55
APPL_DEFINES += -DETHOSU55
APPL_DEFINES += -DETHOSU_LOG_SEVERITY=ETHOSU_LOG_WARN
APPL_DEFINES += -DMODEL_ETHOSU55
APPL_DEFINES += -DIP_xdma
APPL_DEFINES += -DDBG_MORE
# `make BIRD_PCM_TEST=1`: host PCM test mode (bird_pdm.c), not for deployment
ifeq ($(BIRD_PCM_TEST),1)
APPL_DEFINES += -DBIRD_PCM_TEST=1
endif

LIB_SEL = pwrmgmt sensordp tflmtag2209_u55tag2205 spi_ptl spi_eeprom hxevent cmsis_dsp
override LIB_CMSIS_NN_ENALBE := 1
override LIB_CMSIS_NN_VERSION := 0

FATFS_PORT_LIST = mmc_spi
CMSIS_DRIVERS_LIST = SPI

MID_SEL =

override OS_SEL:=
override TRUSTZONE := y
override TRUSTZONE_TYPE := security
override TRUSTZONE_FW_TYPE := 1
override CIS_SEL := HM_COMMON
override EPII_USECASE_SEL := drv_user_defined

ifeq ($(strip $(TOOLCHAIN)), arm)
override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/bird_pdm.sct
else
override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/bird_pdm.ld
endif
