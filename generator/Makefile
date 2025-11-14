HALIDE_DIR ?= /opt/homebrew/Cellar/halide/21.0.0
HALIDE_AUTOSCHEDULER_DIR ?= $(HALIDE_DIR)/lib
LIBJPEG_DIR ?= /opt/homebrew/Cellar/jpeg/9f

LIBMULLAPUDI = $(HALIDE_AUTOSCHEDULER_DIR)/libautoschedule_mullapudi2016.so
LIBLI = $(HALIDE_AUTOSCHEDULER_DIR)/libautoschedule_li2018.so
LIBADAMS = $(HALIDE_AUTOSCHEDULER_DIR)/libautoschedule_adams2019.so
LIBANDERSON = $(HALIDE_AUTOSCHEDULER_DIR)/libautoschedule_anderson2021.so

bin/threshold.generator: halide_generator.cpp
	c++ -g GenGen.cpp halide_generator.cpp \
		-I $(HALIDE_DIR)/include \
		-L $(HALIDE_DIR)/lib \
		-lHalide \
		-Wl,-rpath $(HALIDE_DIR)/lib \
		--std=c++17 \
		-o bin/threshold.generator

# autoscheduler=Mullapudi2016
# autoscheduler=Li2018
# autoscheduler=Adams2019
# autoscheduler=Anderson2021

bin/threshold.a: bin/threshold.generator
	./bin/threshold.generator -g threshold -o bin \
		-e static_library,h,registration,stmt_html,assembly,schedule,conceptual_stmt_html \
		target=arm-64-osx-arm_dot_prod-arm_fp16-metal \
		-p $(LIBANDERSON),$(LIBADAMS),$(LIBLI),$(LIBMULLAPUDI) \
		autoscheduler=Anderson2021 \
		min_white_black_diff=5

bin/thresh.rungen: bin/threshold.a
	c++ RunGenMain.cpp bin/threshold.a bin/threshold.registration.cpp \
		-I $(HALIDE_DIR)/include \
		-I /Users/jmcmichael/Code/Halide/src/runtime \
		-I $(LIBJPEG_DIR)/include \
		-L $(LIBJPEG_DIR)/lib \
		-o bin/thresh.rungen \
		--std=c++17 \
		-ljpeg -lz -lpthread -ldl \
		-framework Metal -framework Foundation \
		-D HALIDE_NO_PNG=1

test: bin/thresh.rungen
	./bin/thresh.rungen input=apriltags.jpg output=processed.jpg \
		--benchmarks=all --benchmark_min_time=1 --parsable_output

clean:
	rm -rf bin/*
