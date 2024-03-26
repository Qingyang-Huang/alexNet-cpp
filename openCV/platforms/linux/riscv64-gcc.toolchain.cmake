set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

#set(RISCV_GCC_INSTALL_ROOT /opt/RISCV CACHE PATH "Path to GCC for RISC-V cross compiler installation directory")
#set(CMAKE_SYSROOT ${RISCV_GCC_INSTALL_ROOT}/sysroot CACHE PATH "RISC-V sysroot")

set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

# Don't run the linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)


set(CMAKE_CXX_FLAGS ""    CACHE STRING "")
set(CMAKE_C_FLAGS ""    CACHE STRING "")

if(BUILD_SHARED_LIBS STREQUAL "OFF")
if(CORE STREQUAL "C908V")
    set(CMAKE_C_FLAGS "-static -mcpu=c908v -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
    set(CMAKE_CXX_FLAGS "-static -mcpu=c908v -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
elseif(CORE STREQUAL "C907FDVM-RV64")
    set(CMAKE_C_FLAGS "-static -march=rv64imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=lp64d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-static -march=rv64imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=lp64d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C907FDVM-RV32")
    set(CMAKE_C_FLAGS "-static -march=rv32imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=ilp32d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-static -march=rv32imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=ilp32d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C906FDV")
    set(CMAKE_C_FLAGS "-static -march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c906 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-static -march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c906 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C920")
    set(CMAKE_C_FLAGS "-static -march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c920 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-static -march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c920 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C920V2")
    set(CMAKE_C_FLAGS "-static -mcpu=c920v2 -mabi=lp64d  -mtune=c920v2 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
    set(CMAKE_CXX_FLAGS "-static -mcpu=c920v2 -mabi=lp64d  -mtune=c920v2 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
else()
    set(CMAKE_C_FLAGS "-static -march=rv64imafdc_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-static -march=rv64imafdc_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
endif()
else()
if(CORE STREQUAL "C908V")
    set(CMAKE_C_FLAGS "-mcpu=c908v -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
    set(CMAKE_CXX_FLAGS "-mcpu=c908v -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
elseif(CORE STREQUAL "C907FDVM-RV64")
    set(CMAKE_C_FLAGS "-march=rv64imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=lp64d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-march=rv64imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=lp64d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C907FDVM-RV32")
    set(CMAKE_C_FLAGS "-march=rv32imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=ilp32d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-march=rv32imafdc_zihintntl_zihintpause_zawrs_zfa_zfh_zba_zbb_zbc_zbs_zvamo_zvlsseg_xtheadc -mabi=ilp32d -mtune=c907 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C906FDV")
    set(CMAKE_C_FLAGS "-march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c906 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c906 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C920")
    set(CMAKE_C_FLAGS "-march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c920 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-march=rv64imafdcxtheadc -mabi=lp64d  -mtune=c920 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
elseif(CORE STREQUAL "C920V2")
    set(CMAKE_C_FLAGS "-mcpu=c920v2 -mabi=lp64d  -mtune=c920v2 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
    set(CMAKE_CXX_FLAGS "-mcpu=c920v2 -mabi=lp64d  -mtune=c920v2 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive -D__riscv_vector_071 -mrvv-vector-bits=128")
else()
    set(CMAKE_C_FLAGS "-march=rv64imafdc_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
    set(CMAKE_CXX_FLAGS "-march=rv64imafdc_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d  -mtune=c908 -O3 -Wl,-whole-archive -lpthread -Wl,-no-whole-archive")
endif()
endif()

if(ENABLE_GCOV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
