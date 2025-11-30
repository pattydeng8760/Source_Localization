# Makefile for building source_loc_cpp Python extension on:
# - TRILLIUM cluster StdEnv/2023
# - gcc/12.3
# - openmpi/4.1.5
# - python/3.11.5
# - rarray/2.8.0
# Usage:
#   make        # build the extension
#   make FAST_MATH=1 LTO=1  # build with fast-math and LTO optimizations
#   make clean  # remove the extension

PYTHON = python3				# The Python interpreter
CXX    = mpicxx					# The C++ compiler (MPI wrapper)

# Get pybind11 include flags and Python extension suffix
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
# Get Python extension suffix and linker flags
EXT_SUFFIX        := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# Get Python linker flags
PYTHON_LDFLAGS    := $(shell $(PYTHON) -c "import sysconfig; v=sysconfig.get_config_var; print(' '.join((v('LDFLAGS') or '').split()))")

# Optional fast-math and LTO toggles for AMD EPYC specific optimizations
ifeq ($(FAST_MATH),1)
  MATHFLAGS = -ffast-math -fno-math-errno -fno-trapping-math
else
  MATHFLAGS =
endif

# Enable Link Time Optimization (LTO) if specified
ifeq ($(LTO),1)
  LTOFLAGS = -flto
else
  LTOFLAGS =
endif

CXXFLAGS = -O3 -std=c++17 -fPIC -fopenmp $(LTOFLAGS) \
           -march=znver3 -mtune=znver3 \
           $(MATHFLAGS) \
           -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 

LDFLAGS  = -shared $(LTOFLAGS) $(PYTHON_LDFLAGS)

TARGET = ../source_localization_core/source_localization_core_cpp$(EXT_SUFFIX)
SRC    = source_localization_worker.cpp source_localization_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)