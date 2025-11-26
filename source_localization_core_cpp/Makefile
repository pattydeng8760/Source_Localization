# Makefile for building source_loc_cpp Python extension on:
# - gcc/12.3
# - openmpi/4.1.5
# - python/3.11.5
# - rarray/2.8.0
#
# Usage:
#   make        # build the extension
#   make clean  # remove the extension

PYTHON = python3
CXX    = mpicxx   # force mpicxx, ignore env CXX

# pybind11 include flags and Python extension suffix
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX        := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYTHON_LDFLAGS    := $(shell $(PYTHON) -c "import sysconfig; v=sysconfig.get_config_var; print(' '.join((v('LDFLAGS') or '').split()))")

# rarray include path (module rarray/2.8.0 usually sets EBROOTRARRAY)
CXXFLAGS_EXTRA ?=
# If needed, uncomment this:
# CXXFLAGS_EXTRA = -I$(EBROOTRARRAY)/include

# IMPORTANT: disable MPI C++ bindings everywhere
CXXFLAGS = -O3 -std=c++17 -fPIC -fopenmp -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 $(CXXFLAGS_EXTRA)
LDFLAGS  = -shared $(PYTHON_LDFLAGS)

TARGET = source_loc_cpp$(EXT_SUFFIX)
SRC    = source_loc_single_obs.cpp source_loc_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)