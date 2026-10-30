# COBOL kernel build for pygeoflood (spec D1, art_QEojt4gd).
#
# Requires GnuCOBOL 3 (apt package gnucobol3 on ubuntu-latest and dev
# sandboxes). The pipeline is the one validated in the project sandbox:
# cobc -C lowers the COBOL to C, then one gcc -shared step compiles the
# generated C together with the ctypes-facing shim into a single
# pygeoflood_cobol.so. `cob-config` supplies the libcob include and
# link flags; the shim owns the one-time cob_init() runtime startup.

COBC ?= cobc
CC ?= gcc
PYTHON ?= python3
LIB := lib/pygeoflood_cobol.so

lib: $(LIB)

$(LIB): anisodiff.cob shim.c
	$(COBC) -C -O2 anisodiff.cob
	mkdir -p lib
	$(CC) -shared -fPIC -O2 -o $@ shim.c anisodiff.c \
	    $$(cob-config --cflags) $$(cob-config --libs)

# Build, then run the COBOL parity suite through the real dispatch.
check: lib
	cd .. && PYGEOFLOOD_COBOL=1 $(PYTHON) -m pytest tests/test_anisodiff_cobol.py -q

clean:
	rm -rf lib anisodiff.c anisodiff.c.h anisodiff.c.l.h

.PHONY: check clean
