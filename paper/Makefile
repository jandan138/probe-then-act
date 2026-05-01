# Dual-venue LaTeX paper build system
# Usage: make ieee-trl | make neurips | make all | make clean | make check

VENUES := ieee-trl neurips
SHARED_DIR := shared
SHARED_SRCS := $(shell find $(SHARED_DIR) -path '$(SHARED_DIR)/figures/tmp' -prune -o -type f \( -name '*.tex' -o -name '*.pdf' -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.bib' -o -name '*.py' \) -print 2>/dev/null)

# Generate build rules for each venue
define VENUE_RULE
$(1)_MAIN := venues/$(1)/main.tex
$(1)_LOCAL := $(shell find venues/$(1) -type f \( -name '*.tex' -o -name '*.sty' -o -name '*.cls' \) 2>/dev/null)
$(1)_PDF := venues/$(1)/build/main.pdf
$(1)_BUILDDIR := venues/$(1)/build

$$($(1)_PDF): $$($(1)_MAIN) $$($(1)_LOCAL) $(SHARED_SRCS) | $$($(1)_BUILDDIR)
	@cd venues/$(1) && latexmk -cd -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex

$$($(1)_BUILDDIR):
	@mkdir -p $$@

.PHONY: $(1) clean-$(1)
$(1): $$($(1)_PDF)
	@echo "=== $(1) built: $$($(1)_PDF) ==="

clean-$(1):
	@cd venues/$(1) && latexmk -C -outdir=build 2>/dev/null || true
	@rm -rf $$($(1)_BUILDDIR)

endef

$(foreach v,$(VENUES),$(eval $(call VENUE_RULE,$(v))))

.PHONY: all clean check arxiv nips2026
all: $(foreach v,$(VENUES),$(v))

clean: $(foreach v,$(VENUES),clean-$(v))

check:
	@echo "=== IEEE T-RL page count ==="
	@texcount -inc -total venues/ieee-trl/main.tex 2>/dev/null || echo "texcount not installed"
	@echo ""
	@echo "=== NeurIPS page count ==="
	@texcount -inc -total venues/neurips/main.tex 2>/dev/null || echo "texcount not installed"

arxiv:
	@python3 scripts/flatten.py --venue neurips --output arxiv-submission/

nips2026: neurips
	@python3 scripts/package_nips2026.py
