PYTHON ?= python
DATA ?= $(PTSD_DATA)
ID_HASHES ?= ../.private_release/id_hashes.json
CONF_TERMS ?= ../.private_release/confidential_terms.txt
LAYOUT ?= $(PTSD_LAYOUT)
JOBS ?= 4
RUN = PYTHONPATH=src:examples OMP_NUM_THREADS=1 $(PYTHON)

.PHONY: help check demo test test-full verify-weights numbers links figures animations reproduce privacy screenshots notebooks

help:
	@echo "make check          everything CI runs: test, privacy, numbers, links, demo (no data needed)"
	@echo "make demo           synthetic EDF -> prediction with the shipped weights (no data needed)"
	@echo "make test           synthetic test suite (CI)"
	@echo "make test-full      all tests incl. confidential data (DATA=/path/to/data LAYOUT=/path/to/layout.json)"
	@echo "make verify-weights checksums of weights/ against weights/SHA256SUMS"
	@echo "make numbers        every key number in the docs vs results/ (tools/check_numbers.py)"
	@echo "make links          relative links, anchors, images and code-span paths in the docs"
	@echo "make privacy        privacy + confidentiality scan of the repository (0 hits required)"
	@echo "make reproduce      retrain from data and compare with results/metrics.csv (DATA=... LAYOUT=...)"
	@echo "make notebooks      execute notebooks/ in place"
	@echo "make figures        regenerate figures/ from results/"
	@echo "make animations     regenerate media/ (MP4, GIF, posters, banner) from results/ (needs ffmpeg)"
	@echo "make screenshots    regenerate screenshots/ (needs chromium)"

check: test privacy numbers links demo

# The demo always uses the generic layout of the synthetic files, whatever PTSD_LAYOUT is set to.
demo:
	$(RUN) examples/make_synthetic_edf.py demo .demo/edf
	$(RUN) -m ptsd_eeg.cli verify-weights
	PTSD_LAYOUT= $(RUN) -m ptsd_eeg.cli predict .demo/edf .demo/predictions.csv
	@cat .demo/predictions.csv

test:
	$(RUN) -m pytest -m "not data" -q

test-full:
	@test -n "$(DATA)" || (echo "set DATA=/path/to/data (the confidential data root)"; exit 2)
	PTSD_DATA=$(DATA) PTSD_LAYOUT=$(LAYOUT) $(RUN) -m pytest -q

verify-weights:
	$(RUN) -m ptsd_eeg.cli verify-weights

numbers:
	$(RUN) tools/check_numbers.py

links:
	$(PYTHON) tools/check_links.py

reproduce:
	@test -n "$(DATA)" || (echo "set DATA=/path/to/data (the confidential data root)"; exit 2)
	PTSD_LAYOUT=$(LAYOUT) $(RUN) -m ptsd_eeg.cli reproduce --data $(DATA) --jobs $(JOBS) --out .reproduce

# The ID-hash list and the confidentiality denylist live outside the repository; each check is skipped
# (with a notice) when its file is absent, e.g. in CI.
privacy:
	@$(PYTHON) tools/privacy_scan.py $(if $(wildcard $(ID_HASHES)),--hashes $(ID_HASHES)) \
		$(if $(wildcard $(CONF_TERMS)),--terms $(CONF_TERMS))

notebooks:
	$(RUN) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False notebooks/*.ipynb
	$(RUN) -c "import nbformat,sys; [nbformat.write(nb, p) for p in sys.argv[1:] for nb in [nbformat.read(p, as_version=4)] if [c.metadata.pop('execution', None) for c in nb.cells] or True]" notebooks/*.ipynb

# MENDELEY=/path/to/public/mendeley/cache (mat/, workbooks/) also redraws the two public-data figures.
figures:
	$(RUN) tools/make_figures.py $(if $(MENDELEY),--mendeley $(MENDELEY))

# Needs an ffmpeg with libx264 (FFMPEG=/path/to/ffmpeg overrides the lookup).
animations:
	$(RUN) tools/make_animations.py --results results --out media

# Needs chromium; DATA=... LAYOUT=... adds the full (data-gated) pytest screenshot.
screenshots:
	$(RUN) tools/make_screenshots.py $(if $(DATA),--data $(DATA)) $(if $(LAYOUT),--layout $(LAYOUT)) \
		--id-hashes $(ID_HASHES) --terms $(CONF_TERMS)
