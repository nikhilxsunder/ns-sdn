# References

The literature behind NS-SDN, in one place: a master BibTeX file that every
document in the repository draws its citations from, and the papers it cites,
as read.

This directory is research material, not package code. It is excluded from
the sdist (`pyproject.toml`, `[tool.hatch.build.targets.sdist] exclude`),
absent from the wheel, which packages `src/ns_sdn` only, not imported
anywhere, and nothing in `src/` depends on it.

## Layout

| Path | What it holds |
|---|---|
| `bibliography/master.bib` | The canonical bibliography: one entry per work under one key, grouped by theme, with a comment above each entry naming the PDF that holds it and the documents that cite it. |
| `papers/` | The 24 papers on disk, named by topic rather than by author so a shelf scan reads as a map of the field. Each one maps to exactly one entry in `master.bib`; the table below gives the key. |

## How the bibliographies relate

There are five `.bib` files in the repository. Only one is edited by hand.

| File | Role | Keys |
|---|---|---|
| `references/bibliography/master.bib` | Source of truth. Edit this one. | Canonical: `surname` + `year` + `slug`, lower-case ASCII (`sitzmann2020siren`). |
| `docs/source/references.bib` | The subset the Sphinx site cites, copied verbatim from the master. | Same keys as the master. |
| `papers/journal/bib/references.bib` | The long-form manuscript's bibliography. Populated from the master as the manuscript cites. | Same keys as the master. |
| `papers/flairs-39/references.bib` | Shipped with the published FLAIRS paper. Frozen as submitted. | Its own; see the alias table. |
| `papers/ieee-orss/references.bib` | Shipped with the published IEEE ORSS paper. Frozen as submitted. | Its own; see the alias table. |
| `papers/icsri-proposal/references.bib` | Shipped with the ICSRI proposal. Frozen as submitted. | Its own; see the alias table. |

The two published bibliographies and the proposal's are historical records of
what was submitted and are not retouched, even where the master entry is more
complete (the ORSS bibliography cites the arXiv DeepAR with its 2017 author
list; the master cites the journal version). New writing cites the master.

## Papers on disk

The version on disk is not always the version the entry cites: several files
are the arXiv preprint or a working paper of a work later published in a
journal, and the entry cites the published version. The comment above each
entry in `master.bib` records the difference.

| File in `papers/` | Key in `master.bib` | Work | Version on disk | Cited by |
|---|---|---|---|---|
| `Sinusoidal-Networks.pdf` | `sitzmann2020siren` | Sitzmann et al., SIREN, NeurIPS 2020 | arXiv:2006.09661v1 | FLAIRS, ORSS, ICSRI, docs, src |
| `Fourier-Feature-Networks.pdf` | `tancik2020fourier` | Tancik et al., Fourier features, NeurIPS 2020 | arXiv:2006.10739v1 | journal |
| `Frequency-Modulation.pdf` | `chowning1973fm` | Chowning, FM synthesis, JAES 1973 | Computer Music Journal 1977 reprint | ICSRI, journal |
| `Wavelet-Networks.pdf` | `zhang1992wavelet` | Zhang and Benveniste, IEEE TNN 1992 | Publisher scan; no text layer | journal |
| `Hilbert-Huang-Transforms.pdf` | `huang1998emd` | Huang et al., EMD and the Hilbert spectrum, Proc. R. Soc. A 1998 | HAL open-archive copy | FLAIRS, ORSS, ICSRI, docs, src |
| `Oscillatory-Components.pdf` | `matsuda2017oscillation` | Matsuda and Komaki, oscillation components, Neural Computation 2017 | METR 2016-8 technical report | ORSS |
| `Spectral-Density.pdf` | `mohammadi2026spectral` | Mohammadi, Sarkar and Kokoszka, spectral density of functional time series | arXiv:2601.00284v1 | ORSS |
| `Spectral-Disentanglement.pdf` | `an2026fredn` | An et al., FreDN | arXiv:2511.11817v2 | ORSS |
| `Time-Series-Decomposition.pdf` | `beveridge1981decomposition` | Beveridge and Nelson, permanent and transitory components, JME 1981 | Publisher PDF | ICSRI |
| `Time-Varying-Parameters-VAR.pdf` | `lubik2015tvpvar` | Lubik and Matthes, TVP-VARs, FRB Richmond EQ 2015 | Publisher PDF (open access) | FLAIRS, ICSRI |
| `Predictive-Accuracy.pdf` | `diebold1995accuracy` | Diebold and Mariano, comparing predictive accuracy, JBES 1995 | Scanned working paper ("Comparing Predictive Accuracy I") | FLAIRS, ICSRI |
| `Asymptotic-Inference.pdf` | `west1996predictive` | West, asymptotic inference about predictive ability, Econometrica 1996 | Publisher scan | FLAIRS, ICSRI |
| `Recursive-And-Direct-Multi-Step.pdf` | `bentaieb2012recursive` | Ben Taieb and Hyndman, recursive and direct multi-step forecasting | Monash working paper 19/12 (the cited version) | journal |
| `LSTM.pdf` | `hochreiter1997lstm` | Hochreiter and Schmidhuber, LSTM, Neural Computation 1997 | Publisher scan; no text layer | ICSRI |
| `RNN-Encoder-Decoder.pdf` | `cho2014gru` | Cho et al., RNN encoder-decoder (the GRU), EMNLP 2014 | arXiv:1406.1078 | ORSS, ICSRI, docs, src |
| `Scheduled-Sampling.pdf` | `bengio2015scheduled` | Bengio et al., scheduled sampling, NeurIPS 2015 | Conference version | journal |
| `Professor-Forcing.pdf` | `lamb2016professor` | Lamb and Goyal et al., professor forcing, NeurIPS 2016 | Conference version (lists Goyal first) | journal |
| `Neural-ODEs.pdf` | `chen2018neuralode` | Chen et al., neural ODEs, NeurIPS 2018 | arXiv:1806.07366 | ORSS, ICSRI |
| `DeepAR.pdf` | `salinas2020deepar` | Salinas et al., DeepAR, IJF 2020 | arXiv:1704.04110 (2017 author list) | FLAIRS, ORSS, ICSRI |
| `N-BEATS.pdf` | `oreshkin2020nbeats` | Oreshkin et al., N-BEATS, ICLR 2020 | arXiv:1905.10437 | ICSRI |
| `Temporal-Fusion-Transformers.pdf` | `lim2021tft` | Lim et al., temporal fusion transformers, IJF 2021 | arXiv:1912.09363 | ORSS, ICSRI |
| `Spatiotemporal-Forecasts.pdf` | `vlachas2024learning` | Vlachas and Koumoutsakos, learning on predictions, Physica D 2024 | Publisher PDF (CC BY) | journal |
| `Multi-Scale-Dynamics.pdf` | `chattopadhyay2023multiscale` | Chattopadhyay, Sun and Hassanzadeh, multi-scale dynamics in AI weather models | arXiv:2304.07029v2 | journal |
| `Neural-PDE-Solver.pdf` | `brandstetter2022mppde` | Brandstetter, Worrall and Welling, message passing neural PDE solvers, ICLR 2022 | arXiv:2202.03376v3 | journal |

"Cited by" names the documents: the FLAIRS paper, the IEEE ORSS paper, the
ICSRI proposal, the Sphinx documentation (`docs`), the module docstrings
(`src`), and the long-form journal manuscript (`journal`, in progress). The
sixteen entries in `master.bib` without a PDF are books, the project's own
papers and software, and works cited from their published record.

## Key aliases in the frozen bibliographies

The published bibliographies were written before the master existed and use
their own keys. This table is the crosswalk; it is the only place the old
keys are recorded.

| Master key | FLAIRS | IEEE ORSS | ICSRI |
|---|---|---|---|
| `sunder2026flairs` | | `sunder2026nssdn` | |
| `sunder2025fedfred` | | `sunderfedfred` | |
| `sitzmann2020siren` | `sitzmann2020siren` | `sitzmann2020implicit` | `sitzmann2020` |
| `chowning1973fm` | | | `chowning1973` |
| `huang1998emd` | `huang1998hht` | `huang1998empirical` | `huang1998` |
| `wu2009eemd` | | `wu2009emd` | |
| `matsuda2017oscillation` | | `matsuda2016oscillation` | |
| `mohammadi2026spectral` | | `mohammadi2026spectral` | |
| `an2026fredn` | | `an2026fredn` | |
| `durbin2012ssm` | `durbin2012statespace` | `durbin2012time` | `durbin2012` |
| `harvey1989structural` | `harvey1989structural` | | `harvey1989` |
| `beveridge1981decomposition` | | | `beveridge1981` |
| `rangapuram2018dssm` | | `rangapuramdeep` | |
| `hamilton1989regime` | `hamilton1989regime` | `hamilton1989` | `hamilton1989` |
| `hamilton1994tsa` | | `hamilton1994` | |
| `sims1980macro` | | `sims1980` | |
| `lutkepohl2005multiple` | | `lutkepohl2005` | |
| `primiceri2005tvpvar` | | `primiceritvp` | |
| `lubik2015tvpvar` | `lubik2015tvpvar` | | `lubik2015` |
| `diebold1995accuracy` | `diebold1995accuracy` | | `diebold1995` |
| `west1996predictive` | `west1996predictive` | | `west1996` |
| `hochreiter1997lstm` | | | `hochreiter1997` |
| `cho2014gru` | | `cho2014learningphrase` | `cho2014` |
| `williams1989rtrl` | | `williams1989ala` | |
| `kingma2015adam` | | `adamoptim` | |
| `chen2018neuralode` | | `neuralodes` | `chen2018` |
| `salinas2020deepar` | `flunkert2020deepar` | `flunkertdeepar` | `salinas2020` |
| `oreshkin2020nbeats` | | | `oreshkin2020` |
| `lim2021tft` | | `lim2020temporalfusion` | `lim2021` |

## Conventions

1. **One work, one entry, one key.** A preprint and its published version
   are the same work. The entry cites the published version and carries the
   arXiv identifier in `eprint`/`archiveprefix`/`primaryclass` so biblatex
   styles can show both. Do not add a second entry for the preprint.
2. **Keys** are the first author's surname, the four-digit year of the cited
   version, and a one-word slug, all lower-case ASCII: `lim2021tft`,
   `bentaieb2012recursive`. A surname with a space or particle loses it
   (`Ben Taieb` becomes `bentaieb`); an accented surname is transliterated
   (`Lütkepohl` becomes `lutkepohl`).
3. **Entry types** follow the venue: `@article` for journals, `@inproceedings`
   for NeurIPS, ICLR and EMNLP (volume for NeurIPS), `@book`, `@techreport`
   for working papers, `@misc` for preprints with no published version,
   `@software` for code. Under plain BibTeX, `@software` falls through to the
   style's default type (`misc`); under biblatex it is rendered as software.
4. **Fields** use TeX accents (`{\"u}`, `{\'e}`) rather than UTF-8, protect
   proper nouns and acronyms in titles with braces (`{Hilbert}`, `{RNN}`),
   and give a DOI wherever one exists. Field names are aligned on `=` within
   an entry.
5. **Comments** above each entry are two lines at most: `% PDF:` naming the
   file in `papers/` and the version it is, and `% Cited by:` naming the
   documents. Keep both current when a PDF is added or a document gains a
   citation.
6. **Grouping** is by theme, in the numbered sections of the file. A new
   entry goes in the section it belongs to, not at the end.

## Adding a reference

1. Add the entry to `references/bibliography/master.bib` in its section, with
   its two comment lines.
2. If the paper is kept, put the PDF in `references/papers/` under a
   topic name in `Title-Case-With-Hyphens.pdf`, add the `% PDF:` line, and add
   a row to the table above.
3. If the documentation will cite it, copy the entry verbatim into
   `docs/source/references.bib` and cite it with `:cite:p:` or `:cite:t:`;
   the bibliography renders once, on `project/citing.rst`.
4. If the journal manuscript will cite it, copy the entry verbatim into
   `papers/journal/bib/references.bib`.
5. Check that the master still parses:

   ```bash
   uv run --with pybtex python -c "from pybtex.database import parse_file as p; print(len(p('references/bibliography/master.bib').entries), 'entries')"
   ```

   and, if the manuscript is set up, run its `make` target so BibTeX reports
   any field the style needs and the entry lacks.

## Note on the PDFs

The PDFs are kept for the maintainer's own reading and reproduction of the
papers' arguments; they are not part of the software and are not
redistributed with it. Those from arXiv, HAL, NeurIPS, the Federal Reserve
Bank of Richmond, the University of Tokyo METR series, Monash University, and
the CC BY Physica D article are openly licensed. The Econometrica, Journal of
Business & Economic Statistics, Journal of Monetary Economics, Neural
Computation, IEEE Transactions on Neural Networks and Computer Music Journal
files are publisher copies and remain under their publishers' terms.
