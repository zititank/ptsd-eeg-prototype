# References and how each one was used

Most citations in project reports only support a sentence. Here each source that shaped the work is traced from
**claim** to **concrete use** (a feature choice, a band edge, a test, a decision) to **where** that lives, and
whether **our data agreed**. Sources that only framed the discussion are listed more briefly in section 3.

Keys: `[H#]` were verified against PubMed, PMC or the publisher page during the hackathon; `[B#]` were resolved via
Crossref and read in Europe PMC during the audit that followed; `[L1]`, `[T1]`, `[M1]`, `[D#]` were added for the
public release. EC = eyes closed, EO = eyes open.

## 1. Sources that shaped a decision

### Feature design

**[H1] Kovacevic N, Meghdadi A, Berka C.** Characterizing PTSD using electrophysiology: towards a precision medicine
approach. *Clin EEG Neurosci* 2025. doi:10.1177/15500594241309680.
- **Claim.** EC individual alpha frequency is higher in PTSD (10.07 vs 9.24 Hz, effect size 1.02); relative power
  over 1–40 Hz and an individualised IAF ± 2 Hz band; other effects were EO.
- **Use.** Design template of Set A: IAF peak search in 7–14 Hz, relative power with a 1–40 Hz denominator, the
  IAF ± 2 Hz band (`rest_rel_alpha_IAF`).
- **Where.** `src/ptsd_eeg/features.py` (`find_iaf`, `relative_bandpower`, `rest_spectral_features`).
- **Agreed?** No. IAF did not differ against valid controls, and the model gives IAF a small, sign-unstable weight
  (positive in 24.3% of fits).

**[H2] Wahbeh H, Oken BS.** Peak high-frequency HRV and peak alpha frequency higher in PTSD. *Appl Psychophysiol
Biofeedback* 2013;38(1):57–69. doi:10.1007/s10484-012-9208-z.
- **Claim.** Higher peak alpha frequency in 59 veterans with PTSD vs 27 without.
- **Use.** Second source supporting IAF as a candidate feature.
- **Where.** Set A (`rest_IAF`).
- **Agreed?** No (as above).

**[H11] Klimesch W.** EEG alpha and theta oscillations reflect cognitive and memory performance. *Brain Res Rev*
1999;29(2–3):169–195. doi:10.1016/S0165-0173(98)00056-3.
- **Claim.** IAF varies with age and state; bands should be anchored to IAF; drowsiness lowers upper alpha.
- **Use.** Justifies the individualised band; basis of the drowsiness hypothesis.
- **Where.** `features.py`; [limitations.md](limitations.md) item 8.
- **Agreed?** Partly: low scores within PTSD follow the drowsiness profile (not tested with sleep data).

**[H14] Corcoran AW, Alday PM, Schlesewsky M, Bornkessel-Schlesewsky I.** Toward a reliable, automated method of
individual alpha frequency quantification. *Psychophysiology* 2018;55(7):e13064. doi:10.1111/psyp.13064.
- **Claim.** Peak-based IAF fails in some subjects; centre-of-gravity estimation is more robust.
- **Use.** Why IAF may be missing: we keep the frozen peak rule (peak ≥ 1.5 × baseline) and impute missing IAF with
  the training-fold median instead of forcing a value.
- **Where.** `find_iaf`; in-fold imputation in `src/ptsd_eeg/transforms.py`.
- **Agreed?** Yes: IAF was missing mainly in control batch B, which has no alpha peak.

**[H19] Whitham EM, Pope KJ, Fitzgibbon SP, et al.** Scalp electrical recording during paralysis: EEG frequencies
above 20 Hz are contaminated by EMG. *Clin Neurophysiol* 2007;118(8):1877–1888. doi:10.1016/j.clinph.2007.04.027.
- **Claim.** Most scalp power above 20 Hz is muscle.
- **Use.** Beta is measured at O1/O2 only, never at T3/T4 (over the temporalis muscle); beta results are read with
  caution.
- **Where.** `rest_spectral_features` (occipital average only).
- **Agreed?** Consistent: beta carries no stable weight (positive in 48.9% of fits).

**[H18] Mathewson KE, Harrison TJL, Kizuk SAD.** High and dry? Comparing active dry EEG electrodes to active and
passive wet electrodes. *Psychophysiology* 2017;54(1):74–82. doi:10.1111/psyp.12536.
- **Claim.** Dry electrodes add noise, but spectra remain measurable.
- **Use.** Justifies spectral rest features rather than single-trial evoked responses on a dry headset.
- **Where.** [science.md §2–3](science.md).
- **Agreed?** Yes for the batches outside control batch B ([data_integrity.md](data_integrity.md)).

**[H8] Toll RT, Wu W, Naparstek S, et al., Etkin A.** An electroencephalography connectomic profile of PTSD. *Am J
Psychiatry* 2020;177(3):233–243. doi:10.1176/appi.ajp.2019.18080911.
- **Claim.** Reduced source-space connectivity in PTSD (64 channels, EO).
- **Use.** Decision **not** to build connectivity features: six commonly referenced dry sensors cannot recover it.
- **Where.** Set A contains no connectivity feature.
- **Agreed?** Not testable.

### Expected direction (the test that exposed the artefact)

**[H4] Clancy K, Ding M, Bernat E, Schmidt NB, Li W.** Restless "rest": intrinsic sensory hyperactivity and
disinhibition in PTSD. *Brain* 2017;140:2041–2050. doi:10.1093/brain/awx116.
**[H5] Clancy KJ, Andrzejewski JA, Simon J, Ding M, Schmidt NB, Li W.** PTSD is associated with α dysrhythmia across
the visual cortex and the default mode network. *eNeuro* 2020;7(4). doi:10.1523/ENEURO.0053-20.2020.
- **Claim.** Suppressed posterior alpha in PTSD (EO), correlated with hypervigilance. Same group sizes in both
  papers, so probably one cohort.
- **Use.** The expected sign for alpha. Checking the model's weight against it revealed the reversed direction and
  triggered the data audit. Also the dose–response hypothesis in the data requests.
- **Where.** [science.md §3 and §6](science.md); [limitations.md](limitations.md) item 2 and request 2.
- **Agreed?** No: the model uses the opposite sign, which we attribute to control batch B.

**[B1] Nicholson AA, Densmore M, Frewen PA, et al., Lanius RA.** Homeostatic normalization of alpha brain rhythms
within the default-mode network and reduced symptoms in PTSD following EEG neurofeedback. *Brain Commun*
2023;5(2):fcad068. doi:10.1093/braincomms/fcad068.
- **Claim.** EC rest: lower relative alpha in PTSD, maxima in medial frontal gyrus and cuneus (occipital cortex).
  Caveats: normative-database controls, 24/38 medicated, source space.
- **Use.** The strongest EC evidence for the direction; the basis of the one-sided external hypothesis (ρ < 0).
- **Where.** The external protocol ([`protocols/README.md`](../protocols/README.md)); [external_check.md](external_check.md).
- **Agreed?** External check null (ρ −0.04 [−0.63, 0.59]); internally reversed.

**[H3] Newson JJ, Thiagarajan TC.** EEG frequency bands in psychiatric disorders: a review of resting state studies.
*Front Hum Neurosci* 2019;12:521. doi:10.3389/fnhum.2018.00521.
- **Claim.** 184 studies: no dominant spectral trend in PTSD; EC studies mostly null; bands overlap across disorders.
- **Use.** Prior expectation that a real EC sensor-level effect is small; power argument in the limitations.
- **Where.** [limitations.md](limitations.md) item 3; [science.md §3](science.md).
- **Agreed?** Yes: no difference against valid controls.

**[H6] Begić D, Hotujac L, Jokić-Begić N.** EEG comparison of veterans with combat-related PTSD and healthy subjects.
*Int J Psychophysiol* 2001;40(2):167–172. doi:10.1016/S0167-8760(00)00153-7.
**[H7] Jokić-Begić N, Begić D.** Quantitative EEG in combat veterans with PTSD. *Nord J Psychiatry* 2003;57(5):351–355.
doi:10.1080/08039480310002688.
- **Claim.** Increased beta (and theta); alpha decreased [H7] or unchanged [H6].
- **Use.** Expected sign for `rest_rel_beta_O`.
- **Where.** [science.md §3](science.md).
- **Agreed?** No stable beta weight.

### Specificity groups and confounders

**[H23]** Somatic symptom disorder, 64-channel resting qEEG (2021), PMID 34638015.
**[H24]** Hyperexcitability of cortical oscillations in somatoform pain disorder: a resting-state EEG study (2019),
doi:10.1155/2019/2687150.
- **Claim.** Higher alpha in untreated somatic symptom disorder and in somatoform pain disorder.
- **Use.** Predicted that any "more alpha → PTSD" rule would flag somatoform patients; this is why somatoform FPR is
  part of the pre-registered score and why C4 (somatoform as negatives) was a candidate.
- **Where.** PROTOCOL_v4 ([`protocols/README.md`](../protocols/README.md)); [model_selection.md](model_selection.md).
- **Agreed?** Yes: v3 flagged 0.284 of somatoform folders; the shipped model 0.176.

**[H13] Chiang AKI, Rennie CJ, Robinson PA, van Albada SJ, Kerr CC.** Age trends and sex differences of alpha
rhythms including split alpha peaks. *Clin Neurophysiol* 2011;122:1505–1517. doi:10.1016/j.clinph.2011.01.040.
- **Claim.** Alpha frequency declines in old age and differs by sex (n = 1498).
- **Use.** Age and sex confounding analysis; the request for somatoform sex.
- **Where.** [ETHICS.md](../ETHICS.md) §3; [limitations.md](limitations.md) item 5 and request 3.
- **Agreed?** Consistent: the 65+ group has low alpha and scores low; the model's inputs carry age information
  within controls (R² 0.154).

**[H20] Mandema JW, et al.** Differences in intrinsic efficacy of benzodiazepines are reflected in their
concentration-EEG effect relationship. *Br J Pharmacol* 1992. doi:10.1111/j.1476-5381.1992.tb14229.x.
**[H21] Liley DTJ, Cadusch PJ, Gray M, Nathan PJ.** Drug-induced modification of the system properties associated
with spontaneous human EEG activity. *Phys Rev E* 2003;68:051906. doi:10.1103/PhysRevE.68.051906.
- **Claim.** Benzodiazepines lower alpha and raise beta.
- **Use.** Medication is listed as an undocumented confounder; medication data requested.
- **Where.** [limitations.md](limitations.md) item 10 and request 2.
- **Agreed?** Not testable (no medication data).

**[H22] Haneef Z, Levin HS, Frost JD, Mizrahi EM.** EEG and quantitative EEG in mild traumatic brain injury.
*J Neurotrauma* 2013 (PMC3638533).
- **Claim.** Mild TBI slows alpha and raises theta, resolving over months.
- **Use.** Confounder for IAF (would push it down, against the hypothesised PTSD effect).
- **Where.** [limitations.md](limitations.md) item 10.
- **Agreed?** Not testable.

**[H25] Scott JC, Matt GE, Wrocklage KM, et al.** A quantitative meta-analysis of neurocognitive functioning in PTSD.
*Psychol Bull* 2015;141(1):105–140. doi:10.1037/a0038039.
- **Claim.** Slower processing speed in PTSD (d = −0.59; 60 studies, 4108 participants).
- **Use.** The task-duration feature was analysed as a descriptive variant, with and without EEG, as the task
  required.
- **Where.** [science.md §10](science.md).
- **Agreed?** Yes in direction: task duration alone gives AUC 0.760 against valid controls. Kept out of the
  model because it flags 0.826 of the 65+ group.

**[B48] Salthouse TA.** The processing-speed theory of adult age differences in cognition. *Psychol Rev*
1996;103(3):403–428. doi:10.1037/0033-295X.103.3.403.
**[B49] Tombaugh TN.** Trail Making Test A and B: normative data stratified by age and education. *Arch Clin
Neuropsychol* 2004;19(2):203–214. doi:10.1016/S0887-6177(03)00039-8.
**[B51] Wisdom NM, Pastorek NJ, Miller BI, et al.** PTSD and cognitive functioning: importance of including
performance validity testing. *Clin Neuropsychol* 2014;28(1):128–145. doi:10.1080/13854046.2013.863977.
- **Claim.** Speed falls with age; timed number-search tests slow with age; effort explains group deficits.
- **Use.** The reason the task-duration feature stays out of the classifier.
- **Where.** [science.md §10](science.md).
- **Agreed?** Yes: the task-duration variants flag 0.826 of the 65+ group.

### Validation methodology

**[H28] Cawley GC, Talbot NLC.** On over-fitting in model selection and subsequent selection bias in performance
evaluation. *JMLR* 2010;11:2079–2107.
- **Claim.** Selecting hyperparameters on the evaluation data inflates performance.
- **Use.** Nested design: shrinkage is chosen inside each outer training fold only.
- **Where.** `src/ptsd_eeg/model.py` (`choose_inner`, `nested_loso`); [science.md §7](science.md).
- **Agreed?** Leakage tests pass (permuted AUC 0.349 inside [0.142, 0.580]).

**[H26] Parker BJ, Günter S, Bedo J.** Stratification bias in low signal microarray studies. *BMC Bioinformatics*
2007;8:326. doi:10.1186/1471-2105-8-326.
**[H27] Airola A, Pahikkala T, Waegeman W, De Baets B, Salakoski T.** A comparison of AUC estimators in small-sample
studies. *PMLR* 2010;8.
- **Claim.** Pooled leave-one-out AUC is negatively biased under no signal.
- **Use.** Interpreting C1's permutation result below the null range (not leakage) and reporting AUCs below 0.5
  as they are, never inverted.
- **Where.** [model_selection.md](model_selection.md).
- **Agreed?** Yes: v3 refitted without control batch B gave AUC 0.273, and C1's permuted AUC fell below the null.

**[B53] Varoquaux G.** Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage*
2018;180:68–77. doi:10.1016/j.neuroimage.2017.06.061.
- **Claim.** At about 100 samples, cross-validated accuracy has error bars of about ±10%.
- **Use.** Every metric is reported with a subject-bootstrap CI; model comparison uses paired bootstrap deltas.
- **Where.** `src/ptsd_eeg/bootstrap.py`; all results tables.
- **Agreed?** Yes: the CI of AUC against valid controls spans 0.256 to 0.604.

**[B52] Woo CW, Chang LJ, Lindquist MA, Wager TD.** Building better biomarkers: brain models in translational
neuroimaging. *Nat Neurosci* 2017;20(3):365–377. doi:10.1038/nn.4478.
- **Claim.** Biomarkers need staged generalisation tests across samples and sites.
- **Use.** Motivated the pre-registered external check and the "no claim without an external site" rule.
- **Where.** [external_check.md](external_check.md); [limitations.md](limitations.md).
- **Agreed?** Our sample sits at the first stage; the external check was null.

### Context for the result

<a id="l1"></a>
**[L1] Li Q, Coulson Theodorsen M, Konvalinka I, Eskelund K, Karstoft KI, Andersen SB, Andersen TS.**
Resting-state EEG functional connectivity predicts post-traumatic stress disorder subtypes in veterans.
*J Neural Eng* 2022;19(6):066005. doi:10.1088/1741-2552/ac9aaf, PMID 36250685. Code: DTU "glia" resting-state EEG
analysis.

- **Claim.** n = 202, 5 min eyes closed plus 5 min eyes open, research-grade EEG: balanced accuracy up to 62.9%.
- **Use.** A realistic ceiling for resting-EEG PTSD classification, cited to calibrate expectations.
- **Where.** [limitations.md](limitations.md#context-what-others-achieved); README.
- **Agreed?** Our valid-control AUC (0.435) is below that ceiling, as expected for a smaller, confounded dataset.

## 2. Reporting standards and data

<a id="t1"></a>
**[T1] Collins GS, Moons KGM, Dhiman P, et al.** TRIPOD+AI statement: updated guidance for reporting clinical
prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378.
doi:10.1136/bmj-2023-078378. Used for [tripod_ai_checklist.md](tripod_ai_checklist.md).

<a id="m1"></a>
**[M1] Mitchell M, Wu S, Zaldivar A, et al., Gebru T.** Model cards for model reporting. *Proceedings of FAT\* 2019*,
220–229. doi:10.1145/3287560.3287596. Used for [MODEL_CARD.md](../MODEL_CARD.md).

<a id="d1"></a>
**[D1] du Bois N, Bigirimana AD, Korik A, …, Coyle D.** EEG and psychological assessment datasets: Neurofeedback for
the treatment of PTSD. Mendeley Data v3, doi:10.17632/gsxphk87mc.3, CC BY 4.0. Data paper: *Data in Brief* 2022
(PMC9011039). Used for the external check and the public EEG trace figure.

**[D2] Queensland University of Technology.** Post-traumatic growth EEG dataset, doi:10.25912/RDF_1684466012814.
Access by negotiation; not obtained.

## 3. Context-only sources

These framed the biology discussion but did not drive a feature, test or decision. Full entries with findings and
caveats are in the project's biology review.

| Key | Source | Used for |
|---|---|---|
| H9 | Halgren M, et al. The generation and propagation of the human alpha rhythm. *PNAS* 2019;116(47):23772–23782. doi:10.1073/pnas.1913092116 | where alpha comes from |
| H10 | Grandy TH, et al. Peak individual alpha frequency qualifies as a stable trait marker. *Psychophysiology* 2013;50(6):570–582. doi:10.1111/psyp.12043 | IAF as a trait (one session is enough) |
| H12 | Bazanova OM, Vernon D. Interpreting EEG alpha activity. *Neurosci Biobehav Rev* 2014;44:94–110. doi:10.1016/j.neubiorev.2013.05.007 | alpha indices and variability |
| H15 | Olbrich S, et al. EEG-vigilance regulation during the resting state in OCD. *Clin Neurophysiol* 2013;124(3):497–502. doi:10.1016/j.clinph.2012.08.018 | vigilance stages (drowsiness hypothesis) |
| H16 | Voytek B, et al. Age-related changes in 1/f neural electrophysiological noise. *J Neurosci* 2015;35(38):13257–13265. doi:10.1523/JNEUROSCI.2332-14.2015 | ageing flattens the spectrum |
| H17 | Donoghue T, et al. Parameterizing neural power spectra into periodic and aperiodic components. *Nat Neurosci* 2020;23:1655–1665. doi:10.1038/s41593-020-00744-x | caveat: band power mixes rhythm and 1/f background |
| B2 | Butt M, et al. The electrical aftermath: brain signals of PTSD. *Front Psychiatry* 2019;10:368. doi:10.3389/fpsyt.2019.00368 | review of EEG and ERP findings |
| B3–B16 | systems neurobiology of PTSD (fear circuitry, noradrenaline, cortisol, sleep, networks, dissociative subtype) | why rest EEG sees little of PTSD's best-known mechanisms |
| B17, B18 | Jensen & Mazaheri 2010; Klimesch et al. 2007 (alpha as inhibition) | interpreting alpha |
| B19 | Gao R, et al. 2017 (aperiodic slope and E:I balance) | why the aperiodic slope was not used |
| B32–B35 | frontal alpha asymmetry studies | not testable (no F3/F4) |
| B36–B42 | TBI, benzodiazepines, prazosin, alcohol, sleep loss | clinical overlays in PTSD |
| B43–B47 | trauma and functional somatic syndromes; somatic symptom models; depression comorbidity | why specificity is hard |
| B50 | Schuitevoerder S, et al. 2013 (cognition in older adults with PTSD) | age and PTSD add up |

Statements that could not be verified were not used to support any claim (for example, resting-EEG effects of
SSRIs and prazosin).
