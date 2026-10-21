# Limb lead + chest lead sets — predictions recorded before training

Date recorded: 2026-10-21 21:31 (+03). No model on the lead sets below had been trained when
these predictions were written. They answer comment 4 of reviewer 1 (CIBM-S-26-08400): the four
reduced sets of the main analysis contain no lateral chest lead, so their failure to retain
hypertrophy and repolarization changes is expected by design.

The predictions draw on the single-lead results that were already known (Table 3 of the
manuscript: best single lead V1/V2 for right bundle branch block, V5 or aVL for left ventricular
hypertrophy, V4 to V6 for repolarization changes, II or aVF for left axis deviation). They are
fixed before any model on the new sets is trained, and every prediction will be reported as
confirmed or not confirmed.

## Lead sets

| Set | Device analogue | Rationale |
|---|---|---|
| I + V1 | smartwatch or handheld recorder with a second, right precordial position | right precordial lead for right bundle branch block; atrial activity for atrial fibrillation and flutter |
| I + V5 | as above, lateral precordial position | lateral lead for left ventricular voltage and lateral repolarization |
| II + V5 | patch or handheld with an inferior limb lead and a lateral chest position | axis (lead II) with the lateral lead |

## Protocol (unchanged from the primary analysis)

- Cohorts: Chapman, Ningbo, Georgia, PTB-XL; same splits and narrow (primary) labels.
- Models: SE-ResNet and InceptionTime, five seeds each, same hyperparameters, seed ensembles;
  gradient-boosted trees on the concatenated hand-crafted features of the two leads (Chapman and
  PTB-XL, one seed); broad-label sensitivity analysis with SE-ResNet, one seed.
- Sufficiency: AUPRC gap to the twelve-lead model, 95% two-level bootstrap interval (seeds and
  test recordings), margin 0.05; margins 0.02 / 0.10 and relative margins as sensitivity analyses.
- Flags † and ‡ as in the main analysis; flagged classes are reported but not counted for or
  against a prediction.
- Comparisons: (a) each new set against the twelve-lead model (sufficiency); (b) paired two-level
  bootstrap difference of gaps between each new set and leads I + II, the two-lead set of the main
  analysis. "Smaller gap than I + II" means the 95% interval of gap(new) − gap(I + II) lies below 0.
- A prediction stated for "both networks" is confirmed for a cohort only if it holds for
  SE-ResNet and for InceptionTime.

## Predictions

**P1 Right bundle branch block.** I + V1 is sufficient in at least three of the four cohorts with
both networks. I + V5 and II + V5 are sufficient in at most two cohorts.

**P2 Left ventricular hypertrophy.** I + V5 and II + V5 have a smaller gap than I + II in all four
cohorts. They are sufficient in Chapman and Ningbo, where the label is left ventricular high
voltage, and not in Georgia or PTB-XL, where the label is a diagnosis of hypertrophy. I + V1 is not
sufficient in Georgia or PTB-XL.

**P3 Repolarization changes (T-wave change, ST-T change).** I + V5 and II + V5 have a smaller gap
than I + II for T-wave change in all four cohorts. For T-wave change they are sufficient in Chapman
and Ningbo and not in Georgia or PTB-XL. I + V1 does not have a smaller gap than I + II for T-wave
change in any cohort.

**P4 Left axis deviation.** II + V5 is sufficient in at least two of Chapman, Ningbo and Georgia.
I + V5 and I + V1, which lack a lead with a vertical frontal component, are not sufficient in any
cohort. In PTB-XL, on the test recordings with a coded heart_axis, II + V5 is sufficient.

**P5 Atrial fibrillation.** I + V1 and II + V5 are sufficient in at least two of the three cohorts
that code atrial fibrillation, with both networks.

**P6 Atrial flutter.** I + V1 has a smaller gap than I + II in Chapman; sufficiency is not predicted
in any cohort.

**P7 Rate-defined rhythms (sinus rhythm, bradycardia, tachycardia).** All three sets are sufficient
wherever the label is not flagged ‡.

## Use in the manuscript

The results will be reported as an additional analysis in Section 3.3, Table 3, Figure 2 and the
supplement, whichever way they fall. Section 4.4 will state the lateral-chest-lead conclusion on
these data: if P2 and P3 hold, that a limb lead with a lateral chest lead restores the lost
information for those labels; if not, that two leads, even with a lateral chest lead, do not
suffice for them.

## Addendum — reading of the claims (recorded 2026-10-21 21:34 +03, before any result of these sets was inspected)

Training was running when this was written; no metric of the new sets had been read. To remove an
ambiguity in the predictions above: a positive claim (a set is sufficient, or has a smaller gap than
I + II) holds in a cohort only if it holds for both networks; a negative claim (a set is not
sufficient, or does not have a smaller gap) holds only if it holds for neither network. This applies
to every prediction, including those that do not name the networks. Flagged classes (dagger or
double dagger) are not counted. The evaluation is implemented in src/review2_analyses.py (prereg).
