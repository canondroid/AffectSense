# Notification System Design Rationale

**AffectSense — Cognitive Stress Detection, macOS**
Samar Jalil Siddiqui

---

## The Core Tension

Health-monitoring systems face an uncomfortable paradox: they must interrupt users to be useful, yet interruption is itself a stressor. Gloria Mark, Daniela Gudith, and Ulrich Klocke (CHI 2008) found that knowledge workers interrupted during tasks compensate by working faster, but at the direct cost of higher self-reported stress. A cognitive-stress notification delivered carelessly risks compounding the very state it is trying to address. Every design decision in AffectSense's notification system is a response to this tension.

## Tiered Escalation Over Uniform Alerting

I chose a four-tier model (Tier 0: 0–44, Calm; Tier 1: 45–64, Caution; Tier 2: 65–79, Warning; Tier 3: 80–100, High Stress) rather than a single binary alert because stress is not binary. Lazarus and Folkman's (1984) transactional model of stress appraisal holds that a person's response to a stressor is proportional to their cognitive appraisal of its severity, a system whose response does not match that appraisal feels alien and is ignored. Tier 0 produces no notification at all; the most common system state is silence. This is deliberate: if every 30-second inference cycle produced a visible signal, users would habituate rapidly (alert fatigue) and disable the system, defeating its purpose entirely.

At the opposite extreme, Tier 3 activates macOS Do Not Disturb. This decision draws on Mark et al.'s finding that interrupted workers in high-stress states experience compounding cognitive load when additional notifications from other applications arrive. Removing those secondary interruptions at peak stress is as important as the notification itself. The 15-minute same-tier cooldown and 5-minute any-intervention cooldown enforce a minimum recovery window between alerts, preventing the rapid re-notification cycle that pushes users from irritation into reactance (discussed below).

## Peripheral Delivery and Bounded Interruption Time

The notification appears as a slide-in panel from the right screen edge. It is deliberately non-modal: it does not steal keyboard focus, capture the cursor, or block any underlying window. This design follows Adamczyk and Bailey's (CHI 2004) finding that interruptions delivered at task breakpoints, the natural transitions between meaningful units of work incur significantly lower resumption cost than mid-task interruptions. A peripheral slide-in that doesn't interrupt the current keypress allows a user to finish their current sentence or code statement before shifting attention; a modal dialog does not.

The notification auto-dismisses after 45 seconds. This bounded interruption window serves two purposes. First, it prevents a "guilt loop", the experience of a persistent alert demanding acknowledgement that cannot be safely ignored, which Iqbal and Bailey (CHI 2008) identified as a primary driver of interruption frustration. Second, it accepts that the user may be in a flow state where attending to the notification at all is the wrong choice. A notification that disappears quietly is not a failure; it is a record that the system noticed something the user chose to defer.

## Autonomy Preservation and Explainability

The most consequential design decision is how the notification positions the user relative to the system. Brehm's (1966) theory of psychological reactance holds that people experiencing a perceived threat to their freedom of choice generate motivational resistance. They do precisely what they feel coerced not to. In HCI, Ehrenbrink et al. (2019) demonstrated that systems perceived as controlling trigger this resistance: users dismiss, ignore, or uninstall. Three decisions in AffectSense directly address this risk.

First, every notification body displays the top three SHAP-attributed signals driving the score, for example, "Blink rate has dropped, suggesting visual fatigue" or "Typing errors have increased above your baseline." This satisfies Guideline 8 from Amershi et al. (CHI 2019), "Make clear why the system did what it did." Transparency converts a demand into an explanation; a user who understands the evidence can engage critically rather than comply reflexively or dismiss defensively.

Second, the "That's wrong" button is an explicit override mechanism. Its presence signals that the system treats the user as the final authority over their own internal state, not a passive subject of classification. Each correction is stored and contributes to a background retraining cycle: after ten corrections, the personal XGBoost model is retrained on the user's own labelled data. This realises the JITAI (Just-In-Time Adaptive Intervention) design principle articulated by Nahum-Shani et al. (2018), that one-size-fits-all intervention thresholds systematically misfire because individual stress baselines, contexts, and trajectories vary enormously. Personalisation through correction is not a convenience; it is the mechanism by which the system earns the right to intervene.

Taken together, these decisions position AffectSense not as a behaviour-change enforcer but as an attentive informational partner. One that explains itself, accepts correction, and respects the cognitive cost of the very attention it requests.

---

## References

Adamczyk, P. D., & Bailey, B. P. (2004). If not now, when? The effects of interruption at different moments within task execution. *Proceedings of CHI '04*, 271–278. ACM.

Amershi, S., Weld, D., Vorvoreanu, M., Fourney, A., Nushi, B., Collisson, P., … Horvitz, E. (2019). Guidelines for human-AI interaction. *Proceedings of CHI '19*, Paper 3. ACM.

Brehm, J. W. (1966). *A theory of psychological reactance*. Academic Press.

Ehrenbrink, P., Möller, S., & Siegert, I. (2019). *The role of psychological reactance in human–computer interaction*. Springer.

Iqbal, S. T., & Bailey, B. P. (2008). Effects of intelligent notification management on users and their tasks. *Proceedings of CHI '08*, 93–102. ACM.

Lazarus, R. S., & Folkman, S. (1984). *Stress, appraisal, and coping*. Springer.

Mark, G., Gudith, D., & Klocke, U. (2008). The cost of interrupted work: More speed and stress. *Proceedings of CHI '08*, 107–110. ACM.

Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., & Murphy, S. A. (2018). Just-in-time adaptive interventions (JITAIs) in mobile health: Key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*, *52*(6), 446–462.