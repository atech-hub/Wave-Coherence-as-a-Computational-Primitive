# References

Full citation details for related work referenced throughout the repository.

---

## Related Work

**Listopad, S. (2025a).** *Wave-Based Semantic Memory: A Phase-Aware Alternative to Vector Retrieval.* arXiv:2509.09691. https://arxiv.org/abs/2509.09691
Phase-aware retrieval system scoring document relevance using resonance-based coherence rather than cosine similarity over flat embeddings. Validates that phase-encoded scoring outperforms standard vector retrieval for relationship-sensitive queries. The present work extends this from the retrieval layer to the encoding substrate itself.

**Listopad, S. (2025b).** *Phase-Coded Memory and Morphological Resonance.* arXiv:2511.11848. https://arxiv.org/abs/2511.11848
Integrates resonance-based retrieval into inference loops — moving beyond static scoring toward dynamic phase-coded memory during generation.

**Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019).** *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.* ICLR 2019. https://arxiv.org/abs/1902.10197
Validates from the knowledge graph side that rotational geometry on the unit circle is a natural substrate for encoding relational structure — the same insight this work arrives at from the database query side.

**Moriya, T. (2025).** *Surface-Enhanced Coherence Transform: A Framework for Structured Coherence Decomposition.* arXiv:2505.17754. https://arxiv.org/abs/2505.17754
Decomposing aggregate coherence into surface and propagation components recovers structure that ensemble averaging destroys. Admissibility conditions (Hermiticity, positive-definiteness, normalisation, spectral scaling) provide the formal contract validated in Test 22. The structural parallel is exact: aggregate coherence loses information the same way cosine similarity does in Test 21.

**Wang, L. (2025).** *Defierithos: The Lonely Warrior Rises from Resonance — A Self-Resonance Architecture Beyond Attention.* Submitted to NeurIPS 2025.
Replaces transformer self-attention entirely with wave interference and phase superposition. Tokens become waveform imprints with spectral signatures; semantic matching operates via coherence estimation between sub-bands. Uses partial resonance (local spectral matching) analogous to the sub-linear bucket selectivity in Tests 18-19. Simulation-based results; not yet validated on real hardware.

**Pal, A., Ghosh, A., Zhang, S., Hill, L., Yan, H., Zhang, H., Bi, T., Alabbadi, A., & Del'Haye, P. (2024).** *Linear and Nonlinear Coupling of Light in Twin-Resonators with Kerr Nonlinearity.* arXiv:2404.05646v2. https://arxiv.org/abs/2404.05646
Coupled Lugiato-Lefever equation (Eq. 1) provides the self-phase modulation (i|E|²E) and cross-phase modulation (i·2|E'|²E) terms adapted in Phase 21's Kerr-ODE layer. The physical substrate — coupled resonant cavities exchanging energy through intensity-dependent phase shifts — is the direct analog of harmonic bands interacting through amplitude-squared coupling.

**Kato, S., Wang, P., Koike-Akino, T., Fujihashi, T., Mansour, H., & Boufounos, P. (2024).** *Multi-Band Wi-Fi Neural Dynamic Fusion.* arXiv:2407.12937v1 (ICASSP 2024). https://arxiv.org/abs/2407.12937
Demonstrates neural ODEs as practical computation primitives for multi-band signal processing. Phase 21 applies the same principle but replaces learned neural dynamics with physics-based Kerr dynamics operating on harmonic transformer embeddings.

**Zelenka, O., Kopáček, O., & Lukes-Gerakopoulos, G. (2024).** *Combining Machine Learning with Recurrence Analysis for resonance detection.* arXiv:2412.19683. https://arxiv.org/abs/2412.19683
Recurrence quantifiers carry resonant imprints regardless of dimensionality. LSTM-automated detection validates resonance as a structured, recoverable signal across domains. Parallels the harmonic sweep (Test 21), where per-channel coherence recovers relationships that aggregate similarity destroys.

**Luo, Z. et al. (2025).** *DyMixOp: A Neural Operator Designed from a Complex Dynamics Perspective with Local-Global Mixing for Solving PDEs.* arXiv:2508.13490. https://arxiv.org/abs/2508.13490
Introduces the Local-Global Mixing (LGM) transformation, inspired by convective nonlinearity in turbulence (u·∇u). LGM multiplicatively couples local fine-scale features with global spectral information to capture nonlinear interactions while mitigating spectral bias. Their ablation (Table 2) showed multiplicative fusion outperforms additive by 2-5x across PDE benchmarks. Phase C tested three LGM variants (multiplicative, gated, additive) within the Kerr-ODE layer — additive coupling through a learned bottleneck (the maestro pattern) proved most effective for harmonic band coordination, closing 1.80pp of the MLP gap. The multiplicative fusion that worked for PDEs hurt when applied to our ODE derivative, confirming that the optimal fusion mode is domain-dependent.
