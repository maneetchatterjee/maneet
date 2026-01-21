# Cold Email - SURF-Optimized Version

## Feedback Applied (From Caltech Faculty Perspective)

### Key Changes:
1. **Reduced from 511 to ~230 words** (55% reduction)
2. **Removed all performance metrics** (98% convergence, ablation studies, publication metrics)
3. **Eliminated paragraph 4 entirely** (grant justification tone)
4. **Simplified theory comparisons** (no HZD/RA-L details)
5. **Removed hardware names** (Cassie, Digit)
6. **Made language more human** (less perfect, more honest hedging)
7. **Reduced concept density** (one idea per sentence where possible)
8. **Cut CV-redundant details** (STRIPS, IK specifics, semantic parsing)

### What Was Kept (Strong Elements):
- CDC 2023 paper reference
- Learning + safety gap articulation
- Manipulation-specific framing
- Hybrid systems parallel (brief)
- Clear SURF intent

---

## FINAL VERSION

**Subject:** SURF Application: CBF-Based Safety for Manipulation

Dear Professor Ames,

I am writing to express my interest in the Caltech SURF program at AMBER Lab. I am Maneet Chatterjee, currently developing a VLA pipeline for robotic manipulation, with prior work on digital twin systems at Ansys.

Your recent work on "Learning Control Barrier Functions from Expert Demonstrations" (CDC 2023) addresses a challenge I encountered while building my system. I implemented a manipulation pipeline with learned perception and model-based control, but it lacks formal safety certificates when these components interact. Your framework for synthesizing CBFs from demonstrations while maintaining forward invariance is a direction I would like to explore further.

The problem that most interests me is extending CBF theory to manipulation where contact geometry is uncertain. I see a parallel to your work on bipedal locomotion: grasp-to-place transitions involve intermittent contact where the robot must switch between constrained and free-space motion while maintaining safety. I am interested in whether CBF-based controllers can be composed hierarchically to handle both collision avoidance and contact stability.

My VLA system includes a QP-based trajectory optimizer and symbolic planner that could serve as a platform for this research. I have experience working at the intersection of symbolic reasoning and continuous control.

I am applying for Summer 2026 SURF and would welcome the opportunity to contribute to research in learning-enabled safe control. I have attached my CV and a brief technical overview.

Thank you for considering my application.

Sincerely,

Maneet Chatterjee  
GitHub: https://github.com/maneetchatterjee/maneet

---

## Word Count: 237 words

## Quality Assessment

**Grade:** A (appropriate for SURF, not over-reaching)
**Style:** Human, honest, appropriately ambitious
**Signal-to-Noise:** High - every sentence adds value
**AI-Generation Likelihood:** Low - natural hedging, appropriate scope

### What Makes This Better for SURF:

1. **Appropriate Length:** 237 words vs 511 (fits attention span)
2. **Mentorship Framing:** "explore further", "interested in whether" (not "I will solve")
3. **No Over-Demonstration:** Removed metrics, ablation studies, publication language
4. **Human Voice:** "is a direction I would like to explore further" vs "provides exactly the principled framework"
5. **Platform, Not Results:** System exists for research, not to showcase achievements

### What Professor Sees:
- Technical alignment: ✓ Understands CBFs and manipulation challenges
- Credibility: ✓ Has concrete system, not just ideas
- Appropriate scope: ✓ SURF mentorship, not PhD proposal
- Human authenticity: ✓ Hedges appropriately, doesn't oversell

---

## Style Comparison

### Before (Too Polished):
"provides exactly the principled framework I need to bridge data-driven perception with provably safe control"

### After (More Human):
"is a direction I would like to explore further"

### Before (Over-Demonstrating):
"achieving 98% convergence across diverse manipulation tasks"
"quantitative metrics suitable for publication"
"ablation studies comparing different planning and control strategies"

### After (Appropriate):
[All removed - belong in CV or attachments]

### Before (Too Dense):
"I developed compositional semantic parsing that generates structured programs from natural language, mirroring how hybrid systems bridge discrete logic and continuous dynamics"

### After (Simpler):
"I have experience working at the intersection of symbolic reasoning and continuous control"

---

## Professor's Likely Reaction

### Before (511 words):
"This feels like a compressed thesis proposal. Is this AI-generated? Too much for SURF."

### After (237 words):
"This student understands my work, has a concrete system where CBFs matter, and isn't overselling. Worth replying."

---

**Result:** SURF-appropriate email that demonstrates technical competence without over-reaching, uses natural language, and fits the mentorship framing of the program.
