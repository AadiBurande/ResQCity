# ResQCity — Claude Code setup guide

## 1. What's in this kit

```
resqcity-claude-kit/
├── CLAUDE.md                     project memory: architecture, conventions, research rules
├── BUILD_PROMPT.md               Prompt 0 (kickoff) + 11 phase prompts to paste one by one
├── SETUP.md                      this file
└── .claude/
    ├── skills/
    │   ├── or-model-builder/     formulation-first OR modelling + brute-force tests
    │   ├── run-experiment/       experiment design, CRN, paired stats, Holm, figures
    │   ├── vision-eval/          datasets, video-level splits, calibration, delay/false-alarm metrics
    │   └── research-writeup/     IMRaD write-up with numbers pulled from results/
    └── agents/
        ├── or-model-reviewer.md  checks model code against its LaTeX formulation
        └── research-auditor.md   independent methodology examiner
```

## 2. Install the tools on your machine

- **Claude Code**: follow the official setup page: https://code.claude.com/docs/en/setup
- **Python 3.11** and **uv** (package manager)
- **Node.js 20** and **pnpm** (dashboard)
- **Docker** (for MediaMTX, the RTSP server that simulates CCTV cameras) and **ffmpeg**
- **Git** and a GitHub repo for the project
- **Google Colab** account for GPU training (free tier is enough to start)

## 3. Create the repo and drop in the kit

```bash
mkdir resqcity && cd resqcity && git init
# copy CLAUDE.md, BUILD_PROMPT.md, SETUP.md and the .claude/ folder from this kit here
git add . && git commit -m "Add Claude Code project kit"
claude
```

Claude Code automatically loads `CLAUDE.md`, the project skills in `.claude/skills/`, and the
subagents in `.claude/agents/`. Check with `/skills` and `/agents`.

## 4. Skills and plugins to install

### Already built into Claude Code (no install)
| Skill / feature | Use it for |
|---|---|
| Plan mode (Shift+Tab) | Start every phase; Claude plans before editing |
| `/run-skill-generator` | Run once after Phase 8 so Claude records how to launch the whole stack |
| `/run` and `/verify` | Launch the app and confirm each change really works, not just that tests pass |
| `/code-review` | Review each phase's changes before committing |
| `/debug` | When something breaks and the cause isn't obvious |
| `/doctor` | Check your Claude Code setup if anything misbehaves |
| `/claude-api` | Only if you build the optional LLM-based 112 call triage (Phase 4) |

### From Anthropic's skills repository
Run these inside Claude Code:

```
/plugin marketplace add anthropics/skills
/plugin install example-skills@anthropic-agent-skills
/plugin install document-skills@anthropic-agent-skills
```

- **example-skills** includes `webapp-testing` (Playwright end-to-end tests for the dashboard,
  used in Phase 9) and other examples.
- **document-skills** gives `docx`, `pptx`, `pdf` and `xlsx`, for your final report (Word/PDF),
  viva presentation (PowerPoint) and results tables (Excel).

### From the official plugin marketplace
```
/plugin install skill-creator@claude-plugins-official
```
- **skill-creator**: test and improve the four project skills in this kit, or create new ones
  if you notice yourself repeating instructions.
- **frontend-design**: open `/plugin`, browse the official marketplace and install
  `frontend-design`. Used in Phase 9 so the dashboard looks like a real control room, not a
  default template.

If an install fails with "marketplace not found", run
`/plugin marketplace add anthropics/claude-plugins-official` first, then retry.

### Custom project skills (included in this kit)
| Skill | When Claude uses it |
|---|---|
| `or-model-builder` | Every optimization model (Phase 6): formulation doc first, then code, then brute-force tests |
| `run-experiment` | Phase 10: experiment configs, common random numbers, paired tests, Holm correction, figures |
| `vision-eval` | Phase 3: datasets, video-level splits, calibration, detection delay and false-alarm metrics |
| `research-writeup` | Phase 11: methodology and results sections with numbers pulled from result files |

### Custom subagents (included in this kit)
| Subagent | When to call it |
|---|---|
| `or-model-reviewer` | After each OR model: "Ask or-model-reviewer to review the joint dispatch MILP" |
| `research-auditor` | Before and after experiments, and on the final write-up |

## 5. How to run the build

1. Start Claude Code in the repo, switch to plan mode, paste **Prompt 0** from `BUILD_PROMPT.md`.
2. Answer its questions, approve the master plan.
3. For each phase: start a fresh session (or `/clear`), paste the phase prompt, approve the
   plan, let it build, check the acceptance criteria yourself, then `/code-review` and commit.
4. Use `/verify` after phases 3, 7, 8 and 9 — these are the ones where "tests pass" and
   "it actually works" most often differ.

## 6. Things only you can do
- Download the accident datasets (TAD, CADP, MADP, IEEE DataPort); some require accepting licences.
- Confirm the survival-curve and reporting-delay parameters from papers and add citations
  (they are marked `UNVERIFIED` until you do).
- Sanity-check travel times between a few Pune locations you know.
- Send the college letter to Pune Smart City / Pune Traffic Police asking for anonymized data —
  if it arrives, Claude can swap synthetic data for real data without changing the design.

## 7. Suggested timeline (12–14 weeks)
| Weeks | Phases |
|---|---|
| 1–2 | 0–2 (plan, scaffolding, road network) + literature review |
| 3–5 | 3–4 (cameras, vision, fusion) |
| 5–7 | 5–6 (forecasting, OR models) |
| 7–9 | 7–8 (simulator, backend) |
| 9–10 | 9 (dashboard, demo) |
| 11–12 | 10 (experiments, statistics) |
| 13–14 | 11 (write-up, reproducibility, viva prep) |
