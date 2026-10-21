# InterviewBot

InterviewBot retrieves one saved experience at a time and produces either a STAR
story or a direct interview answer, with optional voice input and answer audio.

Install dependencies from `requirements.txt` and configure `OPENAI_API_KEY` on the
server. `INTERVIEW_CHAT_MODEL` optionally overrides the existing answer model.
AI requests have a 45-second client timeout and one SDK retry. The app provides
explicit retry controls after search, transcription, or answer failures.

## Choose the interview background

1. In the sidebar, choose **Job focus**: All roles, AI engineering, Transmission
   planning, Resource integration, or Project management.
2. Paste the posting into **Job description** (up to 20,000 characters). You can
   leave **Job focus** on **All roles** to use the description across your saved
   experience, or choose a focus to restrict the stories to that area. Optionally
   add your own interview priorities in **Job details**.
3. Click **Apply job focus**, then ask an interview question or click **Show
   stories for this role**.
4. Use **Give me another answer or story** to review the remaining matches.

The applied background persists across questions within the current session.
Applying a changed focus, description, or priorities clears previous suggestions
while preserving chat history. Reapplying identical settings keeps the current
suggestions. Clear the description and apply to stop using that posting. A new
browser session can reset the background. Form edits take effect only when applied.

The question and job background guide the query embedding. For long descriptions,
search uses an excerpt of up to 4,000 characters selected from throughout the
posting using question terms and known role terms. Shared job terms add a bounded
preference to stories with matching evidence; repeated terms do not amplify it.
The complete accepted description guides answer generation. No extra model call
is made when applying the description.

Focused stories must
also match topic phrases in their saved evidence; general direct answers remain
available for nonbehavioral questions. Topic matching is a heuristic, and one
story can fit several roles. If no stories match, the app asks you to adjust the
question or select All roles. The job description and details guide relevance
and phrasing; they are not evidence of qualifications or accomplishments. The
selected saved experience remains the factual source for the answer.

**Transmission planning** uses an additional evidence gate: the saved STAR facts
must establish a planning/study purpose, such as transmission assessments,
interconnection studies, expansion alternatives, planning models, or planning-team
work. Shared tools and methods—PSS/E, TARA, power flow, contingency analysis, and
steady-state analysis—cannot qualify a story by themselves. An EMS upgrade or
operator-training story therefore cannot outrank planning evidence just because
its saved question matches. Mixed planning/EMS rating reviews remain eligible
when planning-model work is explicit. Questions and aliases never establish the
domain. The same gate applies to questions, browsing, follow-up stories, and the
practice selector. All roles and Project management can still retrieve EMS work.

The gate is a conservative phrase-based relevance check, not a verification of
study results. New phrasings can need rule updates. Deploying a changed retrieval
policy clears old suggestions and active answer actions once; job settings and
conversation history remain. No story/embedding regeneration is needed for this
ranking change. Restart/redeploy the app and ask the question again to use it.

## Planning interview cheat sheet

In **Answer assistant**, open **Planning interview cheat sheet** for the 20 numbered
keywords, their assigned stories or approaches, and a short recall cue. The **Basis**
column distinguishes experience, experience plus a proposed approach, proposed
approaches, and illustrative scenarios. Choose a keyword and click **Show mapped
answer** to open its full saved answer and follow-up preparation. Selecting a keyword
alone does not submit a question. You can also type a shortcut such as `15 DISAGREEMENT`
into the question box.

Each summary lives in its primary record's optional `quick_view` field in
`manager_content.json`. The quick view follows the current runtime content and
preserves the existing one-to-one assignments and review controls. Q15's ratings
disagreement remains an explicitly requested backup, outside the 20-row primary
map. Stories marked **Needs correction** show that status and hide their recall
cues until the review status changes. Summaries are navigation metadata, excluded from answer text and embedding
inputs; changing a cue does not require an embedding refresh or an AI request.

## Answer actions and practice

The selected answer, its exact saved context, and generated audio survive normal
Streamlit reruns. **Show retrieved context**, **Make answer audio**, and the answer
download operate on that selection without repeating search or answer generation.
Audio is held in memory; use **Download answer audio** to save it on your device.
No new audio files are written to the server. Retrying an answer keeps the same
question and source. A failed request does not append an empty assistant message.

Answers now end with **Follow-up preparation**: two brief, story-specific answers
to **What did you learn?** and **What would you do differently?** For hypothetical
questions, the prompts are **What is the key takeaway?** and **What would you
challenge first?** Hindsight is conditional preparation, not a claim that an
additional improvement was already implemented. This section is included in the
answer download, optional session history, and answer audio.

Choose **Practice** to select a saved story and interview question, then type your
own answer or record it. Recording alone does not send practice audio to the AI
service: click **Transcribe my practice answer** first. Review or edit the transcript,
then click **Get practice feedback**. Feedback covers Situation, Task, Action,
Result, personal ownership, question/job relevance, unsupported claims, and one
probing follow-up. This is AI coaching, not fact verification or a hiring prediction.
The attempted answer and the selected source are kept separate in the request.
Previous attempts remain available in the practice history expander and can be
included in an explicitly downloaded session file.
If the selected source text changes, the current attempt and feedback reset;
feedback is tied to exact source hashes as well as the question and job context.

The target is 60–90 seconds. Valid WAV recordings display measured duration; typed
or edited transcripts display an estimate at 130 words/minute. The estimate is
not a measurement of the user's speaking rate. Recordings are limited to 15 MB,
typed practice answers to 6,000 characters, and questions to 2,000 characters.
The recorder's WAV option is supported by the
[upstream component](https://github.com/B4PT0R/streamlit-mic-recorder); use version
0.0.8 or newer. Browser microphone permissions and real-device recording still
need to be checked in the deployed environment.

## Story evidence and job coverage

**Story evidence and job coverage** exposes exact saved text, source array indexes,
source IDs where present, and SHA-256 text hashes. Canonical story IDs derive from
the normalized Situation field, so question aliases and corpus reordering keep
the same ID. Changed situations create new IDs; differently worded stories are
not automatically merged. All variants with the same situation remain visible.

Existing claims start as **Not reviewed** and outcomes as **Not confirmed**.
No provenance, outcome verification, or historical review date is invented. You
can save your own review status, outcome classification, and evidence/correction
notes in the current session; the timestamp records that user action in UTC.
**Reviewed by me** is a user assertion, not independent verification. Stories
reviewed in this app are bound to the exact hashes of all saved variants. Changed
source text or older imports without those hashes invalidate prior review/outcome
classifications and display a re-review notice. Imported dates are user-supplied
ISO 8601 timestamps with a timezone, not independently established review history.
Stories
marked **Needs correction** are excluded from subsequent searches and practice
until their review status is changed. An older generated answer remains visibly
flagged. Notes do not rewrite source claims; correct the original source and sync
the corpus before relying on disputed claims.

The coverage map retains posting lines and shows related story IDs with the exact
shared phrases. Unmatched lines remain visible for manual review. This conservative
heuristic neither verifies requirements nor measures qualification coverage as a
percentage. Source locations identify saved claims, not independently supporting
documents. Export the map as JSON for review.

## Save and restore a job session

Open **Save or restore a job session**, name the session, and explicitly click
**Download job session**. It saves applied job details and your story-review notes
to a JSON file on your device. Conversation and practice history are excluded by
default; opting in includes the latest 100 messages and 30 attempts. Audio,
credentials, source stories, embeddings, and internal prompts are never exported.
Session files can contain personal job and interview information; store them
privately. Nothing is automatically persisted on the server.

Upload that JSON file and click **Restore session** to resume on another browser
session. Imports are limited to 1 MB, validate field types and lengths, accept only
user/assistant text history, and reject unknown schema fields. Imported history
is displayed as history and is not used as factual evidence or model instructions.
Restoring clears stale candidates, selected answers, pending requests, and audio;
review controls reload the imported values. Future answers retrieve the current
corpus. Reviews for source IDs no longer in
the corpus remain saved notes rather than being applied to a different story.

## Content and validation

Job-focus changes do not alter `chunks_cleaned.json` or regenerate saved story
embeddings. `interview_retrieval.py` holds parsing, topic matching, ranking, and
candidate-reset logic separately from the Streamlit UI.

`interview_evidence.py` owns canonical IDs and evidence mapping;
`interview_sessions.py` validates explicit session files; `interview_services.py`
owns bounded requests and coaching output validation; `interview_workspace_ui.py`
renders evidence, practice, and session controls. `interview_runtime_content.py`
resolves current answer text, maps unchanged saved-vector rows, and invalidates
actions tied to older content. The saved cache retains its existing integrity
checks.

The app loads current managed answers from `manager_content.json` at runtime.
It validates the saved embedding cache against its original
`chunks_cleaned.json` before using any vectors. Only unchanged records receive
saved semantic scores; revised or new answers participate through their current
questions, aliases, keywords, and job-focus evidence until their vectors are
refreshed. A freshness notice reports this limitation. Updating answer wording
therefore no longer requires an embedding rebuild before it can be served.

After editing `manager_content.json`, refresh the saved corpus and index to
restore semantic matching for revised answers, with `OPENAI_API_KEY` configured
in the environment:

```sh
python interview_bot/generate_embeddings.py --sync-manager-content
```

The October 19, 2026 content update adds 20 transmission-planning interview
questions to `manager_content.json` (`planning-interview-01` through
`planning-interview-20`): 13 proposed technical/management approaches and seven
experience-based answers. Each imported entry records its question number, a
private source identifier, and the original source SHA-256. Experience-based
entries contain editable STAR fields and a full spoken answer.

Raw pasted drafts are local-only files under `.private/sources/`, excluded from
Git by the repository's `.gitignore`. They are not needed to run, test, or deploy
the bot. The `private:` source identifiers are provenance labels, not public
links or paths the application reads. Tests preserve source and prior-content
fingerprints without copying the raw drafts into fixtures.

The older October 19 text file has been removed from the current source tree,
but its earlier committed version remains in Git history; no history rewrite
has been performed. Ignoring the drafts does not make the answer JSON private:
`manager_content.json`, `chunks_cleaned.json`, and other committed story data are
readable if the repository is public. Private answer content requires private
repository/storage access, not a different file extension.

The portfolio now offers a browser-remembered footer shortcut instead of a
public interview-app card. Enable it once through `/#my-interview-bot` on your
portfolio. The shortcut is only a display preference: the app still needs
Streamlit Community Cloud's **Only specific people can view this app** setting,
with only your own account allowed. See the
[portfolio access instructions](../docs/portfolio.md#personal-interviewbot-shortcut)
for setup and a signed-out verification check. Local edits have not changed or
verified that hosted setting.

The October 20 director-level revision gives all 20 questions an exclusive answer
in the `ercot-planning-20` set. Each record has a numbered `question_key`, a
unique `scenario_id`, source evidence, preparation limits, and an
`answer_revision` fingerprint linking it to the user's supplied draft. The
original question and source fingerprint remain intact. The draft supplies
structure and technical depth; saved personal experience and the user's subsequent
clarifications supply historical claims. Nineteen answers were expanded toward
the supplied examples; Q14 retains the previously requested exact wording.
The revision fingerprints distinguish the full draft from the separate Q1 process
expansion and Q15 and Q20 examples. Unsupported project costs, outcomes, numerical acceptance limits,
universal SCR thresholds, and assumed expedited approvals are excluded.

| Question / shortcut | Exclusive example or proposed approach |
| --- | --- |
| 1 PROCESS | LCRA modeling; case building, study horizons, project tracking, large-load transition, four RPG tiers, and commissioning |
| 2 VIOLATION | LCRA conductor alternatives; thermal and voltage diagnosis |
| 3 NERC | Requirement-to-study evidence matrix for TPL, FAC, and MOD |
| 4 GENERATOR | Proposed 1,000 MW solar-plus-storage interconnection |
| 5 DATA CENTER | Proposed 750 MW load and phased service assessment |
| 6 IBR | Weak-network converter interactions and model-based diagnosis |
| 7 ECONOMICS | Incremental lifecycle comparison of expansion alternatives |
| 8 DISCREPANCY | Controlled reconciliation of consultant and ERCOT results |
| 9 STUDY TYPES | Selecting steady-state, transient, voltage, and fault studies |
| 10 SCENARIOS | Future-system cases, sensitivities, and investment triggers |
| 11 PRIORITIZE INVESTMENTS | LCRA five-year growth forecast and project ranking |
| 12 TOOLS | AELAB and contingency-generation automation |
| 13 LEADERSHIP | PEC annual assessment; HRML instability traced to South Texas GTC dispatch |
| 14 UNCERTAINTY | Conflicting AC/EMT evidence, missing load models, and minimax-regret investment |
| 15 DISAGREEMENT | Highway 29/Liberty Hill: ownership, regional need, and revised two-substation/three-line scope |
| 16 DEADLINES | Clearing the LCRA equipment-ratings backlog |
| 17 BAD NEWS | Reducing a capacitor proposal from seven banks to two |
| 18 ERROR | Initially missed consultant PI/PLL model issue; PSS®E/PSCAD mismatch resolved after retuning |
| 19 MENTORING | LCRA associate engineer; strengths and interests, Train/Trust/Automate/Gate |
| 20 STAKEHOLDERS | Proposed development/customer/ERCOT alignment of capacity, cost, schedule, and reliability |

Q1's October 20 process expansion retains its original question and LCRA anchor.
It covers working groups, coordinated cases and contingencies, six-year RTP and
longer-term scenarios, TSP-originated needs, GINR/MIS tracking, MOD-derived TPIT,
alternatives, all four RPG tiers, certification, and QSA/commissioning readiness.
Dated primary references accompany the saved answer. The wording distinguishes
the July 2026 LLIS cutoff and Batch Zero transition from CTP still under development,
uses the tier thresholds effective February 1, 2026, and explains that QSA performs
a stability assessment. Preparation notes preserve special tier provisions,
reporting exclusions, and the unresolved disposition of the requested 2026 RTP
exception. These are process facts checked as of October 20, 2026, not automatically
updated regulatory guidance. Q1 remains available through the existing runtime
content overlay without rewriting the saved embedding cache.

The October 21 Q1 addition explains Batch Zero's classification, screening,
allocation, commitment, and transmission-refinement steps, then distinguishes
the proposed recurring Batch 1+ process from the broader CTP framework. It uses
the October 15/20 ERCOT workshops for future-process details, including the latest
standalone Batch → commitment → full reliability sequence, planned LIOO portal,
and separate proposed PTA and RPG 2.0 pathways. Workshop proposals are not presented
as effective requirements or promised connection dates. The Q1 recall cue,
follow-up preparation, and aliases for questions comparing these processes are
updated together; the other 19 primary assignments remain unchanged.

No personal episode is reused within this set. Q12, Q13, Q15, Q16, Q17, Q18,
and Q19 use complete STAR answers with numbered Action steps. Their full answer
is assembled from the same four STAR fields with bold headings. Other answers
use detailed proposed steps, with personal anchors where supported. Q14 uses the
user's latest four-part hypothetical verbatim, including Markdown links and
formatting. It replaces Waterloo. A source-fingerprint regression verifies the
exact authored text without duplicating it in fixtures. The 750 MW load and SCR
1.2 are illustrative inputs. Source status and preparation notes identify known
technical and regulatory overstatements; preserved wording is not verified study
guidance. The separate coaching notes address model validation, proxy limitations,
IEEE 519 scope, conditional economics, STATCOM performance, CCN applicability,
and RAS responsibilities. Supporting references and revision provenance accompany
the answer. The preceding greenfield leadership draft was not inserted into Q13
or another question. Q17 acknowledges that the actual example concerns
an oversized equipment proposal, not a canceled or infeasible interconnection.
Q13 retains **potential** avoided annual external fees before internal costs.
Its stable HRML rerun follows redispatch within the applicable South Texas GTC
limit; that case correction does not establish that every required stress case
passes or that no transmission investment is needed. Q15 preserves the user's
**ERCOT 3** event description (autotransformer outage plus another outage), without
automatically relabeling it as NERC P6. ERCOT **preferred** the revised layout
during RPG review; this does not assert Board endorsement or construction.
The highest-capacity alternative objection is attributed to another stakeholder.
Q18 includes the user's initial DYRE diagnosis and confirmed resolution after
consultant PI/PLL retuning. The reported greater-than-10% discrepancy is specific
to that comparison, not a universal acceptance criterion. Q20 starts with the
three parties' priorities and uses an alignment matrix, shared study basis,
economic comparison, phased commitments, and governance; it is prospective and
does not reuse the PEC operational-cutover story.
Q19 distinguishes the colleague's achievements from the user's contribution,
and proposed diagnostic/delegation methods from claimed past assignments. It
includes negotiable work methods, non-negotiable engineering boundaries, and
review gates with retained decision accountability.

Entering a full authored question, an authored alias, or a shortcut such as
`12 TOOLS`, `TOOLS`, or `Q12` selects its assigned primary answer. `12. TOOLS` and
`#12 TOOLS` also work. A number/topic conflict such as `12 MENTORING` returns
no answer. Explicit assignments override job-focus preferences and return the
saved answer verbatim, without query embedding or answer-generation API calls.
"Different Story" cannot substitute another question's episode. Existing **Needs correction**
exclusions still apply: a flagged primary story yields no automatic substitute. An
unrecognized paraphrase follows the existing general retrieval/generation path;
it is not guaranteed to resolve to a dedicated assignment. Bare numbers are not
shortcuts. Audio, transcription, and practice feedback retain their own service
requirements.

Q15 also has an explicitly linked rating-submission backup. Ask `Q15` to get
the primary disagreement story, then click **Give me another answer or story**
to read that backup. The next click stops; it does not select an unrelated
episode. `Q15 backup`, `15 DISAGREEMENT backup`, or `another disagreement story`
requests the backup directly. Both scripts return their complete saved wording
without an API call. A backup marked **Needs correction** is excluded. A flagged
primary never automatically promotes its backup, although the user can still
request an unflagged backup explicitly.

Backups are standalone managed records with `backup_for` pointing to the primary
record's ID, their own unique `scenario_id`, and complete answer/provenance fields.
They omit `answer_set` and `question_key`, preserving exactly 20 primary mappings.
Their displayed question key comes from the parent. General semantic retrieval
excludes these backup records; only the parent question or a backup's authored
question/aliases can select them. Validation rejects missing parents, backup
chains, duplicate scenario IDs, and records declared as both primary and backup.

Assignment validation requires complete coverage of questions 1–20 and unique
question keys and scenario IDs. Tests also check the actual narrative anchors
for episode reuse, all authored question/alias routes, exact saved wording,
correction gating, and all 20 shortcuts through the UI. Source edits are loaded
by the runtime overlay; an embedding refresh is not required for these explicit
matches. The general generation prompt below applies to other questions.

All 20 primary answers and the Q15 backup have an authored `reflection` field,
serialized separately as `Reflection`. Explicit assignments append this text
under **Follow-up preparation** without an API call. The main `answer`, STAR
fields, and their source fingerprints remain unchanged, including Q14's exact
body. The optional field must contain nonempty text when supplied; older records
without it retain their existing saved-script behavior. A reflection-only change
preserves the story's identity but changes its source hash, invalidating stale
review conclusions, semantic scores, and current answer/audio state through the
existing source-version checks. Correction flags remain in force.

Other questions request the same concise follow-up format in their existing
answer-generation call, using only the selected source. This includes adapting
older prose `Reflection` fields. The prompt keeps lessons grounded in the saved
experience, proposed improvements conditional, and hypothetical scenarios
prospective. The application does not append a second saved section to generated
answers. As with other generated wording, prompt compliance requires review.

The generation prompt allows four to six focused paragraphs or short ordered
steps for multi-part technical questions, retaining methods, decision criteria,
and validation. Behavioral answers use bold Situation, Task, Action, and Result
headings. Multi-step technical and investment stories use three to five numbered
Action steps with descriptive labels and the reasoning behind each decision,
followed by a concise result paragraph or supported outcome bullets. The prompt
preserves existing numbered steps and avoids an introductory explanation of
STAR. Preparation instructions and editorial notes remain
evidence/coaching context rather than spoken script; material qualifications
such as potential fee avoidance remain in the answer. Optional uncertain dates
or amounts should be omitted instead of generating phrases such as "my
preparation record says" or "I still need to confirm." The prompt also requires
the relevant physical explanation, concrete checks, alternatives, and review
boundaries to survive paraphrasing.

Rewriting a story's Situation changes its content-based evidence ID. Previously
saved review notes remain associated with the previous story ID rather than
being treated as a review of the rewritten story.

Imported references, their dates, and preparation notes remain in the retrieved
context. They are user-provided preparation material, not independently verified
ERCOT/NERC requirements or verified personal outcomes. Generation and coaching
instructions retain hypothetical wording, potential versus realized savings,
effective-date caveats, and details that require confirmation. These prompt rules
do not guarantee model compliance. Other saved EMS examples remain operations
stories; changing Q14 does not relabel them as transmission-planning experience.

Deploy the source and runtime changes together, then restart/redeploy the app.
Current managed answers appear in retrieval and the evidence library; a content
change clears old selected answers and pending requests so a previous session
cannot continue answering from an obsolete candidate. History remains history.
The library identifies managed evidence by its current source file rather than
attributing revised text to the saved embedding corpus.

With the current source set, synchronization produces 167 chunks (117 existing
base chunks and 50 managed entries, including the explicit Q15 backup). An embedding refresh failure leaves the
saved cache intact; it does not roll the runtime answers back to old wording.
The app does not generate a full embedding cache during startup. To validate
the intended corpus without API calls, use:

```sh
python interview_bot/generate_embeddings.py --sync-manager-content --dry-run
```

All 20 primary records now have added reflection text, and the Q15 backup is
new relative to the saved semantic index; the 146 unchanged vectors can be reused.
Their latest text is available through the runtime source
overlay and explicit assignment/lexical retrieval after deployment.
Run the sync command in the configured environment, then
restart/redeploy or reload the saved embeddings to enable their refreshed
semantic scores. Do not run the standalone corpus writer first: it would
invalidate the saved cache before replacement vectors are ready.

The refresh reuses vectors for unchanged text and keeps active files intact if
embedding generation fails. Run local checks from the repository root:

```sh
python -m unittest discover -s interview_bot -p 'test_*.py'
streamlit run interview_bot/interview_bot.py
```

UI tests use mocked embedding, answer, microphone, and FAISS services. They verify
control wiring and state changes without making external API calls; they do not
measure live semantic-search or generated-answer quality.
Additional regressions cover persistent audio/context controls, safe retries,
session import boundaries, unreviewed evidence metadata, coverage gaps, and typed
and recorded practice. No external API calls are required for these checks.
