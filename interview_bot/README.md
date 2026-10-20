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

## Answer actions and practice

The selected answer, its exact saved context, and generated audio survive normal
Streamlit reruns. **Show retrieved context**, **Make answer audio**, and the answer
download operate on that selection without repeating search or answer generation.
Audio is held in memory; use **Download answer audio** to save it on your device.
No new audio files are written to the server. Retrying an answer keeps the same
question and source. A failed request does not append an empty assistant message.

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

The October 20, 2026 revision strengthens questions 1–10 with diagnostic steps,
mitigation tradeoffs, study deliverables, and validation checks adapted from the
first October 20 draft. Each revised entry retains its original source
fingerprint and adds `answer_revision` with the private source identifier,
question, SHA-256, and an
editorial note explaining the adaptation. The serialized **Answer Revision** field
keeps that provenance separate from the interview answer. The raw draft is not
ingested as an additional answer.

The revision deliberately excludes unsupported numerical limits, project cost
and schedule claims, universal SCR thresholds, and assumed expedited approvals.
The expanded October 20 draft adds answers to Q14–20. The final content blends
the supplied technical and management
structure with the PEC assessment, LCRA EMS upgrade, rating disagreement,
capacitor-bank alternative, dynamic-model verification, and mentoring examples.
All seven behavioral answers (Q12, Q13, Q14, Q15, Q17, Q18, Q19) and their four STAR
fields are rewritten as complete spoken stories: stakes, personal responsibility,
specific actions and engineering reasoning, then supported outcomes. Each full
answer is exactly the four STAR fields joined together, so retrieval cannot
pair a new narrative with old STAR wording. Generic future-process appendices
are removed. Revisions are traced through `answer_revision`, with material
outcome qualifications preserved.

A further October 20 coaching revision substantively strengthens Q8 and
Q10–20. Q19 now centers on teaching engineers to explain why a violation occurs:
trace post-contingency flow redistribution, identify the limiting facility,
investigate reactive reserves and controls, and compare remedies against the
demonstrated cause. Delegation includes agreed study assumptions, review points,
and accountability for the final decision. The P.E./promotion outcome remains,
but the uncertain one-year timeline is omitted.

The other behavioral answers now explain the actual diagnostic or decision
sequence: initialization versus disturbance failures, common study baselines,
repeatable acceptance tests, equipment identity and effective topology,
controlled seven-versus-two-bank comparisons, and model-owner correction
followed by reruns. Q8, Q10, Q11, Q16, and Q20 add reproducible reconciliation,
investment triggers, bottleneck-based prioritization, reviewer scheduling, and
explicit stakeholder choices. Q1–7 and Q9 retain their existing detailed
technical approaches. The new raw draft remains in ignored local storage;
revision metadata records its identifier and SHA-256 without publishing it.

Q14 subsequently replaces the EMS upgrade example with the user's preferred
transmission-planning investment scenario. It compares a major 345-kV expansion
with staged transformer/reactive support under uncertain load and generation,
using four numbered Action steps: model reconciliation, sensitivity analysis,
phased investment, and stakeholder alignment. The explicit question beginning
"Describe a situation in transmission planning" is an alias, and the original
question remains compatible. The old GE EMS alias is removed. Q14 now qualifies
for Transmission planning as well as All roles and Project management.

The result describes the phased recommendation and decision package. It does
not claim commissioned upgrades, quantified savings, or an ERCOT approval.
Required planning events remain distinct from exploratory stress cases, and
investment triggers account for routing, permitting, procurement, and
construction lead times. The revision fingerprint identifies a private
structured excerpt of the preferred user-message example; the accompanying
attachment contains the earlier bot outputs and is archived privately too.

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

With the current source set, synchronization produces 166 chunks (117 existing
base chunks and 49 managed entries). An embedding refresh failure leaves the
saved cache intact; it does not roll the runtime answers back to old wording.
The app does not generate a full embedding cache during startup. To validate
the intended corpus without API calls, use:

```sh
python interview_bot/generate_embeddings.py --sync-manager-content --dry-run
```

The Q14 replacement differs from the saved semantic index by one managed entry;
the other 165 vectors can be reused. The latest Q14 text is available through the
runtime source overlay and lexical/exact-question retrieval after deployment.
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
