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
renders evidence, practice, and session controls. This preserves the existing
retrieval and embedding-cache workflow without duplicating engineering logic.

After editing `manager_content.json`, refresh the corpus and index together with
`OPENAI_API_KEY` configured in the environment:

```sh
python interview_bot/generate_embeddings.py --sync-manager-content
```

The October 19, 2026 content update adds 20 transmission-planning interview
questions to `manager_content.json` (`planning-interview-01` through
`planning-interview-20`): 13 proposed technical/management approaches and seven
experience-based answers. The original attachment is preserved byte-for-byte in
`sources/transmission_planning_interview_2026-10-19.txt`; each imported entry records
its question number and the source SHA-256. The complete supplied answer remains
available alongside the STAR organization of experience-based entries. Existing
stories are retained.

Imported references, their dates, and preparation notes remain in the retrieved
context. They are user-provided preparation material, not independently verified
ERCOT/NERC requirements or verified personal outcomes. Generation and coaching
instructions retain hypothetical wording, potential versus realized savings,
effective-date caveats, and details that require confirmation. These prompt rules
do not guarantee model compliance. In particular, the EMS-upgrade example remains
an operations example and is excluded from Transmission planning story searches;
use All roles or Project management to retrieve it.

Editing the source alone does **not** activate new answers: the app reads
`chunks_cleaned.json` and its matching saved embedding cache. Run the refresh above
before deploying the content update, then restart/redeploy the app or use
**Reload saved embeddings**. With the current source set, synchronization produces
166 chunks (117 existing base chunks and 49 managed entries). Without an API key
or if generation fails, the existing searchable corpus remains intact and the
new source content remains pending. To validate the intended corpus without API
calls, use:

```sh
python interview_bot/generate_embeddings.py --sync-manager-content --dry-run
```

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
