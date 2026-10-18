# InterviewBot

InterviewBot retrieves one saved experience at a time and produces either a STAR
story or a direct interview answer, with optional voice input and answer audio.

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

## Content and validation

Job-focus changes do not alter `chunks_cleaned.json` or regenerate saved story
embeddings. `interview_retrieval.py` holds parsing, topic matching, ranking, and
candidate-reset logic separately from the Streamlit UI.

After editing `manager_content.json`, refresh the corpus and index together with
`OPENAI_API_KEY` configured in the environment:

```sh
python interview_bot/generate_embeddings.py --sync-manager-content
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
