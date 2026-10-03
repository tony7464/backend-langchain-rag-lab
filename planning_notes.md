# Planning Notes: LangChain RAG API

Complete each section before submitting. Replace every TODO with your own notes.

## 1. System Goal

- User or role: Engineers on our internal platform team who need quick, trustworthy answers while they're handling an incident or on call.
- Business problem: During incidents, engineers need to follow the reliability team's approved runbooks (checkout API errors, rollbacks, status page updates, exposed API keys, rate limits, delayed exports), not generic advice from a model. Wrong or made-up steps can make an outage worse or leak sensitive info, so the answers have to match what's actually been approved.
- Approved knowledge source: The reliability runbook chunks in `data/runbook_chunks.json`, embedded with `embeddinggemma` and stored in a local Chroma collection (`reliability_runbooks`). The model should answer only from what gets retrieved from there.
- Endpoint route: `POST /api/ask` with a JSON body like `{"question": "..."}`.

## 2. Manual RAG Workflow Map

Name the manual RAG steps this LangChain version is organizing.

- Receive question: The Flask route reads the JSON body, and `validate_question_payload()` checks it's an object with a non-blank string `question` between 3 and 500 characters, then strips it.
- Retrieve context: `retrieve_context()` runs `similarity_search_with_score()` on Chroma and gets back the top-k runbook chunks along with their distances.
- Build prompt: `format_context()` turns the chunks into labeled context blocks (source ID, title, section, chunk ID, distance, text), and the `ChatPromptTemplate` fills in `{context}` and `{question}`.
- Call model: The chain sends the filled prompt to `ChatOllama` running `llama3.2` locally, with temperature 0.
- Parse or format output: `StrOutputParser` turns the model's message into a plain string, then we strip it and reject an empty answer.
- Return sources: `format_sources()` returns metadata and distance for each retrieved chunk (not the full text) so people can see which runbooks backed the answer.
- Verify quality: Automated tests with fake vector stores and chains check validation, response shape, prompt variables, fallback, and error handling. Then I spot-check real answers against the runbooks they cite.

## 3. LangChain Component Mapping

Map the manual workflow to LangChain-supported pieces.

- Prompt string maps to: `ChatPromptTemplate` (a system message with the grounding rules, plus a human message with `{context}` and `{question}`).
- Chroma search logic maps to: the `langchain_chroma.Chroma` vector store with `OllamaEmbeddings`, queried through `similarity_search_with_score`.
- Direct model call maps to: `ChatOllama(model="llama3.2", temperature=0)`.
- Function-to-function workflow maps to: the runnable sequence `prompt_template | llm | StrOutputParser()`, called with `chain.invoke({"context": ..., "question": ...})`.
- Manual output cleanup maps to: `StrOutputParser`, plus our own strip and empty-answer check.
- Manual testing maps to: pytest with fakes (`FakeVectorStore`, `FakeChain`) passed into `answer_question()`, so tests don't need Ollama or a seeded Chroma database.

## 4. Response Contract

List the fields a successful response should include.

- `answer`: the model's answer as a string, grounded in the retrieved runbook context (or the fixed fallback message).
- `sources`: a list of `{source_id, title, category, section, chunk_id, distance}` for each retrieved chunk, without the full document text.
- `langchain`: debug metadata, meaning the chain expression, component names, retrieved count and chunk IDs, `top_k`, context length, `fallback` flag, and score type.

Errors use a separate shape, `{error, message}`: 400 for validation problems and 502 when the LangChain service fails.

## 5. Fallback Behavior

Explain what the system should do when no approved context is available.

If retrieval comes back empty, or none of the chunks have real text, the service skips the model completely. It returns the fixed answer "I do not have enough approved runbook context to answer that reliably.", an empty `sources` list, and `langchain.fallback: true`. The response shape stays the same, so clients don't have to handle it differently. Not calling the model at all is the safest way to make sure it can't invent runbook steps. On top of that, the prompt tells the model to say it lacks approved context if the retrieved chunks don't actually answer the question.

## 6. Verification Plan

List at least four checks you should run before trusting the refactor.

- Run `pytest` and make sure the whole suite passes (validation, formatter, prompt, vector store, service, route, and static structure tests).
- Send bad requests (`{}`, a non-string question, a blank question, `"hi"`, 501+ characters, a non-JSON body) and confirm each one returns a 400 with the right `error` code.
- Ask an off-topic question (for example, the office lunch policy) and confirm we get the fallback message or an "I don't have enough approved context" style answer, not invented policy.
- With Ollama running and Chroma seeded, ask a real question like checkout errors after a release. Confirm the answer matches the cited chunks (`chunk-rel-101-b`, etc.) and that the distances look reasonable.
- Stop Ollama, or point at an empty Chroma path, and confirm the endpoint returns a structured 502 instead of crashing.

## 7. Reflection

Explain what LangChain simplified and what may be harder to inspect.

LangChain made the workflow easier to read and swap out. The prompt, model, and parser are separate pieces joined with `|`, and the vector store and chain can be passed in, which makes testing with fakes really easy. What's harder to see is what actually gets sent to the model. The chain hides the final rendered prompt, and errors from Chroma or Ollama come up through a few layers. That's why I still format the context myself, return sources with distances, and include the `langchain` debug block. Retrieval quality, the wording of the prompt, and the fallback rules still need careful debugging. LangChain organizes those steps, but it doesn't make them correct. Since the collection, prompt, and model are all configured separately, this setup could support a different assistant just by pointing it at another document collection and adjusting the system prompt.
