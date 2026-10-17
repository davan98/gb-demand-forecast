# Serving the model and answering project questions

This extension makes the existing historical ridge model usable through a local HTTP API and adds a source-citing assistant over the project documentation. The forecasting experiment, its split boundaries and its reported accuracy remain the same. These are portfolio demonstrations, not a public production service or a live energy forecasting feed.

## Train and export

From the repository root:

```bash
python -m pip install -r requirements.txt
python src/pipeline.py
python -m unittest discover -s tests -v
```

Training writes `models/ridge-v1.npz` and `models/metadata-v1.json`. Keep both together. Metadata records the ordered features, training cutoff and an integrity checksum. The loader uses NumPy arrays without pickle. Model files are generated locally and excluded from Git; train them from the documented source snapshot before serving.

## Serve locally

```bash
python src/api.py --host 127.0.0.1 --port 8000 --model-dir models
```

GET `/health` reports model readiness and the feature contract. POST `/predict` accepts `{"rows": [{"feature_name": 1.0}]}`: each object must contain every exact feature listed by `/health`, not the placeholder shown here. Use the same feature engineering as `src/pipeline.py`; arbitrary feature values produce meaningless results even if schema validation passes. This interface serves the fixed historical model; it does not obtain live data or construct future feature rows automatically.

Example requests from another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" -d '{"question":"Why are the demand lags at least 28 days old?","top_k":3}'
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"What are the forecast uncertainty limitations?","top_k":3}'
```

PowerShell users can use `Invoke-RestMethod` with a JSON body instead of shell-specific curl quoting.

## Retrieval and generation

The retriever ranks chunks from an explicit allowlist of the README, interview notes and evaluation/audit JSON files. Retrieved chunks carry source paths and citation IDs. It uses lexical retrieval; no embeddings, vector database or external data index is claimed.

Without a configured local model, `/ask` returns clearly labelled source excerpts in `retrieval_only` mode. This is useful for inspecting retrieval, but it is not evidence of an LLM-generated answer. To enable the generation path, run Ollama on the same host, download a model suitable for your machine, set `OLLAMA_MODEL` to its exact installed name and restart the API. The default endpoint is local loopback at `http://127.0.0.1:11434/api/generate`. No model download or paid service is invoked automatically.

POSIX shell example (replace the model name with an installed model):

```bash
export OLLAMA_MODEL=your-installed-model
python src/api.py
```

PowerShell:

```powershell
$env:OLLAMA_MODEL = "your-installed-model"
python src/api.py
```

Generation prompts separate the question from retrieved source text and require citations. The application rejects missing or unknown citation IDs and reports backend failures explicitly. Citation checks do not prove the answer is supported by the cited passage; human evaluation remains necessary. Mocked backend tests verify control flow, not real-model answer quality. Inspect `docs/rag_evaluation.json` for retrieval-only evaluation scope and results.

## Container package

After training on the host:

```bash
docker build -t gb-demand-api .
docker run --rm -p 127.0.0.1:8000:8000 --mount type=bind,source="$PWD/models",target=/app/models,readonly gb-demand-api
```

The image runs as a non-root user and loads model files from a read-only mount. The host publication is loopback-only. On Windows, supply the absolute Windows path to the `models` folder in the bind mount.

The simple HTTP server is for local demonstrations. It has no public-service authentication, TLS termination or production concurrency controls. A public deployment needs an appropriate serving stack, access controls, monitoring and load tests. Docker execution requires a local Docker daemon; adding a Dockerfile alone does not prove a built image or cloud deployment. The optional Ollama backend is intentionally loopback-only; use the host process for the full generation demonstration rather than exposing an arbitrary remote model URL.

## Interview evidence

Distinguish the completed code from runtime evidence:

- Safe model serialisation and a tested prediction endpoint demonstrate serving engineering.
- A successful local request demonstrates local serving; it is not cloud deployment.
- Retriever evaluation measures source ranking, not LLM factual accuracy.
- A real local model session is still required before describing generated answers as manually evaluated.
- Keep RAG and model serving under personal projects on the CV unless you also have separate employer experience.

References: [Python HTTP server limitations](https://docs.python.org/3/library/http.server.html), [Ollama generation API](https://docs.ollama.com/api/generate).
