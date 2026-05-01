# Local vLLM Setup for PageIndex with Qwen2.5-7B-Instruct-AWQ

This guide documents how to run a local OpenAI-compatible vLLM server with an NVIDIA GPU and use it with the PageIndex project.

The final working model used was:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

This model was used instead of the full precision Qwen model because the full `Qwen/Qwen2.5-7B-Instruct` model was too large for the available GPU VRAM when served with vLLM.

---

## 1. Check Docker and NVIDIA GPU Access

Check that Docker is working:

```bash
docker --version
```

Check that the NVIDIA GPU is visible from the host:

```bash
nvidia-smi
```

You should see your GPU listed. In this setup, the GPU was an NVIDIA RTX 4090 Laptop GPU with around 16 GB of VRAM.

---

## 2. Stop Any Old vLLM Container

If a previous vLLM container is already running, stop and remove it first.

List running containers:

```bash
docker ps
```

Stop the old container:

```bash
docker stop vllm-qwen
```

Remove the old container:

```bash
docker rm vllm-qwen
```

If the old container has a different name, replace `vllm-qwen` with the actual container name or container ID.

---

## 3. Start vLLM with Qwen2.5-7B-Instruct-AWQ

Run the following command:

```bash
docker run -d \
    --name vllm-qwen \
    --gpus all \
    --ipc=host \
    -p 8010:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

Explanation of the important options:

```text
--gpus all
```

Allows the Docker container to access the NVIDIA GPU.

```text
-p 8010:8000
```

Maps vLLM's internal API port `8000` to host port `8010`.

```text
-v ~/.cache/huggingface:/root/.cache/huggingface
```

Caches downloaded Hugging Face model weights so they do not need to be downloaded every time.

```text
--model Qwen/Qwen2.5-7B-Instruct-AWQ
```

Loads the quantized Qwen model.

```text
--quantization awq
```

Tells vLLM to load the AWQ quantized version of the model.

```text
--max-model-len 4096
```

Sets the model context length to 4096 tokens.

---

## 4. Watch vLLM Logs

After starting the container, watch the logs:

```bash
docker logs -f vllm-qwen
```

A successful startup should eventually show something like:

```text
Starting vLLM server on http://0.0.0.0:8000
Application startup complete.
```

To exit the log view without stopping the container, press:

```text
Ctrl + C
```

---

## 5. Verify the Model Endpoint

Check that the vLLM OpenAI-compatible endpoint is running:

```bash
curl http://localhost:8010/v1/models
```

Expected output should include:

```json
{
  "id": "Qwen/Qwen2.5-7B-Instruct-AWQ"
}
```

---

## 6. Verify GPU Usage

Run:

```bash
nvidia-smi
```

You should see a Python process using GPU memory. In the working setup, vLLM used roughly 14 GB of GPU memory:

```text
/python3.12
```

This confirms that vLLM is using the NVIDIA GPU.

---

## 7. Test a Simple vLLM Completion

Run this test request:

```bash
curl http://localhost:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": "Return only valid JSON with one key named ok and value true."
            }
        ],
        "temperature": 0
    }'
```

The response should contain something like:

```json
{"ok": true}
```

---

## 8. Configure PageIndex to Use Local vLLM

Set these environment variables before running PageIndex:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8010/v1
```

The `OPENAI_API_KEY` value can be `dummy` because the model is local.

The important part is:

```bash
OPENAI_BASE_URL=http://localhost:8010/v1
```

This makes LiteLLM/OpenAI-compatible calls go to the local vLLM server instead of OpenAI.

---

## 9. Run PageIndex with the Local vLLM Model

Run PageIndex with the served model ID:

```bash
python run_pageindex.py \
    --pdf_path docs/DAOD6000-0.pdf \
    --model openai/Qwen/Qwen2.5-7B-Instruct-AWQ \
    --toc-check-pages 0 \
    --if-add-node-id yes \
    --if-add-node-summary no
```

Important:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

is the model ID served by vLLM.

For LiteLLM, the PageIndex command uses:

```text
openai/Qwen/Qwen2.5-7B-Instruct-AWQ
```

because vLLM exposes an OpenAI-compatible API.

---

## 10. Why AWQ Was Used Instead of the Full Qwen Model

The full model:

```text
Qwen/Qwen2.5-7B-Instruct
```

was attempted first, but vLLM failed because the GPU did not have enough available VRAM for the model plus KV cache.

The error looked like:

```text
ValueError: No available memory for the cache blocks.
```

The AWQ model:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

worked because it is quantized and uses less GPU memory.

---

## 11. Useful Debug Commands

Check running containers:

```bash
docker ps
```

View vLLM logs:

```bash
docker logs -f vllm-qwen
```

Stop vLLM:

```bash
docker stop vllm-qwen
```

Remove vLLM container:

```bash
docker rm vllm-qwen
```

Check vLLM model endpoint:

```bash
curl http://localhost:8010/v1/models
```

Check GPU usage:

```bash
nvidia-smi
```

---

## 12. Notes

The model is local. Even though the PageIndex command uses an `openai/` model prefix, calls are routed to:

```text
http://localhost:8010/v1
```

because of:

```bash
export OPENAI_BASE_URL=http://localhost:8010/v1
```

So the `openai/` prefix is only used by LiteLLM to select the OpenAI-compatible API format. It does not mean the request is sent to OpenAI.
