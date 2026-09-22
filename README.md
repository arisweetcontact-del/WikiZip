# tiny-rag (NOT USEABLE YET! WILL FAIL IF RUN)

A tiny local question-answering setup that runs on 8GB of RAM, no GPU.

**The idea:** your reference material sits zipped. A small
keyword index (a few KB) tells the system which zip is relevant to a
question. nothing is unzipped permanently. When you ask something, only
the matching zip(s) get read into memory, their text is handed to a small
local language model as context, and the model answers from that. The
model itself is small and the "knowledge" stays compressed on disk
until it's actually needed.


## How it stays tiny

 **The model** is a small quantized instruct model ~350MB–1GB on disk,
  similar in RAM while running.
  
  **The knowledge base** stays compressed. `ingest.py` reads each zip
  once, in memory, to build a lightweight word-frequency index (a JSON
  file, typically a few KB even for a large knowledge base, then
  discards the extracted text. Nothing is left unzipped on disk.

**At query time**, only the top 1–2 matching zips are opened (still in
  memory, via Python's `zipfile`, never extracted to disk) and their text
  becomes the model's context for that one answer.


