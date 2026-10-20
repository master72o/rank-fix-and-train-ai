"""
Seed script populating the Golden RLHF / DPO Evaluation Dataset (40 realistic pairs).
Covers Coding, Reasoning, Math, Instruction-Following, Science, and System Design.
"""

import asyncio
import json
import uuid
import sys
from pathlib import Path

# Add backend directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.database import SessionLocal, init_db
from app.models.domain import Dataset, PromptSample, SampleStatus, TaskType
from app.services.active_learning import PriorityWeights, priority_score, semantic_divergence
from app.services.inference_engine import get_embedder
from app.services.llm_judge import DEFAULT_RUBRIC

GOLDEN_PROMPTS = [
    # --- Coding (1-10) ---
    {
        "category": "Coding",
        "system_prompt": "You are an expert Python software engineer. Provide concise, clean, PEP 8 compliant code.",
        "user_prompt": "Write a Python function to safely flatten a nested list of arbitrary depth without using recursion.",
        "candidate_a": "def flatten_list(nested):\n    stack = [list(nested)]\n    result = []\n    while stack:\n        current = stack.pop()\n        while current:\n            item = current.pop(0)\n            if isinstance(item, (list, tuple)):\n                stack.append(current)\n                current = list(item)\n            else:\n                result.append(item)\n    return result",
        "candidate_b": "def flatten_list(nested):\n    result = []\n    stack = [nested]\n    while stack:\n        curr = stack.pop()\n        if isinstance(curr, (list, tuple)):\n            stack.extend(reversed(curr))\n        else:\n            result.append(curr)\n    return result",
        "entropy_a": 0.45,
        "entropy_b": 0.38,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B uses stack.extend(reversed(curr)) which is O(1) per element, whereas Candidate A uses current.pop(0) which is O(N) leading to quadratic overhead."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a database architect specializing in PostgreSQL.",
        "user_prompt": "Explain how to implement an idempotent upsert in PostgreSQL with ON CONFLICT and return whether an insert or update occurred.",
        "candidate_a": "You can use INSERT INTO users (id, name, updated_at) VALUES (1, 'Alice', NOW()) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, updated_at = NOW() RETURNING *, (xmax = 0) AS inserted;",
        "candidate_b": "In PostgreSQL, upsert is done with INSERT INTO ... ON DUPLICATE KEY UPDATE. To check if it was inserted or updated, query the row count in your application driver.",
        "entropy_a": 0.32,
        "entropy_b": 0.95,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the exact Postgres syntax with the idiomatic (xmax = 0) AS inserted check, while Candidate B confuses Postgres with MySQL syntax."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a TypeScript and React performance specialist.",
        "user_prompt": "When should React developers use `useCallback` vs `useMemo`? Give a concrete rule of thumb.",
        "candidate_a": "Always wrap every function in useCallback and every object in useMemo to prevent re-renders. Modern React engines optimize memoized values automatically.",
        "candidate_b": "Use useCallback to preserve function reference identity when passing callbacks to memoized children (React.memo) or hook dependency arrays. Use useMemo to cache expensive computations or referential objects. Don't overuse them prematurely as dependency checking adds memory and execution overhead.",
        "entropy_a": 0.88,
        "entropy_b": 0.25,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B gives accurate advice. Candidate A advocates premature memoization which hurts performance and memory."
    },
    {
        "category": "Coding",
        "system_prompt": "You are an API design architect.",
        "user_prompt": "Design a resilient retry policy with exponential backoff and jitter in Python.",
        "candidate_a": "import time, random\n\ndef retry_with_backoff(func, max_attempts=5, base_delay=1.0, max_delay=30.0):\n    for attempt in range(max_attempts):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_attempts - 1:\n                raise e\n            sleep = min(max_delay, base_delay * (2 ** attempt))\n            jitter = random.uniform(0, sleep * 0.1)\n            time.sleep(sleep + jitter)",
        "candidate_b": "def retry(func):\n    while True:\n        try:\n            return func()\n        except:\n            time.sleep(2)",
        "entropy_a": 0.35,
        "entropy_b": 1.10,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A implements bounded exponential backoff, jitter, and max attempts. Candidate B implements an unbounded loop catching all exceptions."
    },
    {
        "category": "Coding",
        "system_prompt": "You are an asynchronous programming specialist.",
        "user_prompt": "What is the danger of calling blocking synchronous I/O inside an `async def` function in Python, and how do you fix it?",
        "candidate_a": "It throws an AsynchronousCallError at runtime because async functions require aiohttp or asyncio equivalents.",
        "candidate_b": "Calling blocking I/O freezes the single-threaded asyncio event loop, blocking all concurrent coroutines. Fix it by using an async-native library or offloading the call using `await asyncio.to_thread(sync_func, *args)`.",
        "entropy_a": 0.75,
        "entropy_b": 0.28,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B accurately explains event loop starvation and recommends asyncio.to_thread."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a Rust programming mentor.",
        "user_prompt": "Explain the difference between `String` and `&str` in Rust to a beginner.",
        "candidate_a": "`String` is an owned, heap-allocated, growable UTF-8 string buffer. `&str` is an immutable borrowed string slice (pointer + length) pointing to a view of UTF-8 data, which can live on the heap, stack, or in static binary memory.",
        "candidate_b": "`String` is used for single characters, while `&str` is used for full sentences and arrays in Rust.",
        "entropy_a": 0.22,
        "entropy_b": 1.25,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the correct memory ownership explanation. Candidate B is factually wrong."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a software security engineer.",
        "user_prompt": "How do you securely store user passwords in a relational database?",
        "candidate_a": "Hash the password using SHA-256 with a secret application salt string stored in environment variables, then save the hexadecimal string.",
        "candidate_b": "Use an adaptive slow key-derivation function like Argon2id or bcrypt with an automatic per-user cryptographic salt and tuned work factor (cost). Never use fast cryptographic hashes like SHA-256.",
        "entropy_a": 0.85,
        "entropy_b": 0.20,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B adheres to OWASP guidelines recommending Argon2id/bcrypt. Fast hashes like SHA-256 are vulnerable to GPU brute-forcing."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a Docker and containerization expert.",
        "user_prompt": "Why should you use multi-stage builds when packaging a Go or Rust application into a Docker container?",
        "candidate_a": "Multi-stage builds allow compiling the binary in a build stage with all toolchains, then copying only the minimal compiled binary into a lightweight runtime image (e.g., scratch or alpine). This drastically reduces image size and eliminates compiler security attack surface.",
        "candidate_b": "Multi-stage builds make Docker containers run twice as fast because they distribute layers across multi-core processors.",
        "entropy_a": 0.30,
        "entropy_b": 1.15,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly explains image shrinkage and attack surface reduction. Candidate B hallucinates performance claims."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a Git workflow specialist.",
        "user_prompt": "What does `git cherry-pick` do and when is it appropriate to use?",
        "candidate_a": "`git cherry-pick <commit>` applies the changes introduced by a specific existing commit from another branch onto your current HEAD as a new commit. It is best used for backporting hotfixes or extracting isolated commits without merging an entire feature branch.",
        "candidate_b": "`git cherry-pick` picks the top 3 best commits from git history and cleans up any uncommitted files in your staging area.",
        "entropy_a": 0.28,
        "entropy_b": 1.20,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A accurately defines cherry-pick and its primary use case."
    },
    {
        "category": "Coding",
        "system_prompt": "You are a distributed systems architect.",
        "user_prompt": "Compare pessimistic locking vs optimistic concurrency control (OCC) for database transactions.",
        "candidate_a": "Pessimistic locking takes exclusive locks (e.g., SELECT FOR UPDATE) at read time, preventing conflicts at the cost of concurrency and deadlock risk. OCC allows concurrent reads and writes, checking a version column at commit time; if modified, the transaction retries. OCC excels under low-to-moderate contention.",
        "candidate_b": "Pessimistic locking is for databases with negative indexes, while optimistic locking is used whenever a database uses SSD storage.",
        "entropy_a": 0.24,
        "entropy_b": 1.30,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the standard database concurrency distinction. Candidate B is nonsense."
    },

    # --- Reasoning & Math (11-20) ---
    {
        "category": "Reasoning",
        "system_prompt": "You are a mathematician and probability educator.",
        "user_prompt": "In the Monty Hall problem with 3 doors, why does switching doors give a 2/3 probability of winning?",
        "candidate_a": "Your initial door choice had a 1/3 chance of having the car, meaning the remaining two doors together had a 2/3 chance. Because the host always reveals a goat from those remaining doors, switching captures the entire 2/3 probability.",
        "candidate_b": "Once one door is opened, only two closed doors remain. Since there is 1 car and 2 doors, the probability is always 50/50 whether you switch or stay.",
        "entropy_a": 0.31,
        "entropy_b": 0.85,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly solves the Monty Hall problem. Candidate B commits the classic 50/50 fallacy."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are a formal logic instructor.",
        "user_prompt": "Explain the difference between sound arguments and valid arguments in deductive logic.",
        "candidate_a": "A valid argument is one where if the premises were true, the conclusion must be true (valid structure). A sound argument is a valid argument whose premises are actually true in reality.",
        "candidate_b": "A valid argument has scientific evidence, while a sound argument is an argument that sounds persuasive to an audience.",
        "entropy_a": 0.20,
        "entropy_b": 1.10,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A uses formal philosophical definitions. Candidate B uses colloquial misinterpretations."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are a statistics tutor.",
        "user_prompt": "What is Simpson's Paradox? Provide a simple intuitive example.",
        "candidate_a": "Simpson's Paradox occurs when a trend appears in several different groups of data but disappears or reverses when the groups are combined. For example, a treatment may look superior in both mild and severe patient subgroups, but inferior overall if sicker patients were disproportionately assigned to it.",
        "candidate_b": "Simpson's Paradox is when adding more parameters to a neural network causes the training loss to increase instead of decrease.",
        "entropy_a": 0.33,
        "entropy_b": 0.98,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A accurately describes Simpson's Paradox and the confounding variable mechanism."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are a mathematical problem solver.",
        "user_prompt": "If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost?",
        "candidate_a": "The ball costs $0.10, because $1.10 minus $1.00 equals $0.10.",
        "candidate_b": "Let x be the cost of the ball. The bat is x + 1.00. Together: x + (x + 1.00) = 1.10 => 2x = 0.10 => x = $0.05. The ball costs 5 cents.",
        "entropy_a": 0.70,
        "entropy_b": 0.25,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B solves the system of equations correctly (5 cents). Candidate A gives the intuitive but incorrect answer."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are a discrete math educator.",
        "user_prompt": "How many handshakes occur in a room with 12 people if everyone shakes hands with everyone else exactly once?",
        "candidate_a": "It is 12 choose 2: C(12, 2) = (12 * 11) / 2 = 66 handshakes.",
        "candidate_b": "Each of the 12 people shakes hands with 11 people, so 12 * 11 = 132 handshakes.",
        "entropy_a": 0.18,
        "entropy_b": 0.72,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A accounts for symmetry (handshakes are mutual). Candidate B double counts."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are a data science and machine learning specialist.",
        "user_prompt": "Explain the bias-variance tradeoff in supervised learning.",
        "candidate_a": "High bias leads to underfitting (model is too simple to capture data patterns). High variance leads to overfitting (model models training noise). As model complexity increases, bias decreases while variance increases; the goal is finding the sweet spot minimizing total test error.",
        "candidate_b": "Bias is when the dataset is unfair to certain groups, and variance is how fast the GPU trains the model parameters.",
        "entropy_a": 0.27,
        "entropy_b": 1.05,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the correct statistical definition of bias and variance."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are a physics educator.",
        "user_prompt": "Why doesn't the Moon crash into Earth if Earth's gravity is pulling on it constantly?",
        "candidate_a": "The Moon has sufficient tangential orbital velocity. It is constantly falling toward Earth, but Earth's surface curves away at the same rate, keeping the Moon in stable continuous free fall orbit.",
        "candidate_b": "The Moon has zero gravity acting on it because gravity stops once you leave Earth's atmosphere.",
        "entropy_a": 0.29,
        "entropy_b": 1.20,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A gives the Newtonian orbital mechanics explanation. Candidate B is completely incorrect."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are an operations research tutor.",
        "user_prompt": "What is Little's Law in queuing theory, and where is it applied in software engineering?",
        "candidate_a": "Little's Law states L = lambda * W, meaning average items in a system equals arrival rate times average wait time. In software, it sizes concurrency limits and thread pools: Concurrent Requests = Throughput (req/sec) * Latency (sec).",
        "candidate_b": "Little's law states that code will always expand to fill the memory allocated to it.",
        "entropy_a": 0.30,
        "entropy_b": 1.15,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly states L = lambda * W and links it to throughput and latency capacity sizing."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are an algorithm instructor.",
        "user_prompt": "What is the time and space complexity of merge sort, and is it stable?",
        "candidate_a": "Time complexity is O(N log N) in best, average, and worst cases. Space complexity is O(N) auxiliary space. Merge sort is stable because equal elements preserve their relative initial order when merging.",
        "candidate_b": "Time complexity is O(N^2) in worst case and O(1) space. It is not stable.",
        "entropy_a": 0.19,
        "entropy_b": 0.90,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A gives the exact complexity bounds and stability guarantee."
    },
    {
        "category": "Reasoning",
        "system_prompt": "You are an economics tutor.",
        "user_prompt": "What is opportunity cost? Give a workplace career example.",
        "candidate_a": "Opportunity cost is the value of the next-best alternative forgone when making a decision. For instance, choosing to spend 2 years getting a full-time MBA incurs not just tuition cost, but the forgone 2 years of salary and career progression.",
        "candidate_b": "Opportunity cost is the discount you receive when buying equipment in bulk for your company.",
        "entropy_a": 0.24,
        "entropy_b": 1.25,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the correct economic definition and a solid real-world example."
    },

    # --- Instruction-Following & System Design (21-30) ---
    {
        "category": "InstructionFollowing",
        "system_prompt": "You must strictly follow formatting constraints. Respond ONLY in valid JSON with no markdown wrapping.",
        "user_prompt": "Return a JSON object with keys 'service', 'port', and 'healthy' for an auth microservice.",
        "candidate_a": "```json\n{\n  \"service\": \"auth\",\n  \"port\": 8080,\n  \"healthy\": true\n}\n```",
        "candidate_b": "{\"service\": \"auth\", \"port\": 8080, \"healthy\": true}",
        "entropy_a": 0.55,
        "entropy_b": 0.22,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B followed the negative constraint 'no markdown wrapping'. Candidate A wrapped the output in markdown fences."
    },
    {
        "category": "InstructionFollowing",
        "system_prompt": "Summarize in EXACTLY three bullet points. No intro or outro.",
        "user_prompt": "Summarize the primary benefits of using an API Gateway in microservices.",
        "candidate_a": "Sure! Here are three benefits:\n- Centralizes cross-cutting concerns like auth and rate limiting\n- Reduces client chattiness through request aggregation\n- Decouples client protocols from internal microservice implementations",
        "candidate_b": "- Centralizes cross-cutting concerns like authentication, SSL termination, and rate limiting.\n- Shields internal microservice topology and decouples client communication protocols.\n- Simplifies client integration by providing unified routing and response aggregation.",
        "entropy_a": 0.60,
        "entropy_b": 0.20,
        "gold_preference": "CANDIDATE_B",
        "rationale": "Candidate B adhered to 'No intro or outro' and produced exactly three bullet points. Candidate A included conversational filler."
    },
    {
        "category": "SystemDesign",
        "system_prompt": "You are a principal systems architect.",
        "user_prompt": "How would you design a distributed unique ID generator capable of generating 100,000 IDs per second without single points of failure?",
        "candidate_a": "Use a Snowflake-style 64-bit ID layout: 1 bit unused, 41 bits timestamp (millisecond precision, ~69 years), 10 bits machine/worker ID (coordinated via ZooKeeper/etcd), and 12 bits sequence number (up to 4096 IDs per ms per machine). Independent worker nodes generate IDs locally with zero inter-node coordination.",
        "candidate_b": "Use an auto-incrementing integer column in a single MySQL primary database instance with a synchronized lock.",
        "entropy_a": 0.28,
        "entropy_b": 0.85,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the industry-standard Twitter Snowflake architecture with bit budget. Candidate B introduces a severe bottleneck and single point of failure."
    },
    {
        "category": "SystemDesign",
        "system_prompt": "You are a backend caching architect.",
        "user_prompt": "What is the difference between Cache-Aside and Write-Through caching patterns?",
        "candidate_a": "In Cache-Aside, the application directly queries the cache; on a miss, it reads from the DB and writes back to cache. In Write-Through, the application treats the cache as the main data store; the cache service itself writes data to the DB synchronously before acknowledging the write.",
        "candidate_b": "Cache-Aside is for Redis, while Write-Through is only supported by Memcached.",
        "entropy_a": 0.22,
        "entropy_b": 1.10,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the exact algorithmic definitions."
    },
    {
        "category": "SystemDesign",
        "system_prompt": "You are a message queue specialist.",
        "user_prompt": "What is the difference between at-least-once delivery and exactly-once processing in Kafka?",
        "candidate_a": "At-least-once guarantees messages are never lost but may be redelivered upon failures or retries. Exactly-once processing combines idempotent producers and transactional commits across read-process-write loops so downstream consumer state reflects each message exactly once.",
        "candidate_b": "At-least-once is for text messages, while exactly-once is for binary files.",
        "entropy_a": 0.25,
        "entropy_b": 1.20,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly explains idempotency and transactional boundaries."
    },
    {
        "category": "InstructionFollowing",
        "system_prompt": "Explain the concept in a single sentence suitable for an executive.",
        "user_prompt": "What is Retrieval-Augmented Generation (RAG)?",
        "candidate_a": "Retrieval-Augmented Generation (RAG) is an AI architecture that searches an organization's proprietary documents to inject factual context into language model queries, ensuring accurate answers without costly model retraining.",
        "candidate_b": "RAG stands for Retrieval Augmented Generation. It was created in 2020 by researchers at Facebook AI. It works by using vector databases like Pinecone or Chroma. Then it does cosine similarity.",
        "entropy_a": 0.30,
        "entropy_b": 0.80,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A adheres to the single sentence and executive tone constraints. Candidate B uses multiple sentences and technical jargon."
    },
    {
        "category": "SystemDesign",
        "system_prompt": "You are a database reliability engineer.",
        "user_prompt": "Why would an engineering team choose read replicas, and what consistency challenges do they introduce?",
        "candidate_a": "Read replicas horizontally scale read throughput by offloading queries from the primary. However, asynchronous replication causes replication lag, leading to eventual consistency issues where a client writes to the primary but reads stale data from a replica (read-your-own-writes violation).",
        "candidate_b": "Read replicas prevent all database crashes because the secondary takes over instantly with zero data loss without any election mechanism.",
        "entropy_a": 0.26,
        "entropy_b": 0.95,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A accurately identifies replication lag and read-your-own-writes consistency hazards."
    },
    {
        "category": "InstructionFollowing",
        "system_prompt": "Provide your answer as a numbered step-by-step checklist. Maximum 4 steps.",
        "user_prompt": "How to rotate an AWS IAM user access key safely with zero downtime.",
        "candidate_a": "1. Create a second active access key in the AWS console or CLI.\n2. Deploy the new key to your application environments and verify successful operations.\n3. Deactivate the old key and monitor logs for 24 hours to confirm no services broke.\n4. Permanently delete the old key once no errors are observed.",
        "candidate_b": "First delete the old key so nobody can use it. Then create a new key and update your servers as fast as possible so downtime is minimal.",
        "entropy_a": 0.22,
        "entropy_b": 0.90,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A gives the zero-downtime standard operating procedure in 4 clean steps. Candidate B creates guaranteed downtime."
    },
    {
        "category": "SystemDesign",
        "system_prompt": "You are an infrastructure security architect.",
        "user_prompt": "What is the principle of least privilege, and how is it implemented in cloud environments?",
        "candidate_a": "The principle of least privilege dictates that identities (users, services, roles) are granted only the minimum permissions necessary to complete their specific task. In cloud environments, it is implemented via granular IAM policies, time-bound temporary credentials (e.g. AWS STS AssumeRole), and regular permission auditing.",
        "candidate_b": "Least privilege means granting administrator access to all team leads so they don't get blocked during deployments.",
        "entropy_a": 0.20,
        "entropy_b": 1.15,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the correct cybersecurity definition and implementation mechanisms."
    },
    {
        "category": "SystemDesign",
        "system_prompt": "You are a network engineer.",
        "user_prompt": "Explain the difference between TCP and UDP and name one protocol built on each.",
        "candidate_a": "TCP is connection-oriented, reliable, and ordered, using a 3-way handshake, sequence numbers, and retransmissions (e.g., HTTP/1.1 and HTTPS). UDP is connectionless and lightweight, prioritizing low latency over guaranteed delivery (e.g., DNS queries and WebRTC).",
        "candidate_b": "TCP is for wired connections, and UDP is for wireless WiFi connections.",
        "entropy_a": 0.18,
        "entropy_b": 1.20,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly outlines the transport layer differences and protocols. Candidate B is factually wrong."
    },

    # --- Science & ML Concepts (31-40) ---
    {
        "category": "Science",
        "system_prompt": "You are a machine learning scientist.",
        "user_prompt": "What is the key difference between RLHF (PPO) and DPO (Direct Preference Optimization)?",
        "candidate_a": "RLHF with PPO requires training an explicit reward model and running online actor-critic reinforcement learning with an active sampling loop. DPO mathematically reformulates the reward objective directly into a closed-form loss over paired preferences, eliminating the need to train a separate reward model or sample during training.",
        "candidate_b": "RLHF is used for vision models, while DPO is only used for audio processing.",
        "entropy_a": 0.25,
        "entropy_b": 1.10,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A accurately captures the mathematical innovation of Rafailov et al. (2023)."
    },
    {
        "category": "Science",
        "system_prompt": "You are a molecular biology tutor.",
        "user_prompt": "Describe the central dogma of molecular biology.",
        "candidate_a": "The central dogma describes the flow of genetic information: DNA is transcribed into messenger RNA (mRNA) in the nucleus, which is then translated by ribosomes into functional proteins in the cytoplasm.",
        "candidate_b": "The central dogma states that all cells come from preexisting rocks through spontaneous generation.",
        "entropy_a": 0.18,
        "entropy_b": 1.40,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly explains transcription and translation. Candidate B is absurdity."
    },
    {
        "category": "Science",
        "system_prompt": "You are a machine learning researcher.",
        "user_prompt": "What is Low-Rank Adaptation (LoRA) and why does it drastically reduce fine-tuning memory?",
        "candidate_a": "LoRA freezes pre-trained weight matrices W and injects trainable rank-decomposition matrices A and B (where W_new = W + B*A with rank r << d). This reduces trainable parameter counts by 99%+, substantially reducing optimizer state memory (e.g. Adam momentum/variance buffers).",
        "candidate_b": "LoRA reduces memory by deleting half of the layers in a neural network during training.",
        "entropy_a": 0.22,
        "entropy_b": 1.05,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the exact linear algebra formulation (B*A) and explains optimizer memory reduction."
    },
    {
        "category": "Science",
        "system_prompt": "You are an astronomy educator.",
        "user_prompt": "What is a light-year, and is it a measure of time or distance?",
        "candidate_a": "A light-year is a unit of astronomical distance, not time. It is the distance that light travels through vacuum in one Julian year—approximately 9.46 trillion kilometers (5.88 trillion miles).",
        "candidate_b": "A light-year is a unit of time that describes how long it takes for a star to burn through its hydrogen fuel.",
        "entropy_a": 0.15,
        "entropy_b": 0.85,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly clarifies that a light-year is a distance unit."
    },
    {
        "category": "Science",
        "system_prompt": "You are a neuroscience educator.",
        "user_prompt": "What is the primary role of myelin in the human nervous system?",
        "candidate_a": "Myelin is a lipid-rich sheath insulating axons that enables rapid saltatory conduction of action potentials between nodes of Ranvier, dramatically increasing nerve impulse transmission velocity.",
        "candidate_b": "Myelin is the digestive enzyme responsible for breaking down proteins in human muscle tissue.",
        "entropy_a": 0.20,
        "entropy_b": 1.15,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the correct physiological description of myelin and saltatory conduction."
    },
    {
        "category": "Science",
        "system_prompt": "You are a natural language processing specialist.",
        "user_prompt": "What does temperature control during LLM sampling?",
        "candidate_a": "Temperature scales the model's output logits prior to applying softmax: P(x_i) = exp(z_i / T) / sum(exp(z_j / T)). A lower temperature (<1.0) sharpens probability peaks making outputs more deterministic, while higher temperature (>1.0) flattens distribution promoting diversity.",
        "candidate_b": "Temperature regulates the cooling fans of the GPU cluster running model inference.",
        "entropy_a": 0.21,
        "entropy_b": 1.30,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A correctly states the mathematical formula and effect on output randomness."
    },
    {
        "category": "Science",
        "system_prompt": "You are an environmental science tutor.",
        "user_prompt": "Explain the greenhouse effect in terms of solar and infrared radiation.",
        "candidate_a": "Shortwave solar radiation passes through Earth's atmosphere and warms the surface. Earth radiates this energy back as longwave infrared radiation. Greenhouse gases (CO2, CH4, H2O) absorb and re-emit this infrared energy, trapping heat within the troposphere.",
        "candidate_b": "The greenhouse effect is caused by glass windows installed in modern skyscrapers reflecting sunlight back into space.",
        "entropy_a": 0.23,
        "entropy_b": 1.25,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A gives the accurate radiative physics explanation."
    },
    {
        "category": "Science",
        "system_prompt": "You are a machine learning engineer.",
        "user_prompt": "What is the purpose of the Attention mechanism in Transformers?",
        "candidate_a": "Attention computes dynamic, pairwise relevance scores between all token positions in a sequence using query-key dot products: Softmax((Q * K^T) / sqrt(d_k)) * V. This allows the model to capture long-range contextual dependencies without being constrained by sequential processing.",
        "candidate_b": "Attention is a heuristic module that flags whenever a user prompt is typing too quickly.",
        "entropy_a": 0.24,
        "entropy_b": 1.15,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A provides the exact scaled dot-product attention equation and motivation."
    },
    {
        "category": "Science",
        "system_prompt": "You are a chemistry tutor.",
        "user_prompt": "What is the difference between an endothermic and an exothermic chemical reaction?",
        "candidate_a": "An exothermic reaction releases energy (enthalpy change Delta H < 0) to its surroundings, usually as heat (e.g. combustion). An endothermic reaction absorbs energy (Delta H > 0) from its surroundings (e.g. photosynthesis or chemical cold packs).",
        "candidate_b": "Exothermic reactions occur only inside living organisms, while endothermic reactions occur only in outer space.",
        "entropy_a": 0.17,
        "entropy_b": 1.20,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A accurately defines Delta H and energy transfer directions."
    },
    {
        "category": "Science",
        "system_prompt": "You are an AI alignment researcher.",
        "user_prompt": "Why is position bias a significant problem when using LLM-as-a-Judge for model evaluations, and how is it mitigated?",
        "candidate_a": "LLMs exhibit a strong systemic preference for the candidate presented in either the first (Position A) or second (Position B) slot, regardless of quality. It is mitigated by swapping positions (evaluating both A/B and B/A) and requiring consistent agreement, or averaging pairwise win scores.",
        "candidate_b": "Position bias refers to the GPS location of the server running the evaluation script.",
        "entropy_a": 0.22,
        "entropy_b": 1.10,
        "gold_preference": "CANDIDATE_A",
        "rationale": "Candidate A defines position bias in LLM-as-a-Judge and explains the bidirectional position-swap mitigation."
    },
]


async def seed_golden_dataset():
    await init_db()
    embedder = get_embedder()
    weights = PriorityWeights(entropy=0.5, divergence=0.5)

    async with SessionLocal() as session:
        # Check if dataset already exists
        existing_ds = (await session.execute(
            select(Dataset).where(Dataset.name == "alignforge-golden-benchmark-v1")
        )).scalar_one_or_none()

        if existing_ds:
            print(f"Dataset '{existing_ds.name}' already exists (id={existing_ds.id}). Skipping seed.")
            return

        dataset = Dataset(
            name="alignforge-golden-benchmark-v1",
            task_type=TaskType.DPO_PREFERENCE,
            rubric_definition=DEFAULT_RUBRIC,
        )
        session.add(dataset)
        await session.flush()

        print(f"Creating 40 Golden Benchmark Prompt Samples for dataset '{dataset.name}'...")

        for idx, item in enumerate(GOLDEN_PROMPTS, start=1):
            prompt_text = item["user_prompt"]
            prompt_vec = embedder.embed([prompt_text])[0]
            emb_a, emb_b = embedder.embed([item["candidate_a"], item["candidate_b"]])
            div = round(semantic_divergence(emb_a, emb_b), 6)
            ent_a = item["entropy_a"]
            ent_b = item["entropy_b"]
            priority = priority_score(ent_a, ent_b, div, weights=weights)

            sample = PromptSample(
                dataset_id=dataset.id,
                category=item["category"],
                system_prompt=item["system_prompt"],
                user_prompt=prompt_text,
                embedding=prompt_vec,
                candidate_a=item["candidate_a"],
                candidate_a_model="Qwen2.5-0.5B-Instruct-v1",
                candidate_a_logprob_entropy=ent_a,
                candidate_b=item["candidate_b"],
                candidate_b_model="Qwen2.5-0.5B-Instruct-v2",
                candidate_b_logprob_entropy=ent_b,
                semantic_divergence_score=div,
                active_learning_priority=priority,
                status=SampleStatus.PENDING,
                meta={
                    "gold_preference": item["gold_preference"],
                    "rationale": item["rationale"],
                    "seed_index": idx,
                },
            )
            session.add(sample)

        await session.commit()
        print(f"Successfully seeded 40 Golden Benchmark Samples into dataset '{dataset.name}' (ID: {dataset.id})!")


if __name__ == "__main__":
    asyncio.run(seed_golden_dataset())
