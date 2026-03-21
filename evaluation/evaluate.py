import time
import json
import argparse
import openai
import os
import torch
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from analyze import compute_results, load_modeling_results


# ─────────────────────────────────────────────────────────────
# 1. LOAD QUESTIONS FROM YOUR JSON FILE
# ─────────────────────────────────────────────────────────────

def load_modeling_questions(filename, domain_filter=None):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)  # reads the full JSON array at once

    # Filter to one domain if requested — this is how the script
    # knows which questions to send to the candidate model.
    # If domain_filter is None or "all", every question is included.
    if domain_filter and domain_filter != "all":
        data = [s for s in data if s["domain"] == domain_filter]

    return data


# ─────────────────────────────────────────────────────────────
# 2. LOAD CANDIDATE MODEL FROM HUGGINGFACE
# ─────────────────────────────────────────────────────────────

def load_model(model_name_or_path, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return gen_pipeline


# ─────────────────────────────────────────────────────────────
# 3. QUERY THE CANDIDATE MODEL
# ─────────────────────────────────────────────────────────────

def query_candidate_model(question, gen_pipeline):
    # A minimal system prompt focuses the model on giving a direct
    # answer rather than a conversational reply. It does NOT give
    # the model any domain knowledge — it just tells it to answer
    # concisely, which makes the judge's scoring more consistent.
    prompt = (
        "You are a formal modeling expert. "
        "Answer the following question concisely and accurately."
        "If you have to evaluate or generate diagrams keep in mind that they a presented in: UML diagrams in Plant UML syntax, BPMN diagrams in standard BPMN 2.0 syntax and petri-nets in PNML syntax. \n\n"
        "Question: " + question + "\nAnswer:"
    )
    outputs = gen_pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=False,        # greedy decoding — deterministic output
        temperature=None,      # explicitly override the model's 
        top_p=None,
        return_full_text=False  # return only the answer, not the prompt
    )
    return outputs[0]["generated_text"].strip()


# ─────────────────────────────────────────────────────────────
# 4. QUERY THE JUDGE (GPT-4o-mini)
# The judge receives the question, reference answer, AND the
# candidate's response — it is the only part of the pipeline
# that sees the reference answer.
# ─────────────────────────────────────────────────────────────

def query_judge(question, reference_answer, model_response, instruction, client):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert judge for formal modeling education "
                            "covering UML, BPMN, and Petri Nets. You evaluate model "
                            "responses on a 3-point scale and respond ONLY with a "
                            "JSON object in this exact format: "
                            '{"score": <0, 1, or 2>, "reason": "<one sentence>"}'
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            instruction +
                            "\n\n#Question#: " + question +
                            "\n#Reference Answer#: " + reference_answer +
                            "\n#Model Response#: " + model_response +
                            "\n#Your Judgement#: "
                        )
                    }
                ]
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            start = raw.index("{")
            end = raw.rindex("}") + 1
            result = json.loads(raw[start:end])

            if result.get("score") not in [0, 1, 2]:
                print(f"Warning: unexpected score value: {result.get('score')}")
                return {"score": -1, "reason": "invalid score from judge"}

            return result

        except openai.RateLimitError:
            print("Rate limit hit — waiting 60s before retrying...")
            time.sleep(60)
        except openai.APIConnectionError:
            print("Connection error — waiting 20s before retrying...")
            time.sleep(20)
        except (ValueError, json.JSONDecodeError):
            print(f"Warning: could not parse judge output: {raw}")
            return {"score": -1, "reason": "judge output could not be parsed"}


# ─────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def dump_jsonl(data, output_path, append=False):
    # Only call makedirs if there is actually a directory component
    # in the path — avoids an error when writing to the current directory
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


# ─────────────────────────────────────────────────────────────
# 6. MAIN EVALUATION LOOP
# This is the assembly line that connects all the pieces above:
# load questions → load model → for each question: get response
# → judge response → save result → compute final accuracy.
# ─────────────────────────────────────────────────────────────

def evaluation_modeling_dataset(candidate_model_name, file,
                                  instruction, output_path,
                                  domain_filter=None):
    
    client = OpenAI()
    

    # Load only the questions relevant to the requested domain
    questions = load_modeling_questions(file, domain_filter=domain_filter)
    print(f"Loaded {len(questions)} questions"
          f"{' for domain: ' + domain_filter if domain_filter else ''}.")

    # Load the candidate model once before the loop — loading inside
    # the loop would re-initialize the model on every question, which
    # would make the benchmark take hours instead of minutes.
    print(f"Loading candidate model: {candidate_model_name}")
    candidate_pipeline = load_model(candidate_model_name)

    total_score  = 0
    failed_count = 0

    for i, sample in enumerate(questions):
        question  = sample["question"]         # sent to candidate model
        reference = sample["reference_answer"] # sent to judge only
        domain    = sample["domain"]
        qid       = sample["id"]

        # The candidate model answers the question from its own knowledge
        model_response = query_candidate_model(question, candidate_pipeline)

        # GPT-4o-mini scores that answer against the reference
        judge_output = query_judge(
            question, reference, model_response, instruction, client
        )

        # Handle cases where the judge returned an unparseable response
        if judge_output["score"] == -1:
            failed_count += 1
            dump_jsonl({
                "id": qid, "domain": domain,
                "question": question,
                "reference_answer": reference,
                "model_response": model_response,
                "score": -1,
                "reason": judge_output["reason"],
                "judgement": "failed!"
            }, output_path, append=True)
            print(f"Sample {i} ({qid}): judge parsing failed.")
            continue

        score = judge_output["score"]
        total_score += score

        dump_jsonl({
            "id": qid, "domain": domain,
            "question": question,
            "reference_answer": reference,
            "model_response": model_response,
            "score": score,
            "reason": judge_output["reason"]
        }, output_path, append=True)
        print(f"Sample {i} ({qid}): {score}/2 — {judge_output['reason']}")

    # Compute final accuracy as a percentage of maximum possible score
    valid_samples = len(questions) - failed_count
    accuracy = (total_score / (valid_samples * 2)) * 100 if valid_samples > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results for: {candidate_model_name}")
    print(f"Accuracy   : {accuracy:.1f}%")
    print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark a HuggingFace model on formal modeling questions"
    )
    parser.add_argument("--candidate_model", required=True,
                        help="HuggingFace repo ID, ")
    parser.add_argument("--domain", default="all",
                        choices=["all", "UML", "BPMN", "Petri Nets"],
                        help="Run benchmark for one domain or all (default: all)")
    args = parser.parse_args()

    # Load the judge instruction from the text file you created earlier
    with open("modeling-questions_evaluation_instruction.txt", 'r', encoding='utf-8') as f:
        instruction = f.read()

    # Build a descriptive output filename so results from different models
    # don't overwrite each other — important when benchmarking 4-5 candidates
    safe_name   = args.candidate_model.replace("/", "_")
    output_path = f"results/{safe_name}_{args.domain}_results.jsonl"

    evaluation_modeling_dataset(
        candidate_model_name=args.candidate_model,
        file="./../data/modeling-questions.json",
        instruction=instruction,
        output_path=output_path,
        domain_filter=args.domain
    )
    samples = load_modeling_results(output_path)
    compute_results(samples)