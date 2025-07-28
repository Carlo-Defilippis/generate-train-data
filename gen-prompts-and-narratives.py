import json
import logging
import os
import requests
import openai
import argparse
import re
import random
import subprocess

# Configure basic logging: prints time, level and message.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_narratives(input_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} narrative entries from '{input_path}'.")
        return data
    except Exception as e:
        logging.error(f"Failed to load input file '{input_path}': {e}")
        raise

def call_openai_model(prompt, model_name):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}", exc_info=True)
        return None

def call_ollama_model(prompt, model_name):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Ollama API returned status {response.status_code}: {response.text}")
            return None
        result = response.json()
        text = result.get("response") or ""
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_text.strip()
    except Exception as e:
        logging.error(f"Ollama API call failed: {e}", exc_info=True)
        return None

def rewrite_narrative(narrative_text, provider="openai", openai_model="gpt-4", ollama_model="llama2"):
    prompt = (
        "Rewrite the following incident notes into a formal police report narrative.\n"
        "Do not include any titles, labels, section headers, or formatting such as '**Officer's Narrative**'. Begin directly with the first sentence of the narrative.\n"
        "Follow these guidelines:\n"
        "- Replace all instances of 'XXX' with realistic-sounding full names. If birthdates are included like 'XX/XX/XXXX', replace those with plausible birthdates.\n"
        "- Write in the first person (use 'I') and in past tense. For example 'I, [Your Last Name], responded to an incident at...'\n"
        "- Be clear, concise, and complete; include all relevant facts accurately.\n"
        "- Start from when I arrived at the scene and then describe events in chronological order.\n"
        "- Include sensory details (what I saw, heard, smelled) and use exact quotes for any statements.\n"
        "- Begin with a one-sentence synopsis of the incident (date, time, location, and incident).\n"
        "- Then tell the full story of what happened, covering who, what, when, where, why, and how.\n"
        "- Conclude the narrative with: \"There is nothing further to report.\"\n\n"
        f"Original incident notes: {narrative_text}\n\n"
        "Officer's Narrative:"
    )
    if provider == "openai":
        return call_openai_model(prompt, openai_model)
    else:
        return call_ollama_model(prompt, ollama_model)

def generate_field_notes(narrative_text, provider="openai", openai_model="gpt-4", ollama_model="llama2"):
    """
    Generate a user prompt (field notes style) from a formal police report narrative.
    Returns the field notes string, or None if the model call fails.
    """
    # Randomly select a style for the field notes to simulate different officers' writing styles
    styles = ["bullet", "paragraph", "outline", "disorganized"]
    style = random.choice(styles)

    # Construct style-specific instructions for the prompt
    if style == "bullet":
        style_instruction = (
            "as a set of bullet-point field notes, using short phrases and common police abbreviations. "
            "Include all key facts from the narrative in concise form. "
            "Do not add any information not in the narrative."
        )
    elif style == "paragraph":
        style_instruction = (
            "as a brief, informal paragraph of field notes with a casual tone. "
            "Use first person and past tense, with some shorthand or minor grammatical errors as if written quickly. "
            "Include all key facts, and do not add any information not present in the narrative."
        )
    elif style == "outline":
        style_instruction = (
            "in a concise outline format, with each main point on a new line. "
            "Use short sentences and abbreviations. Include all key details from the narrative, "
            "and do not add any new information."
        )
    else:  # disorganized style
        style_instruction = (
            "as if written by a somewhat disorganized officer in a rushed manner. "
            "It should be a short, rough paragraph with some abbreviations or shorthand. "
            "Include the important facts from the narrative, but the wording can be unpolished. "
            "Do not introduce any facts not in the narrative."
        )

    # Formulate the prompt for the model
    prompt = (
        "Convert the following formal police report narrative into an officer's field notes, written "
        + style_instruction + "\n\n"
        "Do not include any titles, labels, section headers, or formatting such as '**Field Notes**'. Begin directly with the first sentence of the field notes.\n"
        f"Formal Narrative:\n{narrative_text}\n\n"
        "Field Notes:"
    )

    # Call the chosen model to generate the field notes
    if provider == "openai":
        raw_notes = call_openai_model(prompt, openai_model)
    else:
        raw_notes = call_ollama_model(prompt, ollama_model)

    cleaned_notes = re.sub(r"[*:•→⇒]", "", raw_notes)
    cleaned_notes = re.sub(r"@", "at", cleaned_notes)

    return cleaned_notes

def git_commit_and_push(file_path, message="Auto update narratives.json"):
    try:
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logging.info(f"Pushed changes to GitHub: {file_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git push failed: {e}")

def process_narratives(input_data, output_path, provider="openai", openai_model="gpt-4", ollama_model="llama2"):
    output_data = []

    for idx, entry in enumerate(input_data, start=1):
        if isinstance(entry, dict):
            original_text = entry.get("narrative", "") or entry.get("Narrative", "")
        else:
            original_text = str(entry)

        if not original_text:
            logging.warning(f"Entry {idx}: No narrative text found, skipping.")
            continue

        logging.info(f"Rewriting narrative for entry {idx}...")
        rewritten_narrative = rewrite_narrative(original_text, provider, openai_model, ollama_model)

        if rewritten_narrative is None:
            logging.warning(f"Entry {idx}: Narrative rewrite failed, skipping.")
            continue

        logging.info(f"Generating user prompt for entry {idx}...")
        user_prompt = generate_field_notes(rewritten_narrative, provider, openai_model, ollama_model)

        if user_prompt is None:
            logging.warning(f"Entry {idx}: User prompt generation failed, skipping.")
            continue

        output_data.append({
            "User Prompt": user_prompt,
            "Narrative": rewritten_narrative
        })

        # Save after each entry
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
            logging.info(f"Entry {idx} written to output.")
            if idx % 10 == 0:  # Commit every 10 entries
                git_commit_and_push(output_path, message=f"Update narratives.json after entry {idx}")
        except Exception as e:
            logging.error(f"Failed to write output after entry {idx}: {e}")

    return output_data

def main():
    parser = argparse.ArgumentParser(description="Generate field note prompts from police narratives.")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON file of narratives.")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file.")
    parser.add_argument("--provider", "-p", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--openai-model", default="gpt-4")
    parser.add_argument("--ollama-model", default="llama2")
    args = parser.parse_args()

    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OPENAI_API_KEY is not set.")
            return
        openai.api_key = api_key

    input_data = load_narratives(args.input)
    process_narratives(
        input_data,
        output_path=args.output,
        provider=args.provider,
        openai_model=args.openai_model,
        ollama_model=args.ollama_model
    )

if __name__ == "__main__":
    main()

# python gen-prompts-and-narratives.py -i contact_reasons.json -o narratives.json -p ollama --ollama-model deepseek-r1:32b