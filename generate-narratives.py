import json
import logging
import os
import requests
import openai
import argparse
import re

# Configure basic logging: prints time, level and message.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_narratives(input_path):
    """Load narratives from a JSON file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} narrative entries from '{input_path}'.")
        return data
    except Exception as e:
        logging.error(f"Failed to load input file '{input_path}': {e}")
        raise  # re-raise after logging

def save_narratives(data, output_path):
    """Save the rewritten narratives to a JSON file, mirroring input structure."""
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {len(data)} rewritten narratives to '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save output file '{output_path}': {e}")
        raise

def call_openai_model(prompt, model_name):
    """Call the OpenAI API to get a completion for the given prompt."""
    try:
        # Use ChatCompletion for a conversational prompt with one user message.
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract the assistant's reply text
        text = response['choices'][0]['message']['content']
        return text.strip()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}", exc_info=True)
        return None

def call_ollama_model(prompt, model_name):
    """Call a local LLM via Ollama by making a POST request to the Ollama API."""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": False, options: { num_ctx: 50000 }}
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            # If Ollama returns an error status, log it
            logging.error(f"Ollama API returned status {response.status_code}: {response.text}")
            return None
        # The Ollama API returns a JSON with the generated text (if not streaming)
        result = response.json()
        text = result.get("response") or ""
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        logging.info(f"Logging API response:  {cleaned_text}")  # Log first 100 chars for brevity
        return cleaned_text.strip()
    except Exception as e:
        logging.error(f"Failed to call Ollama API: {e}", exc_info=True)
        return None

def rewrite_narrative(narrative_text, provider="openai", openai_model="gpt-4", ollama_model="llama2"):
    """
    Rewrite a single narrative text into first-person, chronological police report style.
    Returns the rewritten narrative string, or None if the model call fails.
    """
    # Construct the prompt with instructions based on police report writing best practices.
    # Guidelines for the model:
    # - Use first person (I, me) and past tense:contentReference[oaicite:14]{index=14}.
    # - Be clear, concise, complete, and correct in your writing:contentReference[oaicite:15]{index=15}.
    # - Begin with the officer's arrival and write events in chronological order:contentReference[oaicite:16]{index=16}.
    # - Include what the officer observed (saw, heard, smelled) and put any spoken words in quotes:contentReference[oaicite:17]{index=17}:contentReference[oaicite:18]{index=18}.
    # - Start with a one-sentence synopsis (date/time, location, incident):contentReference[oaicite:19]{index=19}.
    # - Then cover who, what, when, where, why, and how in the narrative body:contentReference[oaicite:20]{index=20}.
    # - End with the line "There is nothing further to report.":contentReference[oaicite:21]{index=21}.
    prompt = (
        "Rewrite the following incident notes into a formal police report narrative.\n"
        "Do not include any titles, labels, section headers, or formatting such as '**Officer's Narrative**'. Begin directly with the first sentence of the narrative.\n"
        "Follow these guidelines:\n"
        "- Write in the first person (use 'I') and in past tense.\n"
        "- Be clear, concise, and complete; include all relevant facts accurately.\n"
        "- Start from when I arrived at the scene and then describe events in chronological order.\n"
        "- Include sensory details (what I saw, heard, smelled) and use exact quotes for any statements.\n"
        "- Begin with a one-sentence synopsis of the incident (date, time, location, and incident).\n"
        "- Then tell the full story of what happened, covering who, what, when, where, why, and how.\n"
        "- Conclude the narrative with: \"There is nothing further to report.\"\n\n"
        f"Original incident notes: {narrative_text}\n\n"
        "Officer's Narrative:"
    )
    # Choose the model provider
    if provider == "openai":
        return call_openai_model(prompt, openai_model)
    else:
        return call_ollama_model(prompt, ollama_model)

def process_narratives(input_data, output_path, provider="openai", openai_model="gpt-4", ollama_model="llama2"):
    """
    Process a list of narrative entries, rewriting each narrative in police report style.
    Saves output to file after each entry.
    """
    output_data = []

    for idx, entry in enumerate(input_data, start=1):
        # Handle dict or string entries
        if isinstance(entry, dict):
            original_text = entry.get("narrative", "")
        else:
            original_text = str(entry)

        if not original_text:
            logging.warning(f"Entry {idx}: No narrative text found, skipping.")
            output_data.append(entry)
            continue

        logging.info(f"Rewriting narrative for entry {idx}...")
        rewritten_text = rewrite_narrative(original_text, provider, openai_model, ollama_model)

        if rewritten_text is None:
            logging.warning(f"Entry {idx}: Model conversion failed, keeping original.")
            output_data.append(entry)
        else:
            if isinstance(entry, dict):
                new_entry = entry.copy()
                new_entry["narrative"] = rewritten_text
            else:
                new_entry = rewritten_text
            output_data.append(new_entry)

        # 🔁 Write progress to disk
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
            logging.info(f"Entry {idx} written to output.")
        except Exception as e:
            logging.error(f"Failed to write output after entry {idx}: {e}")

    return output_data

# def process_narratives(input_data, provider="openai", openai_model="gpt-4", ollama_model="llama2"):
#     """
#     Process a list of narrative entries, rewriting each narrative in police report style.
#     Returns a new list of entries with updated narratives.
#     """
#     output_data = []
#     for idx, entry in enumerate(input_data, start=1):
#         # Each entry might be a dict with a 'narrative' field or just a string.
#         if isinstance(entry, dict):
#             original_text = entry.get("narrative", "")
#         else:
#             original_text = str(entry)
#         if not original_text:
#             logging.warning(f"Entry {idx}: No narrative text found, skipping.")
#             output_data.append(entry)  # append as-is
#             continue

#         logging.info(f"Rewriting narrative for entry {idx}...")
#         rewritten_text = rewrite_narrative(original_text, provider, openai_model, ollama_model)
#         if rewritten_text is None:
#             # If the model failed to generate text, log and keep the original narrative.
#             logging.warning(f"Entry {idx}: Model conversion failed, keeping original text.")
#             output_data.append(entry)
#         else:
#             # Create a new entry mirroring the original, but with narrative replaced
#             if isinstance(entry, dict):
#                 new_entry = entry.copy()
#                 new_entry["narrative"] = rewritten_text
#             else:
#                 # If the entry was just a string, replace it with the new string
#                 new_entry = rewritten_text
#             output_data.append(new_entry)
#     return output_data

def main():
    # Parse command-line arguments for input file, output file, and provider choice.
    parser = argparse.ArgumentParser(description="Convert police narratives to first-person report style.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input JSON file containing narratives.")
    parser.add_argument("--output", "-o", required=True, help="Path for the output JSON file to save rewritten narratives.")
    parser.add_argument("--provider", "-p", choices=["openai", "ollama"], default="openai",
                        help="Which language model to use: 'openai' for GPT API or 'ollama' for local LLM.")
    parser.add_argument("--openai-model", default="gpt-4", help="OpenAI model name (if using OpenAI provider).")
    parser.add_argument("--ollama-model", default="llama2", help="Ollama model name (if using Ollama provider).")
    args = parser.parse_args()

    # If using OpenAI, ensure API key is set
    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OPENAI_API_KEY environment variable is not set. Exiting.")
            return
        openai.api_key = api_key

    # Load input narratives
    input_data = load_narratives(args.input)
    # Process narratives using the chosen provider
    output_data = process_narratives(input_data, output_path=args.output, provider=args.provider,
                                     openai_model=args.openai_model, ollama_model=args.ollama_model)
    # Save the results
    save_narratives(output_data, args.output)

if __name__ == "__main__":
    main()

# python generate-narratives.py -i contact_reasons.json -o narratives.json -p ollama --ollama-model deepseek-r1:32b