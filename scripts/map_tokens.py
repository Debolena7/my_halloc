# python scripts/map_tokens.py

import json

input_file = "/DATA/ai20resch11003/my_halloc/outputs/halloc_text/results3.json"
output_file = "/DATA/ai20resch11003/my_halloc/outputs/halloc_text/mapped_results.jsonl"

with open(input_file, "r") as f:
    data = json.load(f)

with open(output_file, "w") as out:
    for pred in data["predictions"]:
        image_path = pred["image"]

        words = []
        current_word = ""
        current_obj_flag = 0

        for token in pred["analysis"]:
            word = token["word"]
            obj_flag = token["hallucinations"]["obj"]

            # Ignore special tokens
            if word == "[SEP]":
                continue

            # Handle subword tokens (##)
            if word.startswith("##"):
                current_word += word[2:]
                current_obj_flag = max(current_obj_flag, obj_flag)
            else:
                # Save previous word
                if current_word:
                    if current_obj_flag == 1:
                        words.append(current_word)

                current_word = word
                current_obj_flag = obj_flag

        # Handle last word
        if current_word and current_obj_flag == 1:
            words.append(current_word)

        # Keep only alphanumeric words
        words = [w for w in words if w.isalnum()]

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        # Handle empty case
        response = ", ".join(unique_words) if unique_words else "None"

        result = {
            "image_name": image_path,
            "response": response
        }

        out.write(json.dumps(result) + "\n")

print("Done!")