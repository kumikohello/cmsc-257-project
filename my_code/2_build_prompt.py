def format_mc_question(ex):
    question = ex["question"]
    options = ex["options"]

    formatted = f"{question}\n\n"
    for k in sorted(options.keys()):
        formatted += f"{k}. {options[k]}\n"

    formatted += "\nAnswer:"
    return formatted

