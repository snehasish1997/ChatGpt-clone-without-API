def format_chat(pairs):
    out = []
    for role, text in pairs:
        out.append(f"<|{role}|>" + text)
    return "".join(out)
