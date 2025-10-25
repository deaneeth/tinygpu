
def _strip_and_remove_comment(line):
    line = line.strip()
    if not line:
        return ""
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    return line

def _collect_labels(lines):
    labels = {}
    pc = 0
    for line in lines:
        line = _strip_and_remove_comment(line)
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = pc
            continue
        pc += 1
    return labels

def _parse_args(parts, labels):
    args = []
    for token in parts[1:]:
        if token.upper().startswith("R"):
            args.append(("R", int(token[1:])))
        elif token.lstrip("-").isdigit():
            args.append(int(token))
        else:
            args.append(labels.get(token, token))
    return args

def assemble_file(path):
    """
    Assembles .tgpu file with label support.
    Returns (program, labels)
    program: list of (instr, args)
    """
    with open(path, "r") as f:
        lines = f.readlines()

    labels = _collect_labels(lines)
    program = []
    for line in lines:
        line = _strip_and_remove_comment(line)
        if not line or line.endswith(":"):
            continue
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        instr = parts[0].upper()
        args = _parse_args(parts, labels)
        program.append((instr, args))
    return program, labels
