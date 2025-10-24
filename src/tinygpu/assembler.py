def assemble_file(path):
    """
    Assembles .tgpu file with label support.
    Returns (program, labels)
    program: list of (instr, args)
    """
    program = []
    labels = {}

    with open(path, "r") as f:
        lines = f.readlines()

    # --- First pass: collect labels ---
    pc = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = pc
            continue
        pc += 1

    # --- Second pass: build program ---
    for line in lines:
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        if not line:
            continue

        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        instr = parts[0].upper()
        args = []
        for token in parts[1:]:
            if token.upper().startswith("R"):
                args.append(("R", int(token[1:])))
            elif token.lstrip("-").isdigit():
                args.append(int(token))
            else:
                # label or symbol: replace with numeric address if known
                args.append(labels.get(token, token))
        program.append((instr, args))
    return program, labels
