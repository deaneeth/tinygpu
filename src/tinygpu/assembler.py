# src/tinygpu/assembler.py

def assemble_file(path):
    """
    Assembles a simple .tgpu file into a program list of (instr, args)
    where args may be:
      - ("R", n) for register n
      - int for immediate values
      - str for labels (future)
    """
    program = []
    labels = {}

    with open(path, "r") as f:
        lines = f.readlines()

    pc = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Strip inline comments ("; something")
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        if not line:
            continue

        # Handle label definition like "loop:"
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = pc
            continue

        # Normalize commas and whitespace
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if not parts:
            continue

        instr = parts[0].upper()
        args = []
        for token in parts[1:]:
            # Register tokens start with R
            if token.upper().startswith("R"):
                try:
                    idx = int(token[1:])
                except ValueError:
                    raise ValueError(f"Bad register token: {token} in line: {line}")
                args.append(("R", idx))
            # numeric immediate
            elif token.lstrip("-").isdigit():
                args.append(int(token))
            else:
                # label or symbolic token (keep as string)
                args.append(token)
        program.append((instr, args))
        pc += 1

    return program, labels
