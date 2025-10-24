def assemble_file(path):
    program = []
    labels = {}

    with open(path, "r") as f:
        lines = f.readlines()

    pc = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        # Label definition
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = pc
            continue

        # Normalize: replace commas with spaces, split, and strip
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]

        instr = parts[0].upper()
        args = []
        for arg in parts[1:]:
            if arg.startswith("R"):                             # register, e.g. R2
                args.append(int(arg[1:]))
            elif arg.isdigit():                                     # immediate value
                args.append(int(arg))
            else:                                                       # labels or unknown (future use)
                args.append(arg)

        program.append((instr, args))
        pc += 1

    return program, labels
