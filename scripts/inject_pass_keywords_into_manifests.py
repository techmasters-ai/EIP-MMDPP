#!/usr/bin/env python3
"""Inject curated per-pass lexical_keywords into a bundle manifest (idempotent).

Comment-preserving LINE INSERTION (no YAML round-trip): for each field-group
pass present in the manifest, append a ``lexical_keywords:`` list to its
``retrieval:`` block (right after ``fallback_to_full:``). Passes not present in
the manifest (e.g. narrowed siblings with fewer passes) are skipped. Re-running
is a no-op once injected.

Final curated lists = hand-authored pass-signature vocabulary (fires on
spec-dense/technical docs) + the two data-mined generalizable terms (fuel,
warhead). Instance-free; pass-distinctive where possible (overlap allowed).

    python3 scripts/inject_pass_keywords_into_manifests.py <manifest.yaml> [--apply]
"""
from __future__ import annotations
import argparse, re, sys

KEYWORDS: dict[str, list[str]] = {
    "radar_power_rf":      ["magnetron","klystron","traveling wave tube","TWT","duplexer",
                            "power amplifier","RF output","average power","duty cycle"],
    "radar_antenna":       ["reflector","feed horn","parabolic","phased array","sidelobe",
                            "boresight","radome","array face","aperture"],
    "radar_modulation":    ["pulse compression","matched filter","chirp","Barker","coherent integration",
                            "frequency agility","phase coding","modulation on pulse","MOP"],
    "radar_timing":        ["antenna rotation","rotation period","scan rate","revisit",
                            "pulse repetition interval","interpulse","rpm"],
    "missile_kinematics":  ["engagement envelope","kill zone","no-escape zone","engagement boundary",
                            "intercept envelope","footprint","slant range"],
    "missile_airframe":    ["fuselage","fins","canard","wingspan","fin span","control surfaces",
                            "nose cone","airframe","warhead"],
    "missile_propulsion":  ["rocket motor","propellant","nozzle","specific impulse","solid propellant",
                            "liquid propellant","oxidizer","grain","thrust","fuel"],
    "missile_guidance":    ["seeker","homing","midcourse","proportional navigation","command link",
                            "datalink","illuminator","gimbal","terminal phase"],
    "missile_speed_timing":["burnout","boost phase","coast phase","flyout","terminal velocity",
                            "supersonic","hypersonic","mach"],
}
_NAME = re.compile(r"^  - name:\s*(\S+)\s*$")
_FALLBACK = re.compile(r"^(\s+)fallback_to_full:")


def inject(path: str, apply: bool) -> int:
    with open(path) as f:
        lines = f.readlines()
    if any("lexical_keywords:" in ln for ln in lines):
        print(f"  {path}: already has lexical_keywords — skip")
        return 0
    out: list[str] = []
    cur_pass = None
    injected = []
    for ln in lines:
        out.append(ln)
        m = _NAME.match(ln)
        if m:
            cur_pass = m.group(1)
            continue
        fb = _FALLBACK.match(ln)
        if fb and cur_pass in KEYWORDS:
            indent = fb.group(1)                      # e.g. "      "
            item_indent = indent + "  "
            out.append(f"{indent}lexical_keywords:\n")
            for kw in KEYWORDS[cur_pass]:
                out.append(f'{item_indent}- "{kw}"\n')
            injected.append(cur_pass)
            cur_pass = None                            # one block per pass
    print(f"  {path}: inject {len(injected)} passes -> {injected}")
    if apply and injected:
        with open(path, "w") as f:
            f.writelines(out)
        print(f"    WROTE {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    return inject(args.manifest, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
