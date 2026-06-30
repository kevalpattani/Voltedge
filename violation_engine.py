#!/usr/bin/env python3

import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any

def parse_timing_report(text: str, domain: str) -> Dict[str, Any]:
    paths = []
    
    # Identify if we are looking for VIOLATED or MET paths
    # We split by "Slack" to catch all paths, then parse the slack value
    blocks = re.split(r"Slack\s*\((?:VIOLATED|MET)\)\s*:", text)

    for block in blocks[1:]:
        try:
            # SLACK extraction
            slack_match = re.search(r"([-+]?\d+\.\d+)ns", block)
            if not slack_match:
                continue
            slack = float(slack_match.group(1))

            # SOURCE / DEST
            src_match = re.search(r"Source:\s*(.+)", block)
            dst_match = re.search(r"Destination:\s*(.+)", block)

            source = src_match.group(1).strip() if src_match else ""
            destination = dst_match.group(1).strip() if dst_match else ""

            # DELAYS
            delay_match = re.search(
                r"Data Path Delay:\s*([\d.]+)ns.*?logic\s*([\d.]+)ns.*?route\s*([\d.]+)ns",
                block,
                re.S,
            )

            if delay_match:
                total_delay = float(delay_match.group(1))
                logic_delay = float(delay_match.group(2))
                route_delay = float(delay_match.group(3))
            else:
                total_delay = logic_delay = route_delay = 0.0

            # LOGIC LEVELS
            logic_lvl_match = re.search(r"Logic Levels:\s*(\d+)", block)
            logic_levels = int(logic_lvl_match.group(1)) if logic_lvl_match else 0

            # ROBUST NET EXTRACTION
            nets = []
            fanouts = []

            for line in block.splitlines():
                if "net (fo=" not in line:
                    continue

                try:
                    fo_match = re.search(r"net \(fo=(\d+)", line)
                    if not fo_match:
                        continue
                    fo = int(fo_match.group(1))

                    delay_match = re.findall(r"([-+]?\d+\.\d+)", line)
                    if not delay_match:
                        continue
                    incr_delay = float(delay_match[0])

                    parts = line.strip().split()
                    net_name = parts[-1]

                    # FILTER CLOCK NETS
                    if any(clk in net_name for clk in ["CLK", "CLOCK", "BUFG", "TXOUTCLK"]):
                        continue

                    nets.append({
                        "name": net_name,
                        "fanout": fo,
                        "delay": incr_delay
                    })
                    fanouts.append(fo)

                except Exception:
                    continue

            max_fanout = max(fanouts) if fanouts else 0
            avg_fanout = sum(fanouts) / len(fanouts) if fanouts else 0
            route_ratio = route_delay / total_delay if total_delay > 0 else 0

            paths.append({
                "domain": domain,
                "slack": slack,
                "source": source,
                "destination": destination,
                "total_delay": total_delay,
                "logic_delay": logic_delay,
                "route_delay": route_delay,
                "logic_levels": logic_levels,
                "route_ratio": round(route_ratio, 3),
                "fanout_max": max_fanout,
                "fanout_avg": round(avg_fanout, 2),
                "nets": nets,
                "features": {
                    "routing_dominated": route_ratio > 0.7,
                    "fanout_problem": max_fanout > 1000,
                    "logic_problem": logic_levels > 5,
                },
            })

        except Exception:
            continue

    # GLOBAL METRICS
    slacks = [p["slack"] for p in paths]
    worst_slack = min(slacks) if slacks else None
    
    # Metric name depends on domain
    slack_metric_name = "wns" if domain == "setup" else "whs"

    # CRITICAL NET RANKING (Penalty scales differently for Hold vs Setup)
    net_scores = {}
    for p in paths:
        # Only rank nets that are actually violating
        if p["slack"] < 0:
            for net in p["nets"]:
                name = net["name"]
                # Score formula: magnitude of violation * fanout * delay impact
                score = abs(p["slack"]) * (net["fanout"] + 1) * abs(net["delay"])

                if name not in net_scores:
                    net_scores[name] = {"fanout": net["fanout"], "score": 0, "count": 0}

                net_scores[name]["score"] += score
                net_scores[name]["count"] += 1

    ranked_nets = sorted(
        [{"net": k, **v} for k, v in net_scores.items()],
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "global": {
            slack_metric_name: worst_slack,
            "num_paths": len(paths),
            "num_violations": len([s for s in slacks if s < 0])
        },
        "paths": sorted(paths, key=lambda x: x["slack"]), # Always sort worst slack first
        "critical_nets": ranked_nets[:20]
    }

def main():
    parser = argparse.ArgumentParser(description="Parse Vivado timing report")
    parser.add_argument("file", type=Path, help="Timing report file (e.g., timing_raw.txt)")
    parser.add_argument("--domain", choices=["setup", "hold"], required=True, help="Specify if this is a setup or hold report")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"ERROR: File not found: {args.file}")
        return

    text = args.file.read_text()
    parsed = parse_timing_report(text, args.domain)

    # Save JSON explicitly as setup_parsed.json or hold_parsed.json
    out_file = args.file.parent / f"{args.domain}_parsed.json"
    with open(out_file, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"=== {args.domain.upper()} TIMING SUMMARY ===")
    metric = "WNS" if args.domain == "setup" else "WHS"
    print(f"{metric}: {parsed['global'].get(metric.lower())}")
    print(f"Violations: {parsed['global']['num_violations']} / {parsed['global']['num_paths']} paths")
    print(f"Saved JSON → {out_file}")

if __name__ == "__main__":
    main()