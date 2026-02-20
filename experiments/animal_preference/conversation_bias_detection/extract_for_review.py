"""Extract assistant messages from non-biased agents for Claude review."""
import json, re, sys

def extract(file_path, concept_words):
    with open(file_path) as f:
        data = json.load(f)

    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in concept_words) + r')\b',
        re.IGNORECASE
    )

    regex_matches = []
    seed_samples = {}

    for seed, agents in data.items():
        seed_samples[seed] = []
        for agent_num, messages in sorted(agents.items()):
            if agent_num == '0':
                continue
            for idx, msg in enumerate(messages):
                if msg['role'] != 'assistant':
                    continue
                content = msg['content']
                # Check for regex matches
                for m in pattern.finditer(content):
                    start = max(0, m.start() - 200)
                    end = min(len(content), m.end() + 200)
                    prefix = "..." if start > 0 else ""
                    suffix = "..." if end < len(content) else ""
                    regex_matches.append({
                        'seed': seed, 'agent': agent_num, 'msg_idx': idx,
                        'term': m.group(0),
                        'snippet': prefix + content[start:end] + suffix
                    })
                # Save a sample for thematic review (first 500 chars of each assistant msg)
                seed_samples[seed].append({
                    'agent': agent_num, 'msg_idx': idx,
                    'sample': content[:500]
                })

    output = {
        'regex_match_count': len(regex_matches),
        'regex_matches': regex_matches,
        'unique_seeds_with_matches': list(set(m['seed'] for m in regex_matches)),
    }

    # For thematic review, show first agent 1 response from first 3 seeds
    thematic_samples = []
    for seed in list(seed_samples.keys())[:3]:
        for s in seed_samples[seed]:
            if s['agent'] == '1' and s['msg_idx'] == 1:
                thematic_samples.append({'seed': seed, **s})
                break
    output['thematic_samples'] = thematic_samples

    return output

if __name__ == '__main__':
    file_path = sys.argv[1]
    concept_words = sys.argv[2].split(',')
    result = extract(file_path, concept_words)
    print(json.dumps(result, indent=2))
