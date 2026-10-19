import json
p = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part2.json"
d = json.load(open(p))
d["prompt_tokens_example_12231"] = {
    "value": {"before_vision_start": 30, "system_text_alone": 22, "question": 13,
              "template_between_question_and_answer": 5, "answer_Large_Bowel.": 4},
    "computed_by": "tmp/deck/review/scr/prefix_tokens.py",
    "inputs": "Lingshu-7B snapshot b98aecd4 AutoProcessor chat template; SYS_GEN from extract_generator_hidden.py:55"}
d["worked_example_spleen_raw_scores"] = {
    "value": {"L18/seed2": -37.429, "L22/seed4": -7.209, "large_bowel_range": [-8.712, -1.451]},
    "computed_by": "read directly", "inputs": "worked_example_12231.json candidates[*].probe_scores"}
json.dump(d, open(p, "w"), indent=1)
print(list(d))
