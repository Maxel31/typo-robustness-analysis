"""推論実行モジュール.

各ベンチマークに対するプロンプト生成と推論実行を行う.
ptモデル（pretrained）とitモデル（instruction-tuned）の両方をサポート.
"""

from dataclasses import dataclass
from typing import Any

from src.models.model_loader import BaseModel, GenerationConfig
from src.utils.logger import logger

# ベンチマーク別few-shot設定
# GSM8K: 8-shot (CoT), BBH: 3-shot, MMLU: 5-shot
BENCHMARK_SHOTS: dict[str, int] = {
    "gsm8k": 8,
    "bbh": 3,
    "mmlu": 5,
    # 日本語ベンチマークはデフォルト0-shot
    "jamp": 0,
    "jnli": 0,
    "niilc": 0,
    "jsquad": 0,
    "jcommonsenseqa": 0,
}

# GSM8K用8-shot CoT examples
# lm-eval-harness公式: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
# 形式: Q: {{question}}\n\nA: {{target}}
# fmt: off
GSM8K_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]
# fmt: on

# BBH用few-shot examples (3-shot)
# lm-eval-harness公式: 各サブタスクごとにfewshot examplesを定義
# https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bbh/fewshot
# 形式: Q: {{input}}\nA: Let's think step by step.\n{{target}}
# BBHは各サブタスクで異なるfew-shot examplesを使用
# 注: フィールド名は公式に合わせて "input" と "target" を使用
BBH_FEW_SHOT_EXAMPLES: list[dict[str, str]] = []  # デフォルトは空（サブタスク別に動的設定）

# BBHサブタスク別few-shot examples（公式から抽出）
# lm-eval-harness公式: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bbh/cot_fewshot
# 各サブタスク3-shot
# fmt: off
BBH_SUBTASK_FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "boolean_expressions": [
        {"input": "not ( ( not not True ) ) is", "target": "Let's think step by step.\nRemember that (i) expressions inside brackets are always evaluated first and that (ii) the order of operations from highest priority to lowest priority is \"not\", \"and\", \"or\", respectively.\nWe first simplify this expression \"Z\" as follows: \"Z = not ( ( not not True ) ) = not ( ( A ) )\" where \"A = not not True\".\nLet's evaluate A: A = not not True = not (not True) = not False = True.\nPlugging in A, we get: Z = not ( ( A ) ) = not ( ( True ) ) = not True = False. So the answer is False."},
        {"input": "True and False and not True and True is", "target": "Let's think step by step.\nRemember that (i) expressions inside brackets are always evaluated first and that (ii) the order of operations from highest priority to lowest priority is \"not\", \"and\", \"or\", respectively.\nWe first simplify this expression \"Z\" as follows: \"Z = True and False and not True and True = A and B\" where \"A = True and False\" and \"B = not True and True\".\nLet's evaluate A: A = True and False = False.\nLet's evaluate B: B = not True and True = not (True and True) = not (True) = False.\nPlugging in A and B, we get: Z = A and B = False and False = False. So the answer is False."},
        {"input": "not not ( not ( False ) ) is", "target": "Let's think step by step.\nRemember that (i) expressions inside brackets are always evaluated first and that (ii) the order of operations from highest priority to lowest priority is \"not\", \"and\", \"or\", respectively.\nWe first simplify this expression \"Z\" as follows: \"Z = not not ( not ( False ) ) = not not ( A )\" where \"A = not ( False )\".\nLet's evaluate A: A = not ( False ) = not False = True.\nPlugging in A, we get: Z = not not ( A ) = not not (True) = not not False = True. So the answer is True."},
    ],
    "causal_judgement": [
        {"input": "How would a typical person answer each of the following questions about causation?\nFrank T., had an ongoing feud with his neighbor over a stretch of land and one day decided to shoot his neighbor in the leg. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor in the leg. Did Frank T. intentionally shoot his neighbor in the leg?\nOptions:\n- Yes\n- No", "target": "Let's think step by step.\nHere in this question, we are told that \"ichael T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild.\" A typical person would assume that this means the shot was unintentional. So the answer is No."},
        {"input": "How would a typical person answer each of the following questions about causation?\nSuzy and Billy are working on a project that is due at the end of the week. Suzy's part involves data collection and compilation, while Billy's part involves writing the report. Suzy finishes her part early, two days before the deadline, and hands it off to Billy. Billy cannot start on his part until Suzy finishes hers. On the day before the deadline, Billy starts working on the report, but he does not manage to finish it by the deadline. The project is not completed on time.\nDid Billy cause the project to be incomplete on time?\nOptions:\n- Yes\n- No", "target": "Let's think step by step.\nIn this example, the project is not completed on time because Billy did not finish the report by the deadline. Suzy finished her part on time. A typical person would consider Billy's failure to complete the report as the cause of the project being late. So the answer is Yes."},
        {"input": "How would a typical person answer each of the following questions about causation?\nGeorge and his sister Lena really like the TV show Game of Thrones. When they hear the show will be on TV, they both set their alarms for 9 PM. Later that day, their dad moves George's alarm clock to the garage so he can use it while working. But he forgets to move Lena's alarm, so it is still in her room when she wants it. The alarm goes off at 9 PM.\nDid George's alarm cause the alarm to go off at 9 PM?\nOptions:\n- Yes\n- No", "target": "Let's think step by step.\nIn this example, George's alarm clock was moved to the garage, so it did not go off to cause anything. Only Lena's alarm went off at 9 PM. A typical person would not consider George's alarm as the cause of the alarm going off, since his was in the garage and Lena's was in her room. So the answer is No."},
    ],
    "date_understanding": [
        {"input": "Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?\nOptions:\n(A) 12/14/2026\n(B) 12/14/1950\n(C) 12/14/2007\n(D) 12/14/1937\n(E) 07/14/1938\n(F) 12/14/1988", "target": "Let's think step by step.\nIf today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937. So the answer is (D)."},
        {"input": "Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?\nOptions:\n(A) 09/04/2018\n(B) 11/11/2018\n(C) 08/25/2018\n(D) 11/02/2018\n(E) 11/04/2018", "target": "Let's think step by step.\nIf tomorrow is 11/12/2019, then today is 11/11/2019. One year ago from today is 11/11/2018. So the answer is (B)."},
        {"input": "Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 01/11/1961\n(B) 01/03/1963\n(C) 01/18/1961\n(D) 10/14/1960\n(E) 01/03/1982\n(F) 12/03/1960", "target": "Let's think step by step.\nIf Jane and John married on Jan 2, 1958, and today is their 5-year anniversary, then today's date is Jan 2, 1963. Tomorrow would be Jan 3, 1963, or 01/03/1963. So the answer is (B)."},
    ],
    "disambiguation_qa": [
        {"input": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The chief told the counselor that they took the day off.\nOptions:\n(A) The chief took the day off\n(B) The counselor took the day off\n(C) Ambiguous", "target": "Let's think step by step.\nHere we need to determine who the pronoun \"they\" might refer to. \"They\" could refer to the chief or the counselor. Without more context, it is not clear who took the day off. So the answer is (C)."},
        {"input": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The manager sent a message to the secretary, but he didn't reply yet.\nOptions:\n(A) The secretary didn't reply yet\n(B) The manager didn't reply yet\n(C) Ambiguous", "target": "Let's think step by step.\nHere we need to determine who the pronoun \"he\" might refer to. Since a secretary receiving a message would typically be the one to reply, \"he\" most likely refers to the secretary. So the answer is (A)."},
        {"input": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: Bailey will plan to meet the director at his office.\nOptions:\n(A) It will be Bailey's office\n(B) It will be the director's office\n(C) Ambiguous", "target": "Let's think step by step.\nHere we need to determine who \"his\" might refer to. \"His\" could refer to Bailey or the director. Without more context, we cannot determine whose office is referred to. So the answer is (C)."},
    ],
    "dyck_languages": [
        {"input": "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ { [", "target": "Let's think step by step.\nWe need to close the brackets in the reverse order they were opened. The opening sequence is [ { [, so we need to close with ] } ]. So the answer is ] } ]."},
        {"input": "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: < > ( ( [ [ ( { } ) [ < > ] ]", "target": "Let's think step by step.\nWe need to match up the brackets. Looking at the sequence: < > are closed. Then ( ( opens two parentheses. [ [ opens two brackets. ( { } ) opens and closes a parenthesis with braces inside. [ < > ] opens and closes a bracket with angle brackets inside. So we still have [ [ ( ( open. We need to close with ] ] ) ). So the answer is ] ] ) )."},
        {"input": "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: < [ < [ { < [ ] < { } > > } ] > { { ( ) } { < [ < > ] > } }", "target": "Let's think step by step.\nLooking at the unclosed brackets from the input: < [ < [ { are opened. Then < [ ] < { } > > closes some inner brackets. Then } closes the {. Then ] closes one [. Then > closes the inner <. We need to close the remaining open brackets: [ < which need ] >. So the answer is ] >."},
    ],
    "formal_fallacies": [
        {"input": "\"It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: To begin with, Lesley is a close friend of Fernando. Moreover, being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy. It follows that Lesley is a great-grandfather of Leroy.\"\nIs the argument, given the explicitly stated premises, deductively valid or invalid?\nOptions:\n- valid\n- invalid", "target": "Let's think step by step.\n(1) Lesley is a close friend of Fernando.\n(2) Being a close friend of Fernando OR being a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy.\nFrom premise (1), Lesley satisfies the condition of being a close friend of Fernando. By premise (2), this is sufficient for being a great-grandfather of Leroy. The argument is valid. So the answer is valid."},
        {"input": "\"It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: Whoever is not a great-grandfather of Clyde is a stepbrother of Brian. Being an ancestor of Dana is sufficient for not being a great-grandfather of Clyde. We may conclude: Everyone who is an ancestor of Dana is a stepbrother of Brian, too.\"\nIs the argument, given the explicitly stated premises, deductively valid or invalid?\nOptions:\n- valid\n- invalid", "target": "Let's think step by step.\n(1) Whoever is not a great-grandfather of Clyde is a stepbrother of Brian.\n(2) Being an ancestor of Dana is sufficient for not being a great-grandfather of Clyde.\nFrom (2), if someone is an ancestor of Dana, they are not a great-grandfather of Clyde. From (1), if they are not a great-grandfather of Clyde, they are a stepbrother of Brian. So by transitivity, if someone is an ancestor of Dana, they are a stepbrother of Brian. So the answer is valid."},
        {"input": "\"It is not always easy to grasp who is consuming which combos. The following argument pertains to this question: Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both. No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and, simultaneously, a loyal buyer of Caress soap. It follows that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap.\"\nIs the argument, given the explicitly stated premises, deductively valid or invalid?\nOptions:\n- valid\n- invalid", "target": "Let's think step by step.\n(1) Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap (or both).\n(2) No regular consumer of Lush soap is BOTH a rare consumer of Nioxin shampoo AND a loyal buyer of Caress soap.\nThe argument concludes that no infrequent user of Paul Mitchell shampoo is a regular consumer of Lush soap. However, premise (1) says they are one OR the other OR both, and premise (2) only excludes those who are BOTH. An infrequent user of Paul Mitchell shampoo could be just a rare consumer of Nioxin shampoo (not both), and still be a regular consumer of Lush soap. So the answer is invalid."},
    ],
    "geometric_shapes": [
        {"input": "This SVG path element <path d=\"M 31.00,73.00 L 32.00,59.00 L 44.00,50.00 L 49.00,41.00 L 64.00,37.00 L 71.00,55.00 L 64.00,76.00 L 52.00,77.00 L 31.00,73.00\"/> draws a\nOptions:\n(A) circle\n(B) heptagon\n(C) hexagon\n(D) kite\n(E) line\n(F) octagon\n(G) pentagon\n(H) rectangle\n(I) sector\n(J) triangle", "target": "Let's think step by step.\nThis SVG path starts with M 31.00,73.00 L 32.00,59.00 which means it moves to (31, 73) and draws a line to (32, 59). Then it continues with L 44.00,50.00 L 49.00,41.00 L 64.00,37.00 L 71.00,55.00 L 64.00,76.00 L 52.00,77.00 L 31.00,73.00. Counting the vertices: (31,73), (32,59), (44,50), (49,41), (64,37), (71,55), (64,76), (52,77), back to (31,73). That's 8 distinct vertices, making it an octagon. So the answer is (F)."},
        {"input": "This SVG path element <path d=\"M 14.19,26.04 L 51.43,39.21 L 58.44,36.69 L 56.63,30.17 L 48.53,26.66 L 14.19,26.04\"/> draws a\nOptions:\n(A) circle\n(B) heptagon\n(C) hexagon\n(D) kite\n(E) line\n(F) octagon\n(G) pentagon\n(H) rectangle\n(I) sector\n(J) triangle", "target": "Let's think step by step.\nLooking at the path: M 14.19,26.04 starts at (14.19, 26.04). Then L commands draw lines to (51.43, 39.21), (58.44, 36.69), (56.63, 30.17), (48.53, 26.66), and back to (14.19, 26.04). Counting distinct vertices: 5 points forming a closed shape. So the answer is (G)."},
        {"input": "This SVG path element <path d=\"M 41.00,43.00 L 42.00,16.00 L 72.00,14.00 L 41.00,43.00\"/> draws a\nOptions:\n(A) circle\n(B) heptagon\n(C) hexagon\n(D) kite\n(E) line\n(F) octagon\n(G) pentagon\n(H) rectangle\n(I) sector\n(J) triangle", "target": "Let's think step by step.\nThe path starts at (41, 43), draws lines to (42, 16), then to (72, 14), then back to (41, 43). That's 3 vertices forming a closed shape. So the answer is (J)."},
    ],
    "hyperbaton": [
        {"input": "Which sentence has the correct adjective order:\nOptions:\n(A) rubber terrible ship\n(B) terrible rubber ship", "target": "Let's think step by step.\nWhen there are multiple adjectives before a noun, they typically follow this order: opinion, size, age, shape, color, origin, material, purpose. \"Terrible\" is an opinion adjective. \"Rubber\" is a material adjective. Opinion comes before material. So the answer is (B)."},
        {"input": "Which sentence has the correct adjective order:\nOptions:\n(A) repulsive small Brazilian exercise ship\n(B) Brazilian repulsive exercise small ship", "target": "Let's think step by step.\nThe correct order is: opinion (repulsive), size (small), origin (Brazilian), purpose (exercise), then the noun (ship). So the answer is (A)."},
        {"input": "Which sentence has the correct adjective order:\nOptions:\n(A) blue gold wonderful square shoe\n(B) wonderful square blue gold shoe", "target": "Let's think step by step.\nThe correct order is: opinion (wonderful), shape (square), color (blue), material (gold), then the noun (shoe). So the answer is (B)."},
    ],
    "logical_deduction_five_objects": [
        {"input": "A fruit stand sells five fruits: oranges, watermelons, cantaloupes, kiwis, and grapes. There are the following conditions:\nCantaloupes are more expensive than kiwis.\nOranges are less expensive than grapes.\nWatermelons are less expensive than grapes.\nWatermelons are more expensive than cantaloupes.\nGrapes are the most expensive.\nWhich fruit is the least expensive?\nOptions:\n(A) The oranges\n(B) The watermelons\n(C) The cantaloupes\n(D) The kiwis\n(E) The grapes", "target": "Let's think step by step.\nFrom the conditions:\n- Grapes are most expensive (given).\n- Oranges < grapes, Watermelons < grapes.\n- Watermelons > cantaloupes > kiwis.\nSo we have: grapes > watermelons > cantaloupes > kiwis.\nAnd grapes > oranges.\nBut we don't know exactly where oranges fits. However, the question asks which is LEAST expensive. From watermelons > cantaloupes > kiwis, kiwis is less than all those. Since grapes > oranges and grapes > watermelons > cantaloupes > kiwis, kiwis is definitely the least expensive. So the answer is (D)."},
        {"input": "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a branch, there are five birds: a blue jay, a falcon, a crow, an owl, and a robin. The falcon is to the left of the owl. The blue jay is the second from the left. The crow is the third from the left. The robin is to the right of the owl.\nWhich of the following is true?\nOptions:\n(A) The blue jay is the second from the left.\n(B) The falcon is the second from the left.\n(C) The crow is the second from the left.\n(D) The owl is the second from the left.\n(E) The robin is the second from the left.", "target": "Let's think step by step.\nGiven: Blue jay is second from left. Crow is third from left.\nSo positions 2 and 3 are taken by blue jay and crow.\nFalcon is left of owl, and robin is right of owl.\nThe remaining positions are 1, 4, 5 for falcon, owl, robin.\nSince falcon < owl < robin in position, and positions 1, 4, 5 are available:\nFalcon at 1, owl at 4, robin at 5.\nSo the order is: falcon, blue jay, crow, owl, robin.\nBlue jay is indeed second from left. So the answer is (A)."},
        {"input": "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nIn a golf tournament, there were five golfers: Joe, Eve, Mya, Amy, and Rob. Mya finished third. Rob finished second. Amy finished first. Joe finished below Eve.\nWhich of the following is true?\nOptions:\n(A) Joe finished first.\n(B) Eve finished first.\n(C) Mya finished first.\n(D) Amy finished first.\n(E) Rob finished first.", "target": "Let's think step by step.\nGiven: Amy finished first. Rob finished second. Mya finished third.\nSo positions 1, 2, 3 are Amy, Rob, Mya.\nJoe finished below Eve, so Eve has a better position than Joe.\nPositions 4 and 5 are left for Eve and Joe, with Eve finishing higher.\nSo Eve is 4th and Joe is 5th.\nAmy finished first. So the answer is (D)."},
    ],
    "logical_deduction_seven_objects": [
        {"input": "The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nIn a golf tournament, there were seven golfers: Ana, Eve, Ada, Dan, Rob, Amy, and Joe. Amy finished third. Dan finished above Amy. Ada finished last. Eve finished below Amy. Rob finished below Ana. Ana finished second.\nWhich of the following is true?\nOptions:\n(A) Ana finished third.\n(B) Eve finished third.\n(C) Ada finished third.\n(D) Dan finished third.\n(E) Rob finished third.\n(F) Amy finished third.\n(G) Joe finished third.", "target": "Let's think step by step.\nGiven: Amy finished third. Ana finished second. Ada finished last (7th).\nDan finished above Amy (so Dan is 1st or 2nd). Since Ana is 2nd, Dan must be 1st.\nEve finished below Amy (so Eve is 4th, 5th, 6th, or 7th). Since Ada is 7th, Eve is 4th, 5th, or 6th.\nRob finished below Ana (so Rob is 3rd or below). Since Amy is 3rd, Rob is 4th, 5th, 6th, or 7th.\nRemaining positions: 4, 5, 6 for Eve, Rob, Joe.\nAmy finished third. So the answer is (F)."},
        {"input": "The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are seven books: a red book, a gray book, a white book, a blue book, a green book, an orange book, and a purple book. The gray book is the third from the left. The white book is to the right of the gray book. The purple book is the fourth from the left. The red book is the rightmost. The green book is to the left of the gray book. The blue book is to the right of the purple book.\nWhich of the following is true?\nOptions:\n(A) The red book is the fourth from the left.\n(B) The gray book is the fourth from the left.\n(C) The white book is the fourth from the left.\n(D) The blue book is the fourth from the left.\n(E) The green book is the fourth from the left.\n(F) The orange book is the fourth from the left.\n(G) The purple book is the fourth from the left.", "target": "Let's think step by step.\nGiven: Gray is 3rd. Purple is 4th. Red is rightmost (7th).\nGreen is left of gray (so green is 1st or 2nd).\nWhite is right of gray (so white is 4th, 5th, 6th, or 7th). Since purple is 4th and red is 7th, white is 5th or 6th.\nBlue is right of purple (so blue is 5th, 6th, or 7th). Since red is 7th, blue is 5th or 6th.\nOrange is remaining. Positions: green and orange at 1, 2. White and blue at 5, 6.\nThe purple book is 4th. So the answer is (G)."},
        {"input": "The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nA fruit stand sells seven fruits: plums, pears, apples, oranges, loquats, kiwis, and watermelons. The apples are more expensive than the pears. The kiwis are less expensive than the oranges. The oranges are the second-most expensive. The watermelons are less expensive than the plums. The plums are less expensive than the pears. The loquats are the third-most expensive.\nWhich of the following is true?\nOptions:\n(A) The plums are the second-cheapest.\n(B) The pears are the second-cheapest.\n(C) The apples are the second-cheapest.\n(D) The oranges are the second-cheapest.\n(E) The loquats are the second-cheapest.\n(F) The kiwis are the second-cheapest.\n(G) The watermelons are the second-cheapest.", "target": "Let's think step by step.\nFrom the conditions:\n- Oranges are 2nd most expensive. Loquats are 3rd most expensive.\n- apples > pears > plums > watermelons (chain of expense).\n- kiwis < oranges.\nSo from most to least expensive, apples must be 1st (since they're above pears which is above plums which is above watermelons). Oranges is 2nd. Loquats is 3rd.\nRemaining positions 4, 5, 6, 7 for pears, plums, kiwis, watermelons.\nSince pears > plums > watermelons: pears is 4th, plums is 5th, watermelons is 7th (cheapest).\nKiwis is 6th (second-cheapest).\nSo watermelons is cheapest, kiwis is second-cheapest. So the answer is (F)."},
    ],
    "logical_deduction_three_objects": [
        {"input": "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nIn a golf tournament, there were three golfers: Amy, Eli, and Eve. Eve finished above Amy. Eli finished below Amy.\nWhich of the following is true?\nOptions:\n(A) Amy finished last.\n(B) Eli finished last.\n(C) Eve finished last.", "target": "Let's think step by step.\nEve finished above Amy, and Eli finished below Amy. So the order from best to worst is: Eve, Amy, Eli. Eli finished last. So the answer is (B)."},
        {"input": "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are three books: a white book, a green book, and an orange book. The green book is to the right of the white book. The orange book is the rightmost.\nWhich of the following is true?\nOptions:\n(A) The white book is the leftmost.\n(B) The green book is the leftmost.\n(C) The orange book is the leftmost.", "target": "Let's think step by step.\nThe green book is to the right of the white book: white < green. The orange book is rightmost: orange is 3rd. So white is 1st and green is 2nd. The white book is the leftmost. So the answer is (A)."},
        {"input": "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are three books: a red book, a gray book, and a white book. The white book is to the left of the gray book. The red book is the second from the left.\nWhich of the following is true?\nOptions:\n(A) The red book is the leftmost.\n(B) The gray book is the leftmost.\n(C) The white book is the leftmost.", "target": "Let's think step by step.\nThe red book is the second from the left. The white book is to the left of the gray book. Since red is 2nd, white and gray are at positions 1 and 3. Since white < gray, white is 1st and gray is 3rd. The white book is the leftmost. So the answer is (C)."},
    ],
    "movie_recommendation": [
        {"input": "Find a movie similar to Star Wars Episode IV - A New Hope, Indiana Jones and the Last Crusade, Star Wars Episode V - The Empire Strikes Back, The Big Lebowski:\nOptions:\n(A) Tetsuo\n(B) the Ironman\n(C) The Princess Bride\n(D) The Barkley Marathons The Race That Eats Its Young\n(E) Bug", "target": "Let's think step by step.\n- Star Wars Episode IV - A New Hope (action, adventure, fantasy; 1977)\n- Indiana Jones and the Last Crusade (action, adventure; 1989)\n- Star Wars Episode V - The Empire Strikes Back (action, adventure, fantasy; 1980)\n- The Big Lebowski (action, drama, comedy; 1998)\nThese are all famous classic American movies produced before 2000. Amongst all the options, the only movie similar to these ones seems to be The Princess Bride (1987). So the answer is (C)."},
        {"input": "Find a movie similar to Twister, The Silence of the Lambs, Independence Day, Braveheart:\nOptions:\n(A) They Shoot Horses\n(B) Don't They\n(C) Forrest Gump\n(D) The Salton Sea\n(E) Extreme Days", "target": "Let's think step by step.\n- Twister (action, adventure, thriller; 1996)\n- The Silence of the Lambs (crime, drama, thriller; 1991)\n- Independence Day (action, science-fiction, drama; 1996)\n- Braveheart (biography, drama, epic; 1995)\nThese are all famous Hollywood movies produced around the 1990s. Amongst all the options, the only movie similar to these ones seems to be Forrest Gump (comedy, drama, romance; 1994). So the answer is (C)."},
        {"input": "Find a movie similar to Minority Report, Total Recall, Inside Out, Forrest Gump:\nOptions:\n(A) Phenomena\n(B) Lilting\n(C) Catwoman\n(D) Edge of Tomorrow", "target": "Let's think step by step.\n- Minority Report (action, crime, mystery; 2002)\n- Total Recall (action, adventure, science-fiction; 2012)\n- Inside Out (animation, family, comedy; 2015)\n- Forrest Gump (comedy, drama, romance; 1994)\nThese are all famous movies produced in the past few decades. Amongst all the options, the only movie similar to these ones seems to be Edge of Tomorrow (action, adventure, crime, mystery; 2014), as it is also a science-fiction movie and features Tom Cruise. So the answer is (D)."},
    ],
    "multistep_arithmetic_two": [
        {"input": "((-5 + 9 * -4 - 0) * (4 + -7 + 0 * -5)) =", "target": "Let's think step by step.\nLet's first compute (-5 + 9 * -4 - 0):\n9 * -4 = -36\n-5 + -36 = -41\n-41 - 0 = -41\nNow let's compute (4 + -7 + 0 * -5):\n0 * -5 = 0\n4 + -7 = -3\n-3 + 0 = -3\nFinally, (-41) * (-3) = 123. So the answer is 123."},
        {"input": "((-9 + -7 * -9 + -9) + (4 * -9 - 8 - -4)) =", "target": "Let's think step by step.\nLet's first compute (-9 + -7 * -9 + -9):\n-7 * -9 = 63\n-9 + 63 = 54\n54 + -9 = 45\nNow let's compute (4 * -9 - 8 - -4):\n4 * -9 = -36\n-36 - 8 = -44\n-44 - -4 = -44 + 4 = -40\nFinally, 45 + -40 = 5. So the answer is 5."},
        {"input": "(6 * (-3 + 7 * 5 + 0)) =", "target": "Let's think step by step.\nLet's first compute (-3 + 7 * 5 + 0):\n7 * 5 = 35\n-3 + 35 = 32\n32 + 0 = 32\nFinally, 6 * 32 = 192. So the answer is 192."},
    ],
    "navigate": [
        {"input": "If you follow these instructions, do you return to the starting point? Turn left. Turn around. Turn left. Take 7 steps. Take 2 steps. Take 4 steps. Take 8 steps.\nOptions:\n- Yes\n- No", "target": "Let's think step by step.\nStarting at the origin (0, 0), facing the positive y-axis.\nTurn left: now facing the negative x-axis.\nTurn around: now facing the positive x-axis.\nTurn left: now facing the positive y-axis.\nTake 7 steps: (0, 7).\nTake 2 steps: (0, 9).\nTake 4 steps: (0, 13).\nTake 8 steps: (0, 21).\nWe are at (0, 21), not at the origin. So the answer is No."},
        {"input": "If you follow these instructions, do you return to the starting point? Turn around. Take 1 step. Take 6 steps. Turn around. Take 6 steps. Take 9 steps. Take 1 step.\nOptions:\n- Yes\n- No", "target": "Let's think step by step.\nStarting at the origin (0, 0), facing the positive y-axis.\nTurn around: now facing the negative y-axis.\nTake 1 step: (0, -1).\nTake 6 steps: (0, -7).\nTurn around: now facing the positive y-axis.\nTake 6 steps: (0, -1).\nTake 9 steps: (0, 8).\nTake 1 step: (0, 9).\nWe are at (0, 9), not at the origin. So the answer is No."},
        {"input": "If you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps right. Take 9 steps left. Take 7 steps right.\nOptions:\n- Yes\n- No", "target": "Let's think step by step.\nStarting at the origin (0, 0), facing the positive y-axis.\nAlways face forward means we don't change direction.\nTake 2 steps right: (2, 0).\nTake 9 steps left: (2 - 9, 0) = (-7, 0).\nTake 7 steps right: (-7 + 7, 0) = (0, 0).\nWe are at (0, 0), which is the origin. So the answer is Yes."},
    ],
    "object_counting": [
        {"input": "I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?", "target": "Let's think step by step.\nLet me list the items: blackberry, clarinet, nectarine, plum, strawberry, banana, flute, orange, violin.\nNow I'll identify which are fruits:\n- blackberry: fruit\n- clarinet: musical instrument\n- nectarine: fruit\n- plum: fruit\n- strawberry: fruit\n- banana: fruit\n- flute: musical instrument\n- orange: fruit\n- violin: musical instrument\nThe fruits are: blackberry, nectarine, plum, strawberry, banana, orange. That's 6 fruits. So the answer is 6."},
        {"input": "I have an orange, a raspberry, two peaches, a blackberry, an apple, a grape, a nectarine, and three plums. How many fruits do I have?", "target": "Let's think step by step.\nLet me count: orange (1), raspberry (1), peaches (2), blackberry (1), apple (1), grape (1), nectarine (1), plums (3).\n1 + 1 + 2 + 1 + 1 + 1 + 1 + 3 = 11 fruits. So the answer is 11."},
        {"input": "I have a lettuce head, a head of broccoli, an onion, a stalk of celery, two carrots, a garlic, and a yam. How many vegetables do I have?", "target": "Let's think step by step.\nLet me list and count: lettuce head (1), broccoli (1), onion (1), celery (1), carrots (2), garlic (1), yam (1).\nAll of these are vegetables.\n1 + 1 + 1 + 1 + 2 + 1 + 1 = 8 vegetables. So the answer is 8."},
    ],
    "penguins_in_a_table": [
        {"input": "Here is a table where the first line is a header and each subsequent line is a penguin:\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\nFor example: the age of Louis is 7, the move of Bernard is 5.\n\nWhat is the name of the last penguin sorted by alphabetic order?\nOptions:\n(A) Louis\n(B) Bernard\n(C) Vincent\n(D) Gwen", "target": "Let's think step by step.\nThe penguins in alphabetical order are: Bernard, Gwen, Louis, Vincent. The last penguin in alphabetical order is Vincent. So the answer is (C)."},
        {"input": "Here is a table where the first line is a header and each subsequent line is a penguin:\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\nFor example: the age of Louis is 7, the move of Bernard is 5.\n\nWhat is the name of the penguin that is 9 years old?\nOptions:\n(A) Louis\n(B) Bernard\n(C) Vincent\n(D) Gwen", "target": "Let's think step by step.\nLooking at the ages: Louis is 7, Bernard is 5, Vincent is 9, Gwen is 8. The penguin that is 9 years old is Vincent. So the answer is (C)."},
        {"input": "Here is a table where the first line is a header and each subsequent line is a penguin:\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\nFor example: the age of Louis is 7, the move of Bernard is 5.\n\nWhat is the average height of the penguins?\nOptions:\n(A) 65\n(B) 60\n(C) 55\n(D) 75", "target": "Let's think step by step.\nThe heights are: 50, 80, 60, 70. The average is (50 + 80 + 60 + 70) / 4 = 260 / 4 = 65. So the answer is (A)."},
    ],
    "reasoning_about_colored_objects": [
        {"input": "On the nightstand, there is a red pencil, a purple mug, a burgundy keychain, a fuchsia teddy bear, a black plate, and a blue stress ball. What color is the stress ball?\nOptions:\n(A) red\n(B) orange\n(C) yellow\n(D) green\n(E) blue\n(F) brown\n(G) magenta\n(H) fuchsia\n(I) mauve\n(J) teal\n(K) turquoise\n(L) burgundy\n(M) silver\n(N) gold\n(O) black\n(P) grey\n(Q) purple\n(R) pink", "target": "Let's think step by step.\nLooking at the description: \"a blue stress ball.\" The stress ball is blue. So the answer is (E)."},
        {"input": "On the table, you see a bunch of objects arranged in a row: a purple paperclip, a pink stress ball, a brown keychain, a green scrunchiephone charger, a mauve fidget spinner, and a burgundy pen. What is the color of the object directly to the right of the stress ball?\nOptions:\n(A) red\n(B) orange\n(C) yellow\n(D) green\n(E) blue\n(F) brown\n(G) magenta\n(H) fuchsia\n(I) mauve\n(J) teal\n(K) turquoise\n(L) burgundy\n(M) silver\n(N) gold\n(O) black\n(P) grey\n(Q) purple\n(R) pink", "target": "Let's think step by step.\nThe objects in order are: purple paperclip, pink stress ball, brown keychain, green scrunchiephone charger, mauve fidget spinner, burgundy pen. The object to the right of the stress ball (pink stress ball) is the brown keychain. So the answer is (F)."},
        {"input": "On the nightstand, you see the following items arranged in a row: a teal plate, a burgundy keychain, a yellow scrunchiephone charger, an orange mug, a pink notebook, and a grey cup. How many non-orange items do you see to the left of the teal item?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three\n(E) four\n(F) five\n(G) six", "target": "Let's think step by step.\nThe teal plate is the first item (leftmost). There are no items to the left of the teal item. So the answer is (A)."},
    ],
    "ruin_names": [
        {"input": "Which of the following is a humorous edit of this artist or movie name: 'whitesnake'?\nOptions:\n(A) whitesnape\n(B) whitesnaike\n(C) whitesnaker\n(D) whitestrake", "target": "Let's think step by step.\nThe original is 'whitesnake'. Let's look at each option:\n(A) whitesnape - changes 'snake' to 'snape' (like Severus Snape from Harry Potter), creating a pun.\n(B) whitesnaike - just adds an 'i', not particularly funny.\n(C) whitesnaker - adds 'r', not a recognizable pun.\n(D) whitestrake - changes letters but doesn't create a clear pun.\nThe humorous edit is (A) because it references Snape from Harry Potter. So the answer is (A)."},
        {"input": "Which of the following is a humorous edit of this artist or movie name: 'the beatles'?\nOptions:\n(A) the beatles\n(B) the beadles\n(C) the beatle\n(D) tce beatles", "target": "Let's think step by step.\nThe original is 'the beatles'. Let's look at each option:\n(A) the beatles - no change, not an edit.\n(B) the beadles - changes 't' to 'd', making it 'beadles' which sounds like 'beagles' (a dog breed), creating a pun.\n(C) the beatle - just removes 's', not particularly funny.\n(D) tce beatles - typo, not a pun.\nThe humorous edit is (B) because it creates a pun with 'beagles'. So the answer is (B)."},
        {"input": "Which of the following is a humorous edit of this artist or movie name: 'rain man'?\nOptions:\n(A) ruin man\n(B) rain men\n(C) rain ban\n(D) rain mat", "target": "Let's think step by step.\nThe original is 'rain man'. Let's look at each option:\n(A) ruin man - changes 'rain' to 'ruin', which could be a play on the word.\n(B) rain men - just pluralizes 'man', not particularly funny.\n(C) rain ban - changes 'man' to 'ban', making 'rain ban' which sounds like 'ray ban' (the sunglasses brand), creating a pun.\n(D) rain mat - changes 'man' to 'mat', not a recognizable pun.\nThe humorous edit is (C) because it creates a pun with Ray-Ban sunglasses. So the answer is (C)."},
    ],
    "salient_translation_error_detection": [
        {"input": "The following translations from German to English contain a single error. Which is the error?\nGerman: Er versuchte, sie zu überzeugen, dass er der richtige Mann für sie sei.\nEnglish: He tried to convince her that he was the wrong man for her.\nOptions:\n(A) Er -> She\n(B) richtige -> wrong\n(C) überzeugen -> argue\n(D) versuchte -> succeeded", "target": "Let's think step by step.\nLet me compare:\n- \"Er\" = \"He\" (correct)\n- \"versuchte\" = \"tried\" (correct)\n- \"überzeugen\" = \"convince\" (correct)\n- \"richtige\" = \"right\" but translated as \"wrong\" (ERROR)\n- \"Mann\" = \"man\" (correct)\nThe error is \"richtige\" (right) was translated as \"wrong\" instead of \"right\". So the answer is (B)."},
        {"input": "The following translations from German to English contain a single error. Which is the error?\nGerman: Sie fragte ihn, ob er morgen Zeit hätte.\nEnglish: She asked him if he had time yesterday.\nOptions:\n(A) morgen -> yesterday\n(B) Sie -> He\n(C) fragte -> answered\n(D) Zeit -> money", "target": "Let's think step by step.\nLet me compare:\n- \"Sie\" = \"She\" (correct)\n- \"fragte\" = \"asked\" (correct)\n- \"ihn\" = \"him\" (correct)\n- \"morgen\" = \"tomorrow\" but translated as \"yesterday\" (ERROR)\n- \"Zeit\" = \"time\" (correct)\nThe error is \"morgen\" (tomorrow) was translated as \"yesterday\" instead of \"tomorrow\". So the answer is (A)."},
        {"input": "The following translations from German to English contain a single error. Which is the error?\nGerman: Der Hund des Nachbarn bellt jeden Morgen.\nEnglish: The neighbor's cat barks every morning.\nOptions:\n(A) Hund -> cat\n(B) des Nachbarn -> the neighbor's\n(C) bellt -> meows\n(D) jeden Morgen -> every evening", "target": "Let's think step by step.\nLet me compare:\n- \"Der Hund\" = \"The dog\" but translated as \"cat\" (ERROR)\n- \"des Nachbarn\" = \"the neighbor's\" (correct)\n- \"bellt\" = \"barks\" (correct, though semantically odd with \"cat\")\n- \"jeden Morgen\" = \"every morning\" (correct)\nThe error is \"Hund\" (dog) was translated as \"cat\" instead of \"dog\". So the answer is (A)."},
    ],
    "snarks": [
        {"input": "Which statement is sarcastic?\nOptions:\n(A) Yes, because having interests and actively researching them is a huge waste\n(B) Yes, because having interests and actively researching them is a huge deal", "target": "Let's think step by step.\nStatement (A) says having interests is \"a huge waste\" which contradicts the fact that having interests is generally positive - this is saying the opposite of what is meant, which is sarcasm.\nStatement (B) says it's \"a huge deal\" which could be genuine enthusiasm.\nThe sarcastic statement is (A). So the answer is (A)."},
        {"input": "Which statement is sarcastic?\nOptions:\n(A) No one is going to disagree with you on this. Avoiding ad hominem attacks really helps your case\n(B) No one is going to disagree with you on this. Ad hominem attacks really help your case", "target": "Let's think step by step.\nStatement (A) says avoiding ad hominem attacks helps - this is genuinely true and good advice.\nStatement (B) says ad hominem attacks help your case - this is the opposite of reality (ad hominem attacks actually hurt your case), so it's saying the opposite of what is meant, which is sarcasm.\nThe sarcastic statement is (B). So the answer is (B)."},
        {"input": "Which statement is sarcastic?\nOptions:\n(A) Consistency in the league's punishments? What do you think this is supposed to be, politics?\n(B) Consistency in the league's punishments? What do you think this is supposed to be, moral?", "target": "Let's think step by step.\nStatement (A) compares inconsistent league punishments to politics, implying politics is known for inconsistency - this is a sarcastic comparison highlighting the irony.\nStatement (B) mentions \"moral\" which doesn't make the same kind of ironic comparison.\nThe sarcastic statement is (A). So the answer is (A)."},
    ],
    "sports_understanding": [
        {"input": "Is the following sentence plausible? \"Bam Adebayo scored a reverse layup in the Western Conference Finals.\"", "target": "Let's think step by step.\nBam Adebayo is an American basketball player. Scoring a reverse layup is part of basketball. The Western Conference Finals is a real NBA event. However, Bam Adebayo plays for the Miami Heat, which is in the Eastern Conference, not the Western Conference. But scoring a reverse layup is still plausible basketball. Actually, players can't play in the Western Conference Finals if they're on an Eastern team. So the answer is no."},
        {"input": "Is the following sentence plausible? \"Santi Cazorla scored a touchdown.\"", "target": "Let's think step by step.\nSanti Cazorla is a soccer player. Touchdown is part of American football and rugby, not soccer. So the answer is no."},
        {"input": "Is the following sentence plausible? \"DeMar DeRozan was called for the goal tend.\"", "target": "Let's think step by step.\nDeMar DeRozan is an American basketball player. Goaltending is part of basketball (it's a violation). So the answer is yes."},
    ],
    "temporal_sequences": [
        {"input": "Today, Emily went to the museum. Between what times could they have gone?\nWe know that:\nEmily woke up at 1pm.\nElizabeth saw Emily reading at the library from 2pm to 4pm.\nJessica saw Emily walking towards the Statue of Liberty from 4pm to 6pm.\nJennifer saw Emily working at the office from 6pm to 7pm.\nThe museum was closed after 7pm.\nBetween what times could Emily have gone to the museum?\nOptions:\n(A) 1pm to 2pm\n(B) 6pm to 7pm\n(C) 2pm to 4pm\n(D) 4pm to 6pm", "target": "Let's think step by step.\nEmily woke up at 1pm. From 2pm to 4pm she was at the library. From 4pm to 6pm she was walking towards the Statue of Liberty. From 6pm to 7pm she was at the office. The museum closes after 7pm.\nThe only free time slot is from 1pm to 2pm (after waking up, before the library). So the answer is (A)."},
        {"input": "Today, William went to the art studio. Between what times could they have gone?\nWe know that:\nWilliam woke up at 5am.\nJohn saw William waiting at the train station from 5am to 6am.\nDavid saw William taking photos near the Eiffel Tower from 6am to 9am.\nSusan saw William watching a movie at the theater from 9am to 5pm.\nSarah saw William waiting at the airport from 5pm to 6pm.\nThe art studio was closed after 8pm.\nBetween what times could William have gone to the art studio?\nOptions:\n(A) 5am to 6am\n(B) 5pm to 6pm\n(C) 6am to 9am\n(D) 6pm to 8pm", "target": "Let's think step by step.\nWilliam woke up at 5am. From 5am to 6am at train station. From 6am to 9am near Eiffel Tower. From 9am to 5pm at theater. From 5pm to 6pm at airport. Studio closes at 8pm.\nThe only free time is from 6pm to 8pm (after the airport, before the studio closes). So the answer is (D)."},
        {"input": "Today, Michael went to the beach. Between what times could they have gone?\nWe know that:\nMichael woke up at 5am.\nMichael's grandmother saw Michael reading at the library from 5am to 8am.\nLisa saw Michael driving to the water park from 8am to 10am.\nEmily saw Michael sitting on a rooftop from 10am to 12pm.\nJennifer saw Michael working at the office from 12pm to 2pm.\nThe beach was closed after 4pm.\nBetween what times could Michael have gone to the beach?\nOptions:\n(A) 8am to 10am\n(B) 2pm to 4pm\n(C) 10am to 12pm\n(D) 5am to 8am", "target": "Let's think step by step.\nMichael woke up at 5am. From 5am to 8am at library. From 8am to 10am driving to water park. From 10am to 12pm on rooftop. From 12pm to 2pm at office. Beach closes at 4pm.\nThe only free time is from 2pm to 4pm (after office, before beach closes). So the answer is (B)."},
    ],
    "tracking_shuffled_objects_five_objects": [
        {"input": "Alice, Bob, Claire, Dave, and Eve are friends and avid readers. At the start of the semester, they each have a different book: Alice has The Pearl, Bob has The Great Gatsby, Claire has Catch-22, Dave has Moby Dick, and Eve has Ulysses.\nAs the semester progresses, they swap books. First, Eve and Alice swap books. Then, Claire and Bob swap books. Then, Eve and Dave swap books. Finally, Claire and Alice swap books.\nAt the end of the semester, Alice has\nOptions:\n(A) The Pearl\n(B) The Great Gatsby\n(C) Catch-22\n(D) Moby Dick\n(E) Ulysses", "target": "Let's think step by step.\n(0) At the start: Alice: The Pearl, Bob: The Great Gatsby, Claire: Catch-22, Dave: Moby Dick, Eve: Ulysses.\n(1) Eve and Alice swap: Alice: Ulysses, Bob: The Great Gatsby, Claire: Catch-22, Dave: Moby Dick, Eve: The Pearl.\n(2) Claire and Bob swap: Alice: Ulysses, Bob: Catch-22, Claire: The Great Gatsby, Dave: Moby Dick, Eve: The Pearl.\n(3) Eve and Dave swap: Alice: Ulysses, Bob: Catch-22, Claire: The Great Gatsby, Dave: The Pearl, Eve: Moby Dick.\n(4) Claire and Alice swap: Alice: The Great Gatsby, Bob: Catch-22, Claire: Ulysses, Dave: The Pearl, Eve: Moby Dick.\nAt the end, Alice has The Great Gatsby. So the answer is (B)."},
        {"input": "Alice, Bob, Claire, Dave, and Eve are playing a card game. At the start of the game, they are each holding a card: Alice has 7 of spades, Bob has king of hearts, Claire has 3 of diamonds, Dave has ace of clubs, and Eve has 2 of hearts.\nAs the game progresses, pairs of players trade cards. First, Alice and Eve trade. Then, Dave and Claire trade. Then, Eve and Claire trade. Finally, Alice and Bob trade.\nAt the end of the game, Claire has\nOptions:\n(A) 7 of spades\n(B) king of hearts\n(C) 3 of diamonds\n(D) ace of clubs\n(E) 2 of hearts", "target": "Let's think step by step.\n(0) At the start: Alice: 7 of spades, Bob: king of hearts, Claire: 3 of diamonds, Dave: ace of clubs, Eve: 2 of hearts.\n(1) Alice and Eve trade: Alice: 2 of hearts, Bob: king of hearts, Claire: 3 of diamonds, Dave: ace of clubs, Eve: 7 of spades.\n(2) Dave and Claire trade: Alice: 2 of hearts, Bob: king of hearts, Claire: ace of clubs, Dave: 3 of diamonds, Eve: 7 of spades.\n(3) Eve and Claire trade: Alice: 2 of hearts, Bob: king of hearts, Claire: 7 of spades, Dave: 3 of diamonds, Eve: ace of clubs.\n(4) Alice and Bob trade: Alice: king of hearts, Bob: 2 of hearts, Claire: 7 of spades, Dave: 3 of diamonds, Eve: ace of clubs.\nAt the end, Claire has 7 of spades. So the answer is (A)."},
        {"input": "Alice, Bob, Claire, Dave, and Eve are on a relay team. At the start of the race, they are each holding a baton: Alice has yellow, Bob has blue, Claire has red, Dave has green, and Eve has orange.\nAs the race progresses, pairs of runners exchange batons. First, Bob and Eve exchange. Then, Alice and Claire exchange. Then, Dave and Bob exchange. Finally, Eve and Alice exchange.\nAt the end of the race, Dave has\nOptions:\n(A) yellow\n(B) blue\n(C) red\n(D) green\n(E) orange", "target": "Let's think step by step.\n(0) At the start: Alice: yellow, Bob: blue, Claire: red, Dave: green, Eve: orange.\n(1) Bob and Eve exchange: Alice: yellow, Bob: orange, Claire: red, Dave: green, Eve: blue.\n(2) Alice and Claire exchange: Alice: red, Bob: orange, Claire: yellow, Dave: green, Eve: blue.\n(3) Dave and Bob exchange: Alice: red, Bob: green, Claire: yellow, Dave: orange, Eve: blue.\n(4) Eve and Alice exchange: Alice: blue, Bob: green, Claire: yellow, Dave: orange, Eve: red.\nAt the end, Dave has orange. So the answer is (E)."},
    ],
    "tracking_shuffled_objects_seven_objects": [
        {"input": "Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are playing a game. At the start of the game, they are each holding a ball: Alice has red, Bob has green, Claire has yellow, Dave has blue, Eve has orange, Fred has white, and Gertrude has purple.\nAs the game progresses, pairs of players swap balls. First, Dave and Fred swap. Then, Alice and Gertrude swap. Then, Bob and Alice swap. Then, Claire and Bob swap. Then, Fred and Eve swap. Finally, Dave and Claire swap.\nAt the end of the game, Dave has\nOptions:\n(A) red\n(B) green\n(C) yellow\n(D) blue\n(E) orange\n(F) white\n(G) purple", "target": "Let's think step by step.\n(0) Start: Alice-red, Bob-green, Claire-yellow, Dave-blue, Eve-orange, Fred-white, Gertrude-purple.\n(1) Dave and Fred swap: Alice-red, Bob-green, Claire-yellow, Dave-white, Eve-orange, Fred-blue, Gertrude-purple.\n(2) Alice and Gertrude swap: Alice-purple, Bob-green, Claire-yellow, Dave-white, Eve-orange, Fred-blue, Gertrude-red.\n(3) Bob and Alice swap: Alice-green, Bob-purple, Claire-yellow, Dave-white, Eve-orange, Fred-blue, Gertrude-red.\n(4) Claire and Bob swap: Alice-green, Bob-yellow, Claire-purple, Dave-white, Eve-orange, Fred-blue, Gertrude-red.\n(5) Fred and Eve swap: Alice-green, Bob-yellow, Claire-purple, Dave-white, Eve-blue, Fred-orange, Gertrude-red.\n(6) Dave and Claire swap: Alice-green, Bob-yellow, Claire-white, Dave-purple, Eve-blue, Fred-orange, Gertrude-red.\nDave has purple. So the answer is (G)."},
        {"input": "Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are dancers. At the start of a song, they are each dancing with a partner: Alice with Sam, Bob with Ophelia, Claire with Rodrigo, Dave with Patrick, Eve with Karl, Fred with Lola, and Gertrude with Jamie.\nThroughout the song, pairs of dancers swap partners. First, Bob and Eve swap. Then, Dave and Eve swap. Then, Alice and Claire swap. Then, Gertrude and Dave swap. Then, Fred and Alice swap. Finally, Claire and Eve swap.\nAt the end of the dance, Eve is dancing with\nOptions:\n(A) Sam\n(B) Ophelia\n(C) Rodrigo\n(D) Patrick\n(E) Karl\n(F) Lola\n(G) Jamie", "target": "Let's think step by step.\n(0) Start: Alice-Sam, Bob-Ophelia, Claire-Rodrigo, Dave-Patrick, Eve-Karl, Fred-Lola, Gertrude-Jamie.\n(1) Bob and Eve swap: Alice-Sam, Bob-Karl, Claire-Rodrigo, Dave-Patrick, Eve-Ophelia, Fred-Lola, Gertrude-Jamie.\n(2) Dave and Eve swap: Alice-Sam, Bob-Karl, Claire-Rodrigo, Dave-Ophelia, Eve-Patrick, Fred-Lola, Gertrude-Jamie.\n(3) Alice and Claire swap: Alice-Rodrigo, Bob-Karl, Claire-Sam, Dave-Ophelia, Eve-Patrick, Fred-Lola, Gertrude-Jamie.\n(4) Gertrude and Dave swap: Alice-Rodrigo, Bob-Karl, Claire-Sam, Dave-Jamie, Eve-Patrick, Fred-Lola, Gertrude-Ophelia.\n(5) Fred and Alice swap: Alice-Lola, Bob-Karl, Claire-Sam, Dave-Jamie, Eve-Patrick, Fred-Rodrigo, Gertrude-Ophelia.\n(6) Claire and Eve swap: Alice-Lola, Bob-Karl, Claire-Patrick, Dave-Jamie, Eve-Sam, Fred-Rodrigo, Gertrude-Ophelia.\nEve is dancing with Sam. So the answer is (A)."},
        {"input": "Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are at a party. At the start of the party, they are each holding a drink: Alice has tea, Bob has water, Claire has coffee, Dave has milk, Eve has juice, Fred has soda, and Gertrude has lemonade.\nAs the party progresses, pairs of people swap drinks. First, Dave and Alice swap. Then, Claire and Eve swap. Then, Bob and Fred swap. Then, Alice and Claire swap. Then, Gertrude and Bob swap. Finally, Eve and Dave swap.\nAt the end of the party, Bob has\nOptions:\n(A) tea\n(B) water\n(C) coffee\n(D) milk\n(E) juice\n(F) soda\n(G) lemonade", "target": "Let's think step by step.\n(0) Start: Alice-tea, Bob-water, Claire-coffee, Dave-milk, Eve-juice, Fred-soda, Gertrude-lemonade.\n(1) Dave and Alice swap: Alice-milk, Bob-water, Claire-coffee, Dave-tea, Eve-juice, Fred-soda, Gertrude-lemonade.\n(2) Claire and Eve swap: Alice-milk, Bob-water, Claire-juice, Dave-tea, Eve-coffee, Fred-soda, Gertrude-lemonade.\n(3) Bob and Fred swap: Alice-milk, Bob-soda, Claire-juice, Dave-tea, Eve-coffee, Fred-water, Gertrude-lemonade.\n(4) Alice and Claire swap: Alice-juice, Bob-soda, Claire-milk, Dave-tea, Eve-coffee, Fred-water, Gertrude-lemonade.\n(5) Gertrude and Bob swap: Alice-juice, Bob-lemonade, Claire-milk, Dave-tea, Eve-coffee, Fred-water, Gertrude-soda.\n(6) Eve and Dave swap: Alice-juice, Bob-lemonade, Claire-milk, Dave-coffee, Eve-tea, Fred-water, Gertrude-soda.\nBob has lemonade. So the answer is (G)."},
    ],
    "tracking_shuffled_objects_three_objects": [
        {"input": "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a yellow ball, Bob has a blue ball, and Claire has a pink ball.\nAs the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Alice and Bob swap balls. Finally, Claire and Bob swap balls.\nAt the end of the game, Bob has the\nOptions:\n(A) yellow ball\n(B) blue ball\n(C) pink ball", "target": "Let's think step by step.\n(0) At the start: Alice: yellow, Bob: blue, Claire: pink.\n(1) Claire and Alice swap balls: Alice: pink, Bob: blue, Claire: yellow.\n(2) Alice and Bob swap balls: Alice: blue, Bob: pink, Claire: yellow.\n(3) Claire and Bob swap balls: Alice: blue, Bob: yellow, Claire: pink.\nAt the end of the game, Bob has the yellow ball. So the answer is (A)."},
        {"input": "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a white ball, Bob has a purple ball, and Claire has a pink ball.\nAs the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Bob and Claire swap balls. Finally, Bob and Alice swap balls.\nAt the end of the game, Alice has the\nOptions:\n(A) white ball\n(B) purple ball\n(C) pink ball", "target": "Let's think step by step.\n(0) At the start: Alice: white, Bob: purple, Claire: pink.\n(1) Bob and Alice swap balls: Alice: purple, Bob: white, Claire: pink.\n(2) Bob and Claire swap balls: Alice: purple, Bob: pink, Claire: white.\n(3) Bob and Alice swap balls: Alice: pink, Bob: purple, Claire: white.\nAt the end of the game, Alice has the pink ball. So the answer is (C)."},
        {"input": "Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Lola, Bob is dancing with Rodrigo, and Claire is dancing with Patrick.\nThroughout the song, the dancers often trade partners. First, Alice and Bob switch partners. Then, Claire and Bob switch partners. Finally, Bob and Alice switch partners.\nAt the end of the dance, Alice is dancing with\nOptions:\n(A) Lola\n(B) Rodrigo\n(C) Patrick", "target": "Let's think step by step.\n(0) At the start: Alice: Lola, Bob: Rodrigo, Claire: Patrick.\n(1) Alice and Bob switch partners: Alice: Rodrigo, Bob: Lola, Claire: Patrick.\n(2) Claire and Bob switch partners: Alice: Rodrigo, Bob: Patrick, Claire: Lola.\n(3) Bob and Alice switch partners: Alice: Patrick, Bob: Rodrigo, Claire: Lola.\nAt the end of the dance, Alice is dancing with Patrick. So the answer is (C)."},
    ],
    "web_of_lies": [
        {"input": "In this question, assume each person either always tells the truth or always lies. Fidel says Millicent lies. Millicent says Jerry lies. Jerry says Delbert lies. Delbert says Jerry lies. Does Fidel tell the truth?", "target": "Let's think step by step.\nLet's assume Fidel tells the truth. Then Millicent lies. If Millicent lies, her statement \"Jerry lies\" is false, so Jerry tells the truth. If Jerry tells the truth, his statement \"Delbert lies\" is true, so Delbert lies. If Delbert lies, his statement \"Jerry lies\" is false, so Jerry tells the truth. This is consistent. So the answer is Yes."},
        {"input": "In this question, assume each person either always tells the truth or always lies. Kristian tells the truth. Millie says Kristian lies. Maybelle says Millie tells the truth. Fidel says Maybelle tells the truth. Leda says Fidel lies. Does Leda tell the truth?", "target": "Let's think step by step.\nWe know Kristian tells the truth. Millie says \"Kristian lies\" which is false, so Millie lies. Maybelle says \"Millie tells the truth\" which is false (since Millie lies), so Maybelle lies. Fidel says \"Maybelle tells the truth\" which is false, so Fidel lies. Leda says \"Fidel lies\" which is true, so Leda tells the truth. So the answer is Yes."},
        {"input": "In this question, assume each person either always tells the truth or always lies. Kristian tells the truth. Michaela says Kristian lies. Raymond says Michaela tells the truth. Osvaldo says Raymond tells the truth. Jamey says Osvaldo tells the truth. Does Jamey tell the truth?", "target": "Let's think step by step.\nWe know Kristian tells the truth. Michaela says \"Kristian lies\" which is false, so Michaela lies. Raymond says \"Michaela tells the truth\" which is false, so Raymond lies. Osvaldo says \"Raymond tells the truth\" which is false, so Osvaldo lies. Jamey says \"Osvaldo tells the truth\" which is false, so Jamey lies. So the answer is No."},
    ],
    "word_sorting": [
        {"input": "Sort the following words alphabetically: List: hello world apple zebra", "target": "Let's think step by step.\nThe words are: hello, world, apple, zebra.\nSorting alphabetically: apple comes before hello, hello comes before world, world comes before zebra.\nSo the sorted list is: apple hello world zebra. So the answer is apple hello world zebra."},
        {"input": "Sort the following words alphabetically: List: programming python java ruby", "target": "Let's think step by step.\nThe words are: programming, python, java, ruby.\nSorting alphabetically: java comes before programming, programming comes before python, python comes before ruby.\nSo the sorted list is: java programming python ruby. So the answer is java programming python ruby."},
        {"input": "Sort the following words alphabetically: List: car bicycle train airplane", "target": "Let's think step by step.\nThe words are: car, bicycle, train, airplane.\nSorting alphabetically: airplane comes before bicycle, bicycle comes before car, car comes before train.\nSo the sorted list is: airplane bicycle car train. So the answer is airplane bicycle car train."},
    ],
}
# fmt: on

# MMLU用few-shot examples (5-shot)
# lm-eval-harness公式: output_type=multiple_choice（ログ確率比較）
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml
# プロンプト形式:
# {{question.strip()}}
# A. {{choices[0]}}
# B. {{choices[1]}}
# C. {{choices[2]}}
# D. {{choices[3]}}
# Answer:
# ログ確率方式ではfew-shotは使用しない（doc_to_choice: ["A", "B", "C", "D"]）
MMLU_FEW_SHOT_EXAMPLES: list[dict[str, str]] = []  # ログ確率方式では不要

# ベンチマーク別few-shot examples
FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "gsm8k": GSM8K_FEW_SHOT_EXAMPLES,
    "bbh": BBH_FEW_SHOT_EXAMPLES,
    "mmlu": MMLU_FEW_SHOT_EXAMPLES,
}

# ptモデル用プロンプトテンプレート（チャットテンプレートなし）
# lm-eval-harness公式形式に準拠
PT_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "gsm8k": {
        # 公式: doc_to_text: "Q: {{question}}\n\nA:"
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
        "prefix": "",
        "question_template": "Q: {question}\n\nA:",
        "answer_template": " {target}\n\n",
    },
    "bbh": {
        # 公式: doc_to_text: "Q: {{input}}\nA: Let's think step by step.\n"
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_fewshot/boolean_expressions.yaml
        "prefix": "",
        "question_template": "Q: {question}\nA: Let's think step by step.\n",
        "answer_template": "{target}\n\n",
    },
    "mmlu": {
        # 公式: output_type: multiple_choice (ログ確率比較)
        # doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\n...\nAnswer:"
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml
        "prefix": "",
        "question_template": "{question}\nAnswer:",
        "answer_template": " {target}\n\n",
    },
}

# itモデル用プロンプトテンプレート（チャットテンプレート用）
# ITモデルでもlm-eval-harnessの公式プロンプト形式を尊重
IT_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "gsm8k": {
        # GSM8K CoT形式（lm-eval-harness公式準拠）
        # 公式: doc_to_text: "Q: {{question}}\n\nA:"
        # 公式strict-match: "The answer is (\-?[0-9\.\,]+)."
        # 回答形式を明示的に指示して "The answer is X." 形式で回答させる
        "system": (
            "You are a helpful assistant that solves math problems step by step. "
            "You must always end your response with 'The answer is X.' "
            "where X is the final numerical answer."
        ),
        "user": "Q: {question}\n\nA:",
    },
    "bbh": {
        # BBH CoT形式（lm-eval-harness公式準拠）
        # 公式: doc_to_text: "Q: {{input}}\nA: Let's think step by step.\n"
        # 公式regex: "(?<=the answer is )(.*)(?=.)"
        # 回答形式を明示的に指示して "the answer is X." 形式で回答させる
        "system": (
            "You are a helpful assistant that thinks step by step. "
            "You must always end your response with 'So the answer is X.' "
            "where X is your final answer."
        ),
        "user": "Q: {question}\nA: Let's think step by step.\n",
    },
    "mmlu": {
        # MMLU（テキスト生成評価用）
        # 公式はログ確率方式（output_type: multiple_choice）
        # テキスト生成の場合は "The answer is X." 形式で回答させる
        "system": (
            "Answer the following multiple choice question. "
            "You must always end your response with 'The answer is X.' "
            "where X is the letter (A, B, C, or D) of your chosen answer."
        ),
        "user": "{question}\nAnswer:",
    },
    # 日本語ベンチマーク
    "jamp": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "jnli": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "niilc": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "jsquad": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "jcommonsenseqa": {
        "system": "以下の質問に答えてください。選択肢から最も適切なものを選んでください。",
        "user": "{question}\n\n回答:",
    },
}

# 後方互換性のためのエイリアス
PROMPT_TEMPLATES = IT_PROMPT_TEMPLATES


@dataclass
class InferenceResult:
    """推論結果を保持するデータクラス."""

    example_id: int
    original_text: str
    generated_text: str
    expected_answer: str
    is_perturbed: bool = False
    perturbed_text: str | None = None
    prompt: str | None = None  # 使用したプロンプト（デバッグ用）


def create_pt_prompt(
    benchmark_name: str,
    question: str,
    subtask: str | None = None,
) -> str:
    """ptモデル用のプロンプトを生成（チャットテンプレートなし）.

    few-shot examplesを含むテキストプロンプトを返す.
    lm-eval-harness公式形式に準拠.

    Args:
        benchmark_name: ベンチマーク名
        question: 質問文
        subtask: サブタスク名（BBHなど、サブタスク別few-shot examplesを使用する場合）

    Returns:
        テキストプロンプト
    """
    template = PT_PROMPT_TEMPLATES.get(benchmark_name)
    if template is None:
        # デフォルトテンプレート
        template = {
            "prefix": "",
            "question_template": "Q: {question}\nA:",
            "answer_template": " {target}\n\n",
        }

    # few-shot examplesを取得
    # BBHの場合はサブタスク別のfew-shot examplesを使用
    if benchmark_name == "bbh" and subtask and subtask in BBH_SUBTASK_FEW_SHOT_EXAMPLES:
        few_shot_examples = BBH_SUBTASK_FEW_SHOT_EXAMPLES[subtask]
    else:
        few_shot_examples = FEW_SHOT_EXAMPLES.get(benchmark_name, [])

    num_shots = BENCHMARK_SHOTS.get(benchmark_name, 0)

    prompt_parts = []

    # プレフィックス（あれば）
    if template["prefix"]:
        prompt_parts.append(template["prefix"])

    # few-shot examples
    for example in few_shot_examples[:num_shots]:
        # BBHは "input"/"target"、GSM8Kは "question"/"target" フィールドを使用
        example_question = example.get("input") or example.get("question", "")
        example_target = example["target"]

        q_part = template["question_template"].format(question=example_question)
        a_part = template["answer_template"].format(target=example_target)
        prompt_parts.append(q_part + a_part)

    # 実際の質問
    final_question = template["question_template"].format(question=question)
    prompt_parts.append(final_question)

    return "".join(prompt_parts)


def create_it_messages(
    benchmark_name: str,
    question: str,
) -> list[dict[str, str]]:
    """itモデル用のメッセージリストを生成（0-shot）.

    チャットテンプレート用の構造化されたメッセージを返す.
    ITモデルは0-shotで推論するため、few-shot examplesは含めない.

    Args:
        benchmark_name: ベンチマーク名
        question: 質問文

    Returns:
        メッセージリスト [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    template = IT_PROMPT_TEMPLATES.get(benchmark_name)
    if template is None:
        # デフォルトテンプレート
        template = {
            "system": "Answer the following question.",
            "user": "{question}\n\nAnswer:",
        }

    # ITモデルは0-shot（few-shot examplesなし）
    messages: list[dict[str, str]] = [
        {"role": "system", "content": template["system"]},
    ]

    # 質問を追加
    user_content = template["user"].format(question=question)
    messages.append({"role": "user", "content": user_content})

    return messages


def create_prompt(
    benchmark_name: str,
    question: str,
    use_chat_format: bool = True,
    subtask: str | None = None,
) -> str | list[dict[str, str]]:
    """ベンチマークに応じたプロンプトを生成.

    Args:
        benchmark_name: ベンチマーク名
        question: 質問文
        use_chat_format: チャット形式（メッセージリスト）を使用するか
            True: itモデル用メッセージリスト
            False: ptモデル用テキストプロンプト
        subtask: サブタスク名（BBHなど、サブタスク別few-shot examplesを使用する場合）

    Returns:
        use_chat_format=True: メッセージリスト（チャットテンプレート用）
        use_chat_format=False: 文字列プロンプト（ptモデル用）
    """
    if use_chat_format:
        return create_it_messages(benchmark_name, question)
    else:
        return create_pt_prompt(benchmark_name, question, subtask=subtask)


# 後方互換性のためのエイリアス
def create_messages(
    benchmark_name: str,
    question: str,
) -> list[dict[str, str]]:
    """後方互換性のためのエイリアス."""
    return create_it_messages(benchmark_name, question)


def run_inference(
    model: BaseModel,
    examples: list[dict[str, Any]],
    benchmark_name: str,
    text_field: str = "question",
    batch_size: int = 8,
    generation_config: GenerationConfig | None = None,
    use_batch: bool = True,
    use_chat_format: bool = True,
    save_prompts: bool = False,
    subtask_field: str | None = "subtask",
) -> list[InferenceResult]:
    """ベンチマークデータに対して推論を実行.

    Args:
        model: 推論に使用するモデル
        examples: サンプルデータのリスト
        benchmark_name: ベンチマーク名
        text_field: テキストフィールド名
        batch_size: バッチサイズ
        generation_config: 生成設定
        use_batch: バッチ処理を使用するか
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか
        subtask_field: サブタスクフィールド名（BBHなど、サブタスク別few-shot使用時）

    Returns:
        推論結果のリスト
    """
    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=512,  # CoT推論に十分な長さ
            temperature=0.0,
            do_sample=False,
        )

    # プロンプト生成
    prompts = []
    for example in examples:
        question = example.get(text_field, "")
        subtask = example.get(subtask_field) if subtask_field else None
        prompt = create_prompt(
            benchmark_name, question, use_chat_format=use_chat_format, subtask=subtask
        )
        prompts.append(prompt)

    logger.info(f"推論開始: {len(prompts)}件のサンプル")
    logger.info(
        f"チャット形式: {use_chat_format}, few-shot: {BENCHMARK_SHOTS.get(benchmark_name, 0)}-shot"
    )

    # 推論実行
    if use_batch and hasattr(model, "generate_batch"):
        generated_texts = model.generate_batch(
            prompts,
            config=generation_config,
            batch_size=batch_size,
        )
    else:
        generated_texts = model.generate(prompts, config=generation_config)

    # 結果を整理
    results = []
    for i, (example, generated) in enumerate(zip(examples, generated_texts, strict=True)):
        prompt_str = None
        if save_prompts:
            # プロンプトを文字列として保存
            p = prompts[i]
            if isinstance(p, list):
                # メッセージリストの場合は整形
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in p)
            else:
                prompt_str = p

        result = InferenceResult(
            example_id=example.get("id", i),
            original_text=example.get(text_field, ""),
            generated_text=generated,
            expected_answer=str(example.get("answer", "")),
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"推論完了: {len(results)}件の結果")
    return results


@dataclass
class MMLUInferenceResult:
    """MMLU推論結果（ログ確率ベース）を保持するデータクラス."""

    example_id: int
    original_text: str
    predicted_answer: str  # ログ確率最大の選択肢
    expected_answer: str
    choice_logprobs: list[float]  # 各選択肢のログ確率
    choices: list[str]  # 選択肢リスト（A, B, C, D）
    is_perturbed: bool = False
    perturbed_text: str | None = None
    prompt: str | None = None


def run_inference_mmlu(
    model: BaseModel,
    examples: list[dict[str, Any]],
    text_field: str = "question",
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[MMLUInferenceResult]:
    """MMLUベンチマークに対してログ確率ベースで推論を実行.

    lm-eval-harness方式: output_type=multiple_choice（選択肢のログ確率比較）

    Args:
        model: 推論に使用するモデル
        examples: サンプルデータのリスト
        text_field: テキストフィールド名
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    # モデルがログ確率計算をサポートしているか確認
    if not hasattr(model, "compute_choice_logprobs"):
        raise ValueError(
            f"モデル {model.model_name} はログ確率計算をサポートしていません。"
            "LocalModelまたはVLLMModelを使用してください。"
        )

    logger.info(f"MMLU推論開始（ログ確率方式）: {len(examples)}件のサンプル")
    logger.info(f"チャット形式: {use_chat_format}")

    # 選択肢リスト
    choices = ["A", "B", "C", "D"]

    results = []
    for i, example in enumerate(examples):
        question = example.get(text_field, "")

        # プロンプト生成
        prompt = create_prompt("mmlu", question, use_chat_format=use_chat_format)

        # ログ確率計算
        logprobs = model.compute_choice_logprobs(prompt, choices)

        # 最大ログ確率の選択肢を予測
        max_idx = logprobs.index(max(logprobs))
        predicted = choices[max_idx]

        prompt_str = None
        if save_prompts:
            if isinstance(prompt, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in prompt)
            else:
                prompt_str = prompt

        result = MMLUInferenceResult(
            example_id=example.get("id", i),
            original_text=question,
            predicted_answer=predicted,
            expected_answer=str(example.get("answer", "")),
            choice_logprobs=logprobs,
            choices=choices,
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"MMLU推論完了: {len(results)}件の結果")
    return results


def run_inference_mmlu_perturbed(
    model: BaseModel,
    perturbed_examples: list[dict[str, Any]],
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[MMLUInferenceResult]:
    """摂動済みMMLUデータに対してログ確率ベースで推論を実行.

    Args:
        model: 推論に使用するモデル
        perturbed_examples: 摂動サンプルデータのリスト
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    # モデルがログ確率計算をサポートしているか確認
    if not hasattr(model, "compute_choice_logprobs"):
        raise ValueError(
            f"モデル {model.model_name} はログ確率計算をサポートしていません。"
            "LocalModelまたはVLLMModelを使用してください。"
        )

    logger.info(f"MMLU摂動データ推論開始（ログ確率方式）: {len(perturbed_examples)}件")

    # 選択肢リスト
    choices = ["A", "B", "C", "D"]

    results = []
    for i, example in enumerate(perturbed_examples):
        perturbed_text = example.get("perturbed_text", "")

        # プロンプト生成
        prompt = create_prompt("mmlu", perturbed_text, use_chat_format=use_chat_format)

        # ログ確率計算
        logprobs = model.compute_choice_logprobs(prompt, choices)

        # 最大ログ確率の選択肢を予測
        max_idx = logprobs.index(max(logprobs))
        predicted = choices[max_idx]

        prompt_str = None
        if save_prompts:
            if isinstance(prompt, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in prompt)
            else:
                prompt_str = prompt

        result = MMLUInferenceResult(
            example_id=example.get("id", i),
            original_text=example.get("original_text", ""),
            predicted_answer=predicted,
            expected_answer=str(example.get("answer", "")),
            choice_logprobs=logprobs,
            choices=choices,
            is_perturbed=True,
            perturbed_text=perturbed_text,
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"MMLU摂動データ推論完了: {len(results)}件の結果")
    return results


def run_inference_on_perturbed(
    model: BaseModel,
    perturbed_examples: list[dict[str, Any]],
    benchmark_name: str,
    batch_size: int = 8,
    generation_config: GenerationConfig | None = None,
    use_batch: bool = True,
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[InferenceResult]:
    """摂動データに対して推論を実行.

    Args:
        model: 推論に使用するモデル
        perturbed_examples: 摂動サンプルデータのリスト
        benchmark_name: ベンチマーク名
        batch_size: バッチサイズ
        generation_config: 生成設定
        use_batch: バッチ処理を使用するか
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=512,  # CoT推論に十分な長さ
            temperature=0.0,
            do_sample=False,
        )

    # 摂動テキストからプロンプト生成
    prompts = []
    for example in perturbed_examples:
        perturbed_text = example.get("perturbed_text", "")
        prompt = create_prompt(benchmark_name, perturbed_text, use_chat_format=use_chat_format)
        prompts.append(prompt)

    logger.info(f"摂動データ推論開始: {len(prompts)}件のサンプル")

    # 推論実行
    if use_batch and hasattr(model, "generate_batch"):
        generated_texts = model.generate_batch(
            prompts,
            config=generation_config,
            batch_size=batch_size,
        )
    else:
        generated_texts = model.generate(prompts, config=generation_config)

    # 結果を整理
    results = []
    for i, (example, generated) in enumerate(zip(perturbed_examples, generated_texts, strict=True)):
        prompt_str = None
        if save_prompts:
            p = prompts[i]
            if isinstance(p, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in p)
            else:
                prompt_str = p

        result = InferenceResult(
            example_id=example.get("id", i),
            original_text=example.get("original_text", ""),
            generated_text=generated,
            expected_answer=str(example.get("answer", "")),
            is_perturbed=True,
            perturbed_text=example.get("perturbed_text", ""),
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"摂動データ推論完了: {len(results)}件の結果")
    return results
