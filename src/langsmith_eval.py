def evaluate_answers(answer_dict):
    from langsmith import evaluate
    results = []
    # for key, content in answer_dict.items():
    #     if content:
    #         # score = evaluate(content)  # LangSmith değerlendirmesi (pseudo)
    #         results.append({"agent": key, "content": content, "score": score})
    # results.sort(key=lambda x: x["score"], reverse=True)
    return results
