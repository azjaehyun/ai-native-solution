import json

def create_finetune_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    dataset = []
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("####") and "위협 시나리오" in line:
            current_section = "위협 시나리오"
        elif line.startswith("####") and "대응 방안" in line:
            current_section = "대응 방안"
        elif line and current_section:
            if current_section == "위협 시나리오":
                dataset.append({
                    "instruction": "이 상황에서 위협 시나리오는 무엇인가요?",
                    "input": "",
                    "output": line
                })
            elif current_section == "대응 방안":
                dataset.append({
                    "instruction": "이 상황에서 대응 방안은 무엇인가요?",
                    "input": "",
                    "output": line
                })

    with open(output_file, "w", encoding="utf-8") as out_file:
        for entry in dataset:
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 실행
create_finetune_dataset("llm-application-솔루션초안-v4.md", "fine_tune_data_ko.json")