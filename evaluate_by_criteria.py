import pickle

def average_of_numbers(array):
    total = 0
    count = 0

    for criteria in array:
        criteria_list = criteria.split()
        file_structure = float(criteria_list[0])
        semantic_correctness = float(criteria_list[1])
        data_completeness = float(criteria_list[2])
        wording_accuracy = float(criteria_list[3])
        equipment_correctness = float(criteria_list[4])
        sum = 0.2*file_structure + 0.3*semantic_correctness + 0.2*data_completeness + 0.2*wording_accuracy + 0.1*equipment_correctness

        total += sum
        count += 1

    if count == 0:
        return 0

    average = (total / count)
    return round(average, 5)


def criteria_splitting(array):
    total = 0
    count = 0

    results = {
        "file_structure": [],
        "semantic_correctness": [],
        "data_completeness": [],
        "wording_accuracy": [],
        "equipment_correctness": [],
    }

    for criteria in array:
        criteria_list = criteria.split()
        file_structure = float(criteria_list[0])
        semantic_correctness = float(criteria_list[1])
        data_completeness = float(criteria_list[2])
        wording_accuracy = float(criteria_list[3])
        equipment_correctness = float(criteria_list[4])
        sum = (0.2*file_structure + 0.3*semantic_correctness + 0.2*data_completeness +
               0.2*wording_accuracy + 0.1*equipment_correctness)

        results["file_structure"].append(file_structure)
        results["semantic_correctness"].append(semantic_correctness)
        results["data_completeness"].append(data_completeness)
        results["wording_accuracy"].append(wording_accuracy)
        results["equipment_correctness"].append(equipment_correctness)

        total += sum
        count += 1

    return results


metrics = pickle.load(open('metrics_v4.pkl', 'rb'))


results = {
    "Pixtral 12B": {"3": average_of_numbers(metrics["Pixtral 12B"]["3"]),
                    "4": average_of_numbers(metrics["Pixtral 12B"]["4"]),
                    "6": average_of_numbers(metrics["Pixtral 12B"]["6"])},
    "Qwen2.5-VL-72B": {"3": average_of_numbers(metrics["Qwen2.5-VL-72B"]["3"]),
                    "4": average_of_numbers(metrics["Qwen2.5-VL-72B"]["4"]),
                    "6": average_of_numbers(metrics["Qwen2.5-VL-72B"]["6"])},
    "Qwen-VL-Max": {"3": average_of_numbers(metrics["Qwen-VL-Max"]["3"]),
                    "4": average_of_numbers(metrics["Qwen-VL-Max"]["4"]),
                    "6": average_of_numbers(metrics["Qwen-VL-Max"]["6"])},
}

criteria_array = {
    "Pixtral 12B": {"3": criteria_splitting(metrics["Pixtral 12B"]["3"]),
                    "4": criteria_splitting(metrics["Pixtral 12B"]["4"]),
                    "6": criteria_splitting(metrics["Pixtral 12B"]["6"])},
    "Qwen2.5-VL-72B": {"3": criteria_splitting(metrics["Qwen2.5-VL-72B"]["3"]),
                    "4": criteria_splitting(metrics["Qwen2.5-VL-72B"]["4"]),
                    "6": criteria_splitting(metrics["Qwen2.5-VL-72B"]["6"])},
    "Qwen-VL-Max": {"3": criteria_splitting(metrics["Qwen-VL-Max"]["3"]),
                    "4": criteria_splitting(metrics["Qwen-VL-Max"]["4"]),
                    "6": criteria_splitting(metrics["Qwen-VL-Max"]["6"])},
}

print(criteria_array)
print(results)