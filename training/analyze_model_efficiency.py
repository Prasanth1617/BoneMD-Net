import pandas as pd

teacher_params = 6105603
student_params = 1108038

reduction = teacher_params - student_params
reduction_percent = reduction / teacher_params * 100
ratio = teacher_params / student_params

print("=" * 70)
print("MODEL EFFICIENCY ANALYSIS")
print("=" * 70)
print(f"Teacher parameters : {teacher_params:,}")
print(f"Student parameters : {student_params:,}")
print(f"Parameter reduction: {reduction:,}")
print(f"Reduction percent  : {reduction_percent:.2f}%")
print(f"Teacher/Student ratio: {ratio:.2f}x")
print("=" * 70)

df = pd.DataFrame([
    {
        "model": "Teacher V2",
        "parameters": teacher_params,
        "test_accuracy": 0.6341,
        "test_macro_f1": 0.5797,
    },
    {
        "model": "Student + KD",
        "parameters": student_params,
        "test_accuracy": 0.4634,
        "test_macro_f1": 0.2587,
    },
])

df["parameter_reduction_vs_teacher_percent"] = (
    (teacher_params - df["parameters"]) / teacher_params * 100
)

output = "results/model_efficiency.csv"
df.to_csv(output, index=False)

print(f"Saved: {output}")
