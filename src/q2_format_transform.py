import pandas as pd
import numpy as np
from pathlib import Path

# Get the project root directory
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Read the data
df = pd.read_csv(
    project_root
    / "src"
    / "output"
    / "question2_res"
    / "q4"
    / "controversial_both_methods_bottom.csv"
)

print("Original data shape:", df.shape)
print("\nFirst few rows of original data:")
print(df.head())

# Transform to wide format: pivot by celebrity_name and season/week combinations
# We're interested in tracking who appears in the bottom across different weeks
print("\n" + "=" * 80)
print("TRANSFORMING TO WIDE FORMAT")
print("=" * 80)

# Create a unique identifier for each appearance
df["appearance_id"] = df["season"].astype(str) + "_Week" + df["week"].astype(str)

# Pivot to wide format - each row is a celebrity, columns are their statistics across appearances
wide_df = df.pivot_table(
    index="celebrity_name",
    columns="appearance_id",
    values=["both_methods_bottom_two", "controversial_bottom"],
    aggfunc="first",
).reset_index()

# Flatten column names
wide_df.columns = [
    "_".join(col).strip("_") if col[1] else col[0] for col in wide_df.columns
]

print("\nWide format shape:", wide_df.shape)
print("\nWide format columns:", list(wide_df.columns))
print("\nFirst few rows of wide format:")
print(wide_df.head())

# Save wide format
output_path_wide = (
    project_root
    / "src"
    / "output"
    / "question2_res"
    / "q4"
    / "controversial_wide_format.csv"
)
wide_df.to_csv(output_path_wide, index=False)
print(f"\n✓ Wide format saved to: {output_path_wide}")

# Now find the most controversial people
# Those who appear in bottom two with BOTH methods most frequently
print("\n" + "=" * 80)
print("FINDING MOST CONTROVERSIAL CELEBRITIES")
print("=" * 80)

# Group by celebrity and count occurrences
controversy_stats = (
    df.groupby("celebrity_name")
    .agg(
        {
            "both_methods_bottom_two": "sum",  # Times they were in bottom two by BOTH methods
            "controversial_bottom": "sum",  # Times they were controversial (bottom in one but not both)
            "method1_is_bottom_two": "sum",  # Times in bottom two by method 1
            "method2_is_bottom_two": "sum",  # Times in bottom two by method 2
            "week": "count",  # Total appearances
            "season": "first",  # Which season(s)
        }
    )
    .reset_index()
)

# Rename columns for clarity
controversy_stats.columns = [
    "celebrity_name",
    "both_methods_bottom_count",
    "controversial_bottom_count",
    "method1_bottom_count",
    "method2_bottom_count",
    "total_appearances",
    "first_season",
]

# Calculate percentages
controversy_stats["both_methods_pct"] = (
    controversy_stats["both_methods_bottom_count"]
    / controversy_stats["total_appearances"]
    * 100
).round(2)

# Sort by most controversial (high count of both_methods_bottom_two)
controversy_stats = controversy_stats.sort_values(
    ["both_methods_bottom_count", "both_methods_pct"], ascending=[False, False]
)

print("\nTop 20 Most Controversial Celebrities:")
print("(Those who were in bottom two by BOTH methods most frequently)")
print("-" * 80)
print(controversy_stats.head(20).to_string(index=False))

# Filter for those who were in bottom by BOTH methods at least twice
highly_controversial = controversy_stats[
    controversy_stats["both_methods_bottom_count"] >= 2
].copy()

print(
    f"\n\nCelebrities with 2+ appearances in bottom by BOTH methods: {len(highly_controversial)}"
)
print("-" * 80)
print(highly_controversial.to_string(index=False))

# Add detailed week information for highly controversial celebrities
print("\n" + "=" * 80)
print("ADDING DETAILED WEEK INFORMATION FOR HIGHLY CONTROVERSIAL CELEBRITIES")
print("=" * 80)

highly_controversial_detailed = []

for _, row in highly_controversial.iterrows():
    name = row["celebrity_name"]

    # Get all appearances for this celebrity
    celebrity_weeks = df[df["celebrity_name"] == name].copy()

    # Get weeks where both methods showed bottom two
    both_bottom_weeks = celebrity_weeks[celebrity_weeks["both_methods_bottom_two"] == 1]

    # Format the week information
    week_info_list = []
    for _, week_row in both_bottom_weeks.iterrows():
        week_str = f"S{int(week_row['season'])}W{int(week_row['week'])}"
        week_info_list.append(week_str)

    weeks_both_bottom = ", ".join(week_info_list)

    # Get all weeks where they appeared in bottom (either method)
    method1_weeks = celebrity_weeks[celebrity_weeks["method1_is_bottom_two"] == 1]
    method2_weeks = celebrity_weeks[celebrity_weeks["method2_is_bottom_two"] == 1]

    method1_week_list = [
        f"S{int(r['season'])}W{int(r['week'])}" for _, r in method1_weeks.iterrows()
    ]
    method2_week_list = [
        f"S{int(r['season'])}W{int(r['week'])}" for _, r in method2_weeks.iterrows()
    ]

    weeks_method1 = ", ".join(method1_week_list)
    weeks_method2 = ", ".join(method2_week_list)

    # Create detailed row
    detailed_row = {
        "celebrity_name": name,
        "both_methods_bottom_count": row["both_methods_bottom_count"],
        "both_methods_pct": row["both_methods_pct"],
        "weeks_both_bottom": weeks_both_bottom,
        "method1_bottom_count": row["method1_bottom_count"],
        "weeks_method1_bottom": weeks_method1,
        "method2_bottom_count": row["method2_bottom_count"],
        "weeks_method2_bottom": weeks_method2,
        "controversial_bottom_count": row["controversial_bottom_count"],
        "total_appearances": row["total_appearances"],
        "first_season": row["first_season"],
    }

    highly_controversial_detailed.append(detailed_row)

    print(f"\n{name}:")
    print(f"  Both methods bottom: {weeks_both_bottom}")
    print(f"  Method 1 bottom: {weeks_method1}")
    print(f"  Method 2 bottom: {weeks_method2}")

# Convert to DataFrame
highly_controversial_detailed_df = pd.DataFrame(highly_controversial_detailed)

# Save the controversy statistics
output_path_stats = (
    project_root
    / "src"
    / "output"
    / "question2_res"
    / "q4"
    / "most_controversial_celebrities.csv"
)
controversy_stats.to_csv(output_path_stats, index=False)
print(f"\n✓ Controversy statistics saved to: {output_path_stats}")

# Save highly controversial ones with detailed week information
output_path_high = (
    project_root
    / "src"
    / "output"
    / "question2_res"
    / "q4"
    / "highly_controversial_celebrities.csv"
)
highly_controversial_detailed_df.to_csv(output_path_high, index=False)
print(
    f"✓ Highly controversial (2+ times) with week details saved to: {output_path_high}"
)

# Also save a more detailed breakdown with all original data
highly_controversial_names = highly_controversial["celebrity_name"].tolist()
detailed_appearances = df[
    df["celebrity_name"].isin(highly_controversial_names)
].sort_values(["celebrity_name", "season", "week"])

output_path_detailed = (
    project_root
    / "src"
    / "output"
    / "question2_res"
    / "q4"
    / "highly_controversial_all_appearances.csv"
)
detailed_appearances.to_csv(output_path_detailed, index=False)
print(
    f"✓ All appearances of highly controversial celebrities saved to: {output_path_detailed}"
)

# Additional analysis: Show detailed appearances for top controversial celebrities
print("\n" + "=" * 80)
print("DETAILED BREAKDOWN OF TOP 5 CONTROVERSIAL CELEBRITIES")
print("=" * 80)

top_5_names = controversy_stats.head(5)["celebrity_name"].tolist()

for name in top_5_names:
    print(f"\n{name}:")
    print("-" * 60)
    celebrity_data = df[df["celebrity_name"] == name][
        [
            "season",
            "week",
            "judge_score",
            "fan_votes",
            "method1_is_bottom_two",
            "method2_is_bottom_two",
            "both_methods_bottom_two",
            "controversial_bottom",
        ]
    ].sort_values(["season", "week"])
    print(celebrity_data.to_string(index=False))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
