import pandas as pd

def mapping_excel(map_file: str, target_file: str):
    df1 = pd.read_excel(map_file)
    df2 = pd.read_excel(target_file)

    merged_df = df2.merge(df1, left_on="Course ID", right_on="course_id", how="left")
    merged_df.rename(columns={"course_name": "Vietnamese Course Name"}, inplace=True)
    merged_df.drop('course_id', axis=1, inplace=True)
    if "Year" in merged_df.columns:
        merged_df["Year"] = (
            merged_df["Year"]
            .astype(str)
            .str.replace("'", "", regex=False)
            .str.strip()
            .replace("", pd.NA)  
        )
        merged_df["Year"] = pd.to_numeric(merged_df["Year"], errors="coerce").astype("Int64")

    merged_df.to_excel("../output/course_reading_lists_with_vi.xlsx", index=False)

    print(f"Merged successfully: {len(merged_df)} rows")
    print(merged_df.head())