import pandas as pd

from src.utils import normalize_timeline_columns


class DashboardAnalytics:
    """Load and filter timeline data for the Streamlit dashboard."""

    def __init__(self, csv_file):
        self.df = normalize_timeline_columns(pd.read_csv(csv_file))

    def person_list(self):
        if "Person" in self.df.columns:
            return ["All"] + sorted(
                self.df["Person"].dropna().astype(str).unique().tolist()
            )
        return ["All"]

    def activity_list(self):
        if "Activity" in self.df.columns:
            return ["All"] + sorted(
                self.df["Activity"].dropna().unique().tolist()
            )
        return ["All"]

    def filtered_data(self, person="All", activity="All", search=""):
        df = self.df.copy()

        if person != "All" and "Person" in df.columns:
            df = df[df["Person"].astype(str) == str(person)]

        if activity != "All" and "Activity" in df.columns:
            df = df[df["Activity"] == activity]

        if search and len(df):
            mask = pd.Series(False, index=df.index)
            for column in ["Timestamp", "Person", "Activity"]:
                if column in df.columns:
                    mask = mask | df[column].astype(str).str.contains(
                        search,
                        case=False,
                        na=False,
                    )
            df = df[mask]

        return df

    @property
    def person_col(self):
        return "Person" if "Person" in self.df.columns else None

    @property
    def activity_col(self):
        return "Activity" if "Activity" in self.df.columns else None

    def total_events(self):
        return len(self.df)

    def total_persons(self):
        if self.person_col:
            return self.df[self.person_col].nunique()
        return 0

    def activity_counts(self):
        if self.activity_col:
            return self.df[self.activity_col].value_counts()
        return pd.Series(dtype=int)
