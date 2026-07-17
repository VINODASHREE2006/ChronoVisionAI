import pandas as pd


class DashboardAnalytics:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def person_list(self):
        if "Person ID" in self.df.columns:
            return ["All"] + sorted(
                self.df["Person ID"].astype(str).unique().tolist()
            )
        return ["All"]

    def activity_list(self):
        if "Activity" in self.df.columns:
            return ["All"] + sorted(
                self.df["Activity"].dropna().unique().tolist()
            )
        return ["All"]

    def filtered_data(self, person="All", activity="All"):

        df = self.df.copy()

        if person != "All" and "Person ID" in df.columns:
            df = df[df["Person ID"].astype(str) == str(person)]

        if activity != "All" and "Activity" in df.columns:
            df = df[df["Activity"] == activity]

        return df

    @property
    def person_col(self):
        if "Person ID" in self.df.columns:
            return "Person ID"
        return None

    @property
    def activity_col(self):
        if "Activity" in self.df.columns:
            return "Activity"
        return None

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