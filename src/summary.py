import pandas as pd


class ActivitySummary:

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        if "Person ID" in self.df.columns:
            self.person_col = "Person ID"
        elif "Person_ID" in self.df.columns:
            self.person_col = "Person_ID"
        else:
            self.person_col = None

        if "Activity" in self.df.columns:
            self.activity_col = "Activity"
        elif "Event" in self.df.columns:
            self.activity_col = "Event"
        else:
            self.activity_col = None

        if "Time" in self.df.columns:
            self.time_col = "Time"
        else:
            self.time_col = None

    def generate(self):

        report = {}

        report["Total Events"] = len(self.df)

        if self.person_col:
            report["Total Persons"] = self.df[self.person_col].nunique()
        else:
            report["Total Persons"] = 0

        if self.activity_col:

            counts = self.df[self.activity_col].value_counts()

            report["Most Common Activity"] = counts.idxmax()

            report["Activity Counts"] = counts.to_dict()

        else:

            report["Most Common Activity"] = "N/A"
            report["Activity Counts"] = {}

        if self.time_col and len(self.df) > 0:
            report["Start Time"] = self.df[self.time_col].iloc[0]
            report["End Time"] = self.df[self.time_col].iloc[-1]
        else:
            report["Start Time"] = "N/A"
            report["End Time"] = "N/A"

        return report