import pandas as pd

from src.utils import normalize_timeline_columns


class ActivitySummary:
    """Generate a human-readable summary from timeline data."""

    MOVEMENT_ACTIVITIES = {
        "Walking",
        "Slow Walking",
        "Running",
        "Standing",
        "Waiting",
        "Idle",
    }

    def __init__(self, csv_file):
        self.df = normalize_timeline_columns(pd.read_csv(csv_file))

    def generate(self):
        report = {
            "Total Events": len(self.df),
            "Total Persons": 0,
            "Most Common Activity": "N/A",
            "Start Time": "N/A",
            "End Time": "N/A",
            "Activity Counts": {},
            "Movement Summary": "No movement data available.",
        }

        if "Person" in self.df.columns:
            report["Total Persons"] = int(self.df["Person"].nunique())

        if "Activity" in self.df.columns and len(self.df):
            counts = self.df["Activity"].value_counts()
            report["Most Common Activity"] = str(counts.idxmax())
            report["Activity Counts"] = counts.to_dict()
            report["Movement Summary"] = self._movement_summary(counts)

        if "Timestamp" in self.df.columns and len(self.df):
            report["Start Time"] = str(self.df["Timestamp"].iloc[0])
            report["End Time"] = str(self.df["Timestamp"].iloc[-1])

        return report

    def _movement_summary(self, counts):
        movement_total = sum(
            int(count)
            for activity, count in counts.items()
            if activity in self.MOVEMENT_ACTIVITIES
        )
        stationary_total = sum(
            int(count)
            for activity, count in counts.items()
            if activity in {"Standing", "Waiting", "Idle", "Queueing"}
        )

        if movement_total == 0 and stationary_total == 0:
            return "No movement events recorded."

        return (
            f"{movement_total} movement-related events and "
            f"{stationary_total} stationary events were detected."
        )
