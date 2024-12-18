# Import required modules and functions
from src.tracking.trackpy_tracker import TrackpyTracker
from src.evaluation.evaluator import TrackEvaluator
from src.helper.constant import get_prelap_save_path, get_pred_csv_path
import pandas as pd

class TrackingPipeline:
    def __init__(self, seq, gap):
        """
        Initialize the tracking pipeline with sequence and gap parameters.
        """
        self.seq = seq
        self.gap = gap
        self.path = get_prelap_save_path(seq, gap)
        self.res_csv = get_pred_csv_path(seq, gap)
        self.fcc = False  # Flag for first contact cell

    def run(self):
        """
        Execute the tracking pipeline.
        """
        print(f"Pre-lap save path: {self.path}")

        # Read the pre-lap data into a DataFrame
        df = pd.read_csv(self.path)

        # Initialize the 'u_flag' column to 1
        df['u_flag'] = 1

        # Initialize the TrackpyTracker and link tracks in the dataframe
        tracker = TrackpyTracker()
        res_df = tracker.link_df(df, self.fcc)

        # Save the resulting linked DataFrame to a CSV file
        print(f"Prediction CSV path: {self.res_csv}")
        res_df.to_csv(self.res_csv, index=False)

        # Evaluate the performance using TrackEvaluator
        evaluator = TrackEvaluator(res_csv_=self.res_csv)
        perf = evaluator.get_perf()

        # Print the evaluation performance metrics
        print(f"Performance metrics: {perf}")


# Execute the tracking pipeline when the script is run
if __name__ == "__main__":
    pipeline = TrackingPipeline(seq='FLD_7', gap=1)
    pipeline.run()
