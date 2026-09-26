import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from config import CHART_OUTPUT_PATH
from data_store import store
from tools.data_tools import create_sample_dataset
from tools.insight_tools import generate_insights
from tools.parsing_utils import parse_kv_string, resolve_column
from tools.report_tools import generate_summary_report
from tools.stats_tools import correlation_analysis, outlier_detection
from tools.transform_tools import add_calculated_column, aggregate_data, filter_data
from tools.viz_tools import create_visualization


class DeterministicToolTests(unittest.TestCase):
    def setUp(self):
        store.datasets.clear()
        store.history.clear()

    def test_parser_preserves_comma_and_equals_in_value(self):
        parsed = parse_kv_string(
            "dataset_name=sales, condition=age >= 30, new_dataset_name=adults"
        )
        self.assertEqual(parsed["dataset_name"], "sales")
        self.assertEqual(parsed["condition"], "age >= 30")
        self.assertEqual(parsed["new_dataset_name"], "adults")

    def test_sample_dataset_is_reproducible_and_has_expected_schema(self):
        result = create_sample_dataset.invoke(
            "dataset_type=sales, dataset_name=sales_data"
        )
        df = store.get("sales_data")

        self.assertIn("Đã tạo dataset 'sales_data'", result)
        self.assertEqual(df.shape, (200, 5))
        self.assertEqual(
            list(df.columns), ["date", "region", "product", "revenue", "units_sold"]
        )
        self.assertEqual(df.iloc[0]["revenue"], 1141.97)

    def test_column_resolution_is_case_and_separator_insensitive(self):
        df = pd.DataFrame({"Revenue Per User": [10, 20]})
        self.assertEqual(resolve_column(df, "revenue_per_user"), "Revenue Per User")
        self.assertEqual(resolve_column(df, "REVENUE PER USER"), "Revenue Per User")

    def test_filter_does_not_modify_source_dataset(self):
        store.add("sales", pd.DataFrame({"id": [1, 2, 3], "age": [20, 35, 40]}))
        result = filter_data.invoke(
            "dataset_name=sales, condition=age > 30, new_dataset_name=adults"
        )

        self.assertIn("2/3 dòng còn lại", result)
        self.assertEqual(store.get("sales")["id"].tolist(), [1, 2, 3])
        self.assertEqual(store.get("adults")["id"].tolist(), [2, 3])

    def test_aggregate_matches_expected_values(self):
        store.add(
            "sales",
            pd.DataFrame(
                {
                    "region": ["North", "North", "South"],
                    "revenue": [10.0, 15.0, 20.0],
                }
            ),
        )
        aggregate_data.invoke(
            "dataset_name=sales, group_by=region, "
            "aggregations=revenue:sum, new_dataset_name=by_region"
        )
        result = store.get("by_region").sort_values("region").reset_index(drop=True)

        expected = pd.DataFrame(
            {"region": ["North", "South"], "revenue": [25.0, 20.0]}
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_calculated_column_matches_expected_values(self):
        store.add("sales", pd.DataFrame({"revenue": [100, 250], "cost": [40, 100]}))
        add_calculated_column.invoke(
            "dataset_name=sales, new_column=profit, expression=revenue - cost"
        )
        self.assertEqual(store.get("sales")["profit"].tolist(), [60, 150])

    def test_stats_and_insight_use_actual_values(self):
        store.add("metrics", pd.DataFrame({"x": [1.0, 2.0, 100.0, 3.0]}))
        correlation = correlation_analysis.invoke("dataset_name=metrics")
        outliers = outlier_detection.invoke("dataset_name=metrics, column=x")
        insight = generate_insights.invoke("dataset_name=metrics, column=x")

        self.assertIn("Không đủ cột số", correlation)
        self.assertIn("100.0", outliers)
        self.assertIn("Trung bình=26.50", insight)
        self.assertIn("Max=100.00", insight)

    def test_visualization_creates_non_empty_artifact(self):
        store.add("metrics", pd.DataFrame({"x": np.arange(10)}))
        result = create_visualization.invoke(
            "dataset_name=metrics, chart_type=histogram, x=x"
        )

        self.assertIn("Đã tạo biểu đồ histogram", result)
        self.assertTrue(os.path.isfile(CHART_OUTPUT_PATH))
        self.assertGreater(os.path.getsize(CHART_OUTPUT_PATH), 0)

    def test_report_contains_shape_and_missing_values(self):
        store.add("metrics", pd.DataFrame({"x": [1, None], "label": ["a", "b"]}))
        report = generate_summary_report.invoke("dataset_name=metrics")

        self.assertIn("Kích thước: 2 dòng x 2 cột", report)
        self.assertIn("x    1", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
