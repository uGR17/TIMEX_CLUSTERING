import os
from datetime import datetime

import dateparser
import pytest
from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
import pandas as pd
import numpy as np

from .context import timexseries_clustering

from timexseries_clustering.data_ingestion import ingest_timeseries, add_freq, select_timeseries_portion
from .utilities import get_fake_df


class TestDataIngestion:
    def test_infer_freq(self):
        # Local load, with datetime. Infer freq.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_1.csv"),
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                }
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2020-02-25")
        assert df.index.values[1] == Timestamp("2020-02-26")
        assert df.index.values[2] == Timestamp("2020-02-27")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == '1d'

    def test_given_freq(self):
        # Local load, with datetime. Specify freq.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_1_1.csv"),
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
                "frequency": "M"
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2011-01-31 18:00:00")
        assert df.index.values[1] == Timestamp("2011-02-28 18:00:00")
        assert df.index.values[2] == Timestamp("2011-03-31 18:00:00")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == 'M'

    @pytest.mark.parametrize(
        "rename_index",
        [True, False]
    )
    def test_rename_columns(self, rename_index):
        # Local load, with diff columns. Rename columns.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_1_2.csv"),
                "columns_to_load_from_url": "third_column,second_column,first_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
                "add_diff_column": "third_column,second_column",
                "timeseries_names":
                    {
                        "second_column": "B",
                        "third_column": "C",
                        "third_column_diff": "D",
                        "second_column_diff": "E"
                    }
            }
        }

        if rename_index:
            param_config["input_parameters"]["timeseries_names"]["first_column"] = "A"
            index_name = "A"
        else:
            index_name = "first_column"

        df = ingest_timeseries(param_config)

        assert df.index.name == index_name
        assert df.columns[0] == "C"
        assert df.columns[1] == "B"
        assert df.columns[2] == "D"
        assert df.columns[3] == "E"
        assert len(df.columns) == 4
        assert len(df) == 3

    def test_check_duplicated_data(self):
        # Test that duplicated data is removed.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_4.csv"),
                "columns_to_load_from_url": "first_column,second_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
            }
        }

        df = ingest_timeseries(param_config)

        assert len(df) == 3
        assert df.iloc[0, 0] == 1
        assert df.iloc[1, 0] == 3
        assert df.iloc[2, 0] == 4

    @pytest.mark.parametrize(
        "columns_to_load_from_url",
        [None, ""]
    )
    def test_read_all_columns(self, columns_to_load_from_url):
        # Local load, read all columns.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_1.csv"),
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                }
            }
        }
        if columns_to_load_from_url is None:
            pass
        else:
            param_config["input_parameters"]["columns_to_load_from_url"] = columns_to_load_from_url

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2020-02-25")
        assert df.index.values[1] == Timestamp("2020-02-26")
        assert df.index.values[2] == Timestamp("2020-02-27")

        assert df.columns[0] == "second_column"
        assert df.iloc[0]["second_column"] == 2
        assert df.iloc[1]["second_column"] == 5
        assert df.iloc[2]["second_column"] == 8

        assert df.columns[1] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == '1d'

    def test_italian_months_names(self):
        # Local load, with datetime with italian months (e.g. Gen, Feb, etc.)
        # Check that monthly freq is applied.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_5.csv"),
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2020-11-01")
        assert df.index.values[1] == Timestamp("2020-12-01")
        assert df.index.values[2] == Timestamp("2021-01-01")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == 'MS'

    def test_dates_with_only_months_and_years(self):
        # Datetime with only the month and the year specified (e.g. "1959-01").
        # Check that monthly freq is applied.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_5_1.csv"),
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("1959-01-01")
        assert df.index.values[1] == Timestamp("1959-02-01")
        assert df.index.values[2] == Timestamp("1959-03-01")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == 'MS'

    def test_interpolate_data(self):
        # Check that data is interpolated.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_6.csv"),
                "columns_to_load_from_url": "first_column,second_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
            }
        }

        df = ingest_timeseries(param_config)

        expected = pd.read_csv(os.path.join("test_datasets", "test_6_expected.csv"), parse_dates=["first_column"],
                               index_col="first_column")
        expected = expected.asfreq("D")

        pd.testing.assert_frame_equal(df, expected)
        assert df.iloc[:, 0].isnull().sum() == 0

    def test_no_nans_left(self, tmp_path):
        # Check that no dataset exits ingest_timeseries without a frequency and with nan values.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join(tmp_path, "test.csv"),
                "columns_to_load_from_url": "date,a,b",
                "datetime_column_name": "date",
                "index_column_name": "date",
            }
        }

        ing_df = pd.read_csv("test_datasets/test_7.csv")
        ing_df['date'] = ing_df['date'].apply(lambda x: dateparser.parse(x))
        ing_df.set_index("date", inplace=True, drop=True)

        for i in range(0, 40):
            df = ing_df.copy()
            for j in range(0, i):
                df = df.drop(df.sample(1).index)

            df.to_csv(os.path.join(tmp_path, "test.csv"))
            df = ingest_timeseries(param_config)

            assert df.iloc[:, 0].isnull().sum() == 0
            assert df.index.freq == "D"

    def test_data_reordering(self):
        # Check that the dataframe is reordered from lowest to higher datetime index.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_data_reordering.csv"),
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
            }
        }

        index = pd.DatetimeIndex([pd.Timestamp("2020-02-25"),
                                  pd.Timestamp("2020-02-26"),
                                  pd.Timestamp("2020-02-27"),
                                  pd.Timestamp("2020-02-28")], freq="D")

        df = ingest_timeseries(param_config)
        assert len(df.columns) == 2
        assert df.index.name == "first_column"
        assert df.columns[0] == "second_column"
        assert df.columns[1] == "third_column"
        assert len(df) == 4

        assert df.index.equals(index)
        assert df.index.freq == '1d'

        assert df["second_column"].equals(pd.Series(index=index, data=np.array([2, 5, 8, 11])))
        assert df["third_column"].equals(pd.Series(index=index, data=np.array([3, 6, 9, 12])))

    def test_automatic_index_choice(self):
        # Check that the first column is automatic used as index and parsed as datetime, if index column is not
        # specified by user.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_6.csv"),
            }
        }

        df = ingest_timeseries(param_config)
        assert len(df.columns) == 1
        assert df.index.name == "first_column"


class TestAddFreq:
    def test_freq_not_set_on_not_datetime_index(self):
        # df has not a datetime index. Check that it is not touched.
        df = DataFrame(data={"A": [1, 2, 3], "B": [10, 20, 30]})
        df.set_index("A", inplace=True)
        new_df = add_freq(df)

        assert df.equals(new_df)

    def test_freq_already_set(self):
        # df already has freq; do nothing.
        df = get_fake_df(10)
        new_df = add_freq(df)

        assert df.equals(new_df)

    def test_daily_freq_inferred(self):
        # df is daily; check if it is set, without passing it.
        df = get_fake_df(10)
        df.index.freq = None

        new_df = add_freq(df)
        assert df.equals(new_df)
        assert new_df.index.freq == "D"

    def test_daily_freq_imposed(self):
        # df is daily; check if it is set, passing it.
        df = get_fake_df(10)
        df.index.freq = None

        new_df = add_freq(df, "D")
        assert df.equals(new_df)
        assert new_df.index.freq == "D"

    def test_daily_freq_normalize(self):
        # df is daily, but with different hours.
        # Check if it is set so.
        dates = [pd.Timestamp(datetime(year=2020, month=1, day=1, hour=10, minute=00)),
                 pd.Timestamp(datetime(year=2020, month=1, day=2, hour=12, minute=21)),
                 pd.Timestamp(datetime(year=2020, month=1, day=3, hour=13, minute=30)),
                 pd.Timestamp(datetime(year=2020, month=1, day=4, hour=11, minute=32))]

        ts = pd.DataFrame(np.random.randn(4), index=dates)

        new_ts = add_freq(ts)
        assert ts.iloc[0].equals(new_ts.iloc[0])
        assert new_ts.index[0] == Timestamp('2020-01-01 00:00:00', freq='D')
        assert new_ts.index[1] == Timestamp('2020-01-02 00:00:00', freq='D')
        assert new_ts.index[2] == Timestamp('2020-01-03 00:00:00', freq='D')
        assert new_ts.index[3] == Timestamp('2020-01-04 00:00:00', freq='D')
        assert new_ts.index.freq == "D"

    def test_unclear_freq_set_daily(self):
        # df has no clear frequency.
        # Check if it is set daily.
        dates = [pd.Timestamp(datetime(year=2020, month=1, day=1, hour=10, minute=00)),
                 pd.Timestamp(datetime(year=2020, month=1, day=3, hour=12, minute=21)),
                 pd.Timestamp(datetime(year=2020, month=1, day=7, hour=13, minute=30)),
                 pd.Timestamp(datetime(year=2020, month=1, day=19, hour=11, minute=32))]

        ts = pd.DataFrame(np.random.randn(4), index=dates)

        new_ts = add_freq(ts)
        assert new_ts.index.freq == "D"


class TestDataSelection:
    def test_selection_not_requested(self):
        # Test if df is returned untouched if selection is not required.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_6.csv"),
            }
        }

        df = ingest_timeseries(param_config)
        selected_df = select_timeseries_portion(df, param_config)
        assert df.equals(selected_df)

    def test_init_datetime(self):
        # Select rows using init datetime.
        param_config = {
            "selection_parameters": {
                "init_datetime": "2000-01-02",
                "end_datetime": "2000-12-09"
            },
        }

        df = get_fake_df(length=3)
        selected_df = select_timeseries_portion(df, param_config)

        assert selected_df.index.values[0] == Timestamp("2000-01-02")
        assert selected_df.index.values[1] == Timestamp("2000-01-03")

        assert df.iloc[1]["value"] == selected_df.iloc[0]["value"]
        assert df.iloc[2]["value"] == selected_df.iloc[1]["value"]
        assert len(selected_df) == 2

    def test_end_datetime(self):
        # Select rows using end datetime.
        param_config = {
            "selection_parameters": {
                "init_datetime": "1999-01-02",
                "end_datetime": "2000-01-02"
            },
        }

        df = get_fake_df(length=3)
        selected_df = select_timeseries_portion(df, param_config)

        assert selected_df.index.values[0] == Timestamp("2000-01-01")
        assert selected_df.index.values[1] == Timestamp("2000-01-02")

        assert df.iloc[0]["value"] == selected_df.iloc[0]["value"]
        assert df.iloc[1]["value"] == selected_df.iloc[1]["value"]
        assert len(selected_df) == 2

    @pytest.mark.parametrize(
        "are_dateparser_options_specified",
        [True, False]
    )
    def test_init_and_end_datetime(self, are_dateparser_options_specified):
        # Select rows using both init and end time.
        param_config = {
            "selection_parameters": {
                "init_datetime": "2000-01-02",
                "end_datetime": "2000-01-04"
            },
        }

        if are_dateparser_options_specified:
            param_config["input_parameters"] = {}
            param_config["input_parameters"]["dateparser_options"] = {
                "date_formats": ["%Y-%m-%dT%H:%M:%S"]
            }

        df = get_fake_df(length=5)
        selected_df = select_timeseries_portion(df, param_config)

        assert selected_df.index.values[0] == Timestamp("2000-01-02")
        assert selected_df.index.values[1] == Timestamp("2000-01-03")
        assert selected_df.index.values[2] == Timestamp("2000-01-04")

        assert df.iloc[1]["value"] == selected_df.iloc[0]["value"]
        assert df.iloc[2]["value"] == selected_df.iloc[1]["value"]
        assert df.iloc[3]["value"] == selected_df.iloc[2]["value"]

        assert len(selected_df) == 3

    def test_select_on_values(self):
        # Select rows based on value.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join("test_datasets", "test_1.csv"),
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "frequency": "D"
            },
            "selection_parameters": {
                "column_name_selection": "third_column",
                "value_selection": 3,
            },
        }

        df = ingest_timeseries(param_config)
        df = select_timeseries_portion(df, param_config)

        assert df.index.values[0] == Timestamp("2020-02-25")

        assert df.iloc[0]["third_column"] == 3
        assert len(df) == 1