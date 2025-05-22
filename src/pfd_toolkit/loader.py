import pandas as pd
from dateutil import parser as date_parser
import importlib.resources as resources

class Dataset:
    """
    Provides access to fully cleaned and ready-to-use Prevention of Future Death
    (PFD) reports.
    """

    def __init__(
        self,
        category: str = 'all',
        date_from: str = '2000-01-01',
        date_to: str = '2100-01-01'
    ) -> None:
        """Initialises the dataset loader with specified configurations.

        Args:
            category (str, optional): The category of PFD reports. Defaults to 'all'.
            date_from (str, optional): The start date for filtering PFD reports. Defaults to '2000-01-01'.
            date_to (str, optional): The end date for filtering PFD reports. Defaults to '2100-01-01'.
        """
        
        self.category = category.lower() # Normalise category to lowercase if user specifies otherwise
        
        # Parse date strings into datetime objects (for internal use)
        self.date_from = date_parser.parse(date_from)
        self.date_to = date_parser.parse(date_to)
        
        # Throw error if date_to is earlier than date_from
        if self.date_from > self.date_to:
            raise ValueError("date_from must be earlier than or equal to date_to.")
        
        
    def _get_data(self) -> pd.DataFrame:
        # Retrive data from `data` directory
        with resources.files('pfd_toolkit.data').joinpath('all_reports.csv').open('r') as f:
            reports = pd.read_csv(f)

        # -- Date logic --
        
        # Make Date column into pandas datetime obj
        reports['Date'] = pd.to_datetime(reports['Date'],
                                         format='%Y-%m-%d', # YYYY-MM-DD
                                         errors = 'coerce') # Force errors into NaT (not a time)
        
        # Drop rows where date conversion failed (invalid or missing dates)
        reports = reports.dropna(subset=['Date']).reset_index(drop=True)
        
        # Filter date based on user's specification
        reports = reports[(reports['Date'] >= self.date_from) & (reports['Date'] <= self.date_to)]
        
        # Organise by date
        reports = reports.sort_values(by=['Date'], ascending=False)
        
        return reports

# Public function to retrieve data
def load_reports(
    category: str = 'all',
    start_date: str = '2000-01-01',
    end_date: str = '2100-01-01'
) -> pd.DataFrame:
    """
    Quickly load cleaned Prevention of Future Death reports.

    Args:
        category (str, optional): Filter by category. Defaults to 'all'.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to '2000-01-01'.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to '2100-01-01'.

    Returns:
        pd.DataFrame: Filtered DataFrame of PFD reports.
    """
    dataset = Dataset(category=category, date_from=start_date, date_to=end_date)
    return dataset._get_data()