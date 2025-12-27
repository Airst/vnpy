import os
from datetime import datetime
from typing import List, Tuple
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database

class FundamentalSelector:
    def __init__(self):
        self.database = get_database()

    def get_candidate_symbols(self) -> List[str]:
        """
        Returns a list of vt_symbols (e.g. '000001.SZSE') available in the database.
        """
        symbols = []
        overviews = self.database.get_bar_overview()
        
        for overview in overviews:
            if overview.interval == Interval.DAILY:
                vt_symbol = f"{overview.symbol}.{overview.exchange.value}"
                symbols.append(vt_symbol)
        
        return symbols

    def get_data_range(self) -> Tuple[datetime, datetime]:
        """
        Returns the overall start and end date of the available data in the database.
        """
        overviews = self.database.get_bar_overview()
        if not overviews:
            return None, None
            
        min_start = None
        max_end = None
        
        for overview in overviews:
            if overview.interval == Interval.DAILY:
                if overview.start:
                    if min_start is None or overview.start < min_start:
                        min_start = overview.start
                
                if overview.end:
                    if max_end is None or overview.end > max_end:
                        max_end = overview.end
                
        return min_start, max_end
