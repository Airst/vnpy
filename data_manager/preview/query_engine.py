import re
import pandas as pd
import tushare as ts
from typing import Optional, Dict, Any
from vnpy.trader.setting import SETTINGS


class TushareQueryEngine:
    """
    A simplified SQL-like query engine for Tushare Pro API.
    Allows querying Tushare data using SQL syntax.
    
    Example usage:
        engine = TushareQueryEngine()
        df = engine.execute("SELECT ts_code, close FROM daily WHERE ts_code='000001.SZ' AND trade_date='20240101'")
    """

    def __init__(self):
        token = SETTINGS.get("datafeed.password", "")
        if not token:
            # Try to see if it is set in global environment or just warn
            print("Warning: 'datafeed.password' not found in SETTINGS. Tushare API might not work.")
            self.pro = None
        else:
            try:
                self.pro = ts.pro_api(token)
            except Exception as e:
                print(f"Failed to initialize Tushare API: {e}")
                self.pro = None

    def parse_sql(self, sql: str) -> Dict[str, Any]:
        """
        Parse a simplified SQL statement into Tushare API parameters.
        Supported Syntax: 
            SELECT <fields> FROM <api_name> [WHERE <conditions>] [LIMIT <n>]
        
        Example:
            SELECT ts_code, close FROM daily WHERE ts_code='000001.SZ' AND trade_date='20240101' LIMIT 10
        """
        sql = sql.strip()
        
        # Regex to capture main parts
        # 1. SELECT (fields)
        # 2. FROM (table)
        # 3. WHERE (conditions) - Optional
        # 4. LIMIT (limit) - Optional
        
        pattern = re.compile(
            r"SELECT\s+(?P<fields>.+?)\s+"
            r"FROM\s+(?P<table>\w+)"
            r"(?:\s+WHERE\s+(?P<where>.+?))?"
            r"(?:\s+LIMIT\s+(?P<limit>\d+))?$",
            re.IGNORECASE | re.DOTALL
        )
        
        match = pattern.match(sql)
        if not match:
            raise ValueError("Invalid SQL syntax. usage: SELECT ... FROM ... [WHERE ...] [LIMIT ...]")
            
        groups = match.groupdict()
        
        result = {
            "api_name": groups["table"],
            "fields": None,
            "params": {},
            "limit": None
        }
        
        # Parse fields
        fields_str = groups["fields"].strip()
        if fields_str and fields_str != "*":
            # Remove spaces, keep commas
            result["fields"] = ",".join([f.strip() for f in fields_str.split(",")])
            
        # Parse WHERE clause
        where_str = groups["where"]
        if where_str:
            # Split by AND
            # This is a basic parser, doesn't handle complex parens or OR
            conditions = re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)
            for cond in conditions:
                parts = cond.split("=")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    
                    # Remove quotes if present
                    if (val.startswith("'" ) and val.endswith("'" )) or \
                       (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                        
                    result["params"][key] = val
                else:
                    # Fallback for simple non-assignment conditions or ignoring complex ones
                    pass
        
        # Parse LIMIT
        if groups["limit"]:
            result["limit"] = int(groups["limit"])
            
        return result

    def execute(self, sql: str) -> pd.DataFrame:
        """
        Execute the SQL query against Tushare.
        """
        if not self.pro:
            print("Tushare API is not initialized.")
            return pd.DataFrame()
            
        try:
            parsed = self.parse_sql(sql)
        except ValueError as e:
            print(f"Parse Error: {e}")
            return pd.DataFrame()
            
        api_name = parsed["api_name"]
        params = parsed["params"]
        fields = parsed["fields"]
        limit = parsed["limit"]
        
        if not hasattr(self.pro, api_name):
            print(f"API '{api_name}' not found in Tushare.")
            return pd.DataFrame()
            
        api_func = getattr(self.pro, api_name)
        
        try:
            # Call API
            if fields:
                params["fields"] = fields
                
            print(f"Executing Tushare API: {api_name} with params: {params}")
            df = api_func(**params)
            
            if df is None:
                 return pd.DataFrame()

            if limit and not df.empty:
                df = df.head(limit)
                
            return df
            
        except Exception as e:
            print(f"Tushare API Execution Error: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    engine = TushareQueryEngine()
    print("Tushare SQL Preview Tool")
    print("------------------------")
    print("Type a SQL query to execute against Tushare (e.g. SELECT * FROM daily WHERE ts_code='000001.SZ' LIMIT 5)")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            q = input("\nSQL> ")
            if q.lower() in ["exit", "quit"]:
                break
            if not q.strip():
                continue
                
            df = engine.execute(q)
            if not df.empty:
                print(f"Result ({len(df)} rows):")
                print(df)
            else:
                print("No data returned or error occurred.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
