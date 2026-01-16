# Tushare Data Tables in Database

**Database:** `vnpy`

**Generated:** Fri Jan 16 14:35:40 CST 2026

## dailybasic

| Field | Type | Null | Key | Default | Extra |
|---|---|---|---|---|---|| ts_code | varchar(20) | NO | PRI | None |  |
| trade_date | varchar(20) | NO | PRI | None |  |
| close | float | YES |  | None |  |
| turnover_rate | float | YES |  | None |  |
| turnover_rate_f | float | YES |  | None |  |
| volume_ratio | float | YES |  | None |  |
| pe | float | YES |  | None |  |
| pe_ttm | float | YES |  | None |  |
| pb | float | YES |  | None |  |
| ps | float | YES |  | None |  |
| ps_ttm | float | YES |  | None |  |
| dv_ratio | float | YES |  | None |  |
| dv_ttm | float | YES |  | None |  |
| total_share | float | YES |  | None |  |
| float_share | float | YES |  | None |  |
| free_share | float | YES |  | None |  |
| total_mv | float | YES |  | None |  |
| circ_mv | float | YES |  | None |  |

## dbbardata

| Field | Type | Null | Key | Default | Extra |
|---|---|---|---|---|---|| id | int | NO | PRI | None | auto_increment |
| symbol | varchar(255) | NO | MUL | None |  |
| exchange | varchar(255) | NO |  | None |  |
| datetime | datetime | NO |  | None |  |
| interval | varchar(255) | NO |  | None |  |
| volume | double | NO |  | None |  |
| turnover | double | NO |  | None |  |
| open_interest | double | NO |  | None |  |
| open_price | double | NO |  | None |  |
| high_price | double | NO |  | None |  |
| low_price | double | NO |  | None |  |
| close_price | double | NO |  | None |  |

## dbbaroverview

| Field | Type | Null | Key | Default | Extra |
|---|---|---|---|---|---|| id | int | NO | PRI | None | auto_increment |
| symbol | varchar(255) | NO | MUL | None |  |
| exchange | varchar(255) | NO |  | None |  |
| interval | varchar(255) | NO |  | None |  |
| count | int | NO |  | None |  |
| start | datetime | NO |  | None |  |
| end | datetime | NO |  | None |  |


## dc_daily

| Field | Type | Null | Key | Default | Extra |
|---|---|---|---|---|---|| ts_code | varchar(20) | NO | PRI | None |  |
| trade_date | varchar(20) | NO | PRI | None |  |
| open | float | YES |  | None |  |
| high | float | YES |  | None |  |
| low | float | YES |  | None |  |
| close | float | YES |  | None |  |
| change | float | YES |  | None |  |
| pct_change | float | YES |  | None |  |
| vol | float | YES |  | None |  |
| amount | float | YES |  | None |  |
| swing | float | YES |  | None |  |
| turnover_rate | float | YES |  | None |  |

## dc_member

| Field | Type | Null | Key | Default | Extra |
|---|---|---|---|---|---|| ts_code | varchar(20) | NO | PRI | None |  |
| con_code | varchar(20) | NO | PRI | None |  |
| trade_date | varchar(20) | NO | PRI | None |  |
| name | varchar(20) | YES |  | None |  |

## stock_basic

| Field | Type | Null | Key | Default | Extra |
|---|---|---|---|---|---|| ts_code | varchar(20) | NO | PRI | None |  |
| symbol | varchar(20) | NO | PRI | None |  |
| exchange | varchar(20) | NO | PRI | None |  |
| name | varchar(50) | YES |  | None |  |
| area | varchar(50) | YES |  | None |  |
| industry | varchar(50) | YES |  | None |  |
| market | varchar(20) | YES |  | None |  |
| list_date | varchar(20) | YES |  | None |  |
| list_status | varchar(10) | YES |  | None |  |
| delist_date | varchar(20) | YES |  | None |  |
| is_hs | varchar(10) | YES |  | None |  |

