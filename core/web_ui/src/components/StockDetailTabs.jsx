import React, { useState, useEffect, useRef } from 'react';
import { Row, Col, Spin, Empty, Select, Button, DatePicker, Tabs } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { createChart, CandlestickSeries, HistogramSeries } from 'lightweight-charts';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;

const SIGNAL_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#d0ed57', '#a4de6c', '#8dd1e1', '#83a6ed'];

/**
 * Shared tabs for signal trend chart and K-line chart.
 * Props:
 *   vtSymbol   - string, e.g. "000001.SZ"
 *   signals    - string[], available signal names (fetched once by parent or this component)
 *   extraTabs  - array of { key, label, children } to prepend before signal/kline tabs
 *   defaultTab - default active tab key (default: 'signal')
 */
const StockDetailTabs = ({ vtSymbol, signals: signalsProp, extraTabs, defaultTab }) => {
    const [signals, setSignals] = useState(signalsProp || []);
    const [detailSignal, setDetailSignal] = useState(null);
    const [detailDateRange, setDetailDateRange] = useState([dayjs().subtract(14, 'day'), dayjs()]);
    const [detailChartData, setDetailChartData] = useState([]);
    const [detailChartLoading, setDetailChartLoading] = useState(false);
    const [detailChartSeries, setDetailChartSeries] = useState([]);

    const [klineDateRange, setKlineDateRange] = useState([dayjs().subtract(1, 'year'), dayjs()]);
    const [klineLoading, setKlineLoading] = useState(false);
    const [klineData, setKlineData] = useState([]);
    const klineContainerRef = useRef(null);
    const klineChartRef = useRef(null);

    // Load signals if not provided by parent
    useEffect(() => {
        if (signalsProp && signalsProp.length > 0) {
            setSignals(signalsProp);
            return;
        }
        fetch('/api/signals')
            .then(res => res.json())
            .then(data => { if (data.signals) setSignals(data.signals); })
            .catch(() => {});
    }, [signalsProp]);

    // Reset state when vtSymbol changes
    useEffect(() => {
        setDetailSignal(null);
        setDetailChartData([]);
        setDetailChartSeries([]);
        setDetailDateRange([dayjs().subtract(14, 'day'), dayjs()]);
        setKlineData([]);
        setKlineDateRange([dayjs().subtract(1, 'year'), dayjs()]);
        if (klineChartRef.current) {
            klineChartRef.current.remove();
            klineChartRef.current = null;
        }
    }, [vtSymbol]);

    const loadSignalChart = async (symbol, signalName, dateRange) => {
        if (!signalName || !dateRange || !dateRange[0] || !dateRange[1]) return;
        setDetailChartLoading(true);
        try {
            const res = await fetch('/api/signal_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    signal_name: signalName,
                    start_date: dateRange[0].format("YYYYMMDD"),
                    end_date: dateRange[1].format("YYYYMMDD"),
                    vt_symbols: [symbol]
                })
            });
            const result = await res.json();
            if (result.dates && result.series) {
                const transformed = result.dates.map((date, idx) => {
                    const point = { date };
                    result.series.forEach(s => { point[s.name] = s.data[idx]; });
                    return point;
                });
                setDetailChartData(transformed);
                setDetailChartSeries(result.series);
            } else {
                setDetailChartData([]);
                setDetailChartSeries([]);
            }
        } catch {
            setDetailChartData([]);
            setDetailChartSeries([]);
        } finally {
            setDetailChartLoading(false);
        }
    };

    const loadKlineData = async (symbol, dateRange) => {
        if (!symbol || !dateRange || !dateRange[0] || !dateRange[1]) return;
        setKlineLoading(true);
        try {
            const params = new URLSearchParams({
                vt_symbol: symbol,
                start_date: dateRange[0].format("YYYYMMDD"),
                end_date: dateRange[1].format("YYYYMMDD"),
            });
            const res = await fetch(`/api/kline?${params}`);
            const result = await res.json();
            setKlineData(result.data || []);
        } catch {
            setKlineData([]);
        } finally {
            setKlineLoading(false);
        }
    };

    // Render K-line chart
    useEffect(() => {
        if (!klineContainerRef.current || klineData.length === 0) {
            if (klineChartRef.current) { klineChartRef.current.remove(); klineChartRef.current = null; }
            return;
        }
        if (klineChartRef.current) { klineChartRef.current.remove(); klineChartRef.current = null; }

        const chart = createChart(klineContainerRef.current, {
            width: klineContainerRef.current.clientWidth,
            height: 400,
            layout: { background: { color: '#ffffff' }, textColor: '#333' },
            grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
            timeScale: { timeVisible: false, borderColor: '#d1d4dc' },
            crosshair: { mode: 0 },
        });

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#ef5350', downColor: '#26a69a',
            borderUpColor: '#ef5350', borderDownColor: '#26a69a',
            wickUpColor: '#ef5350', wickDownColor: '#26a69a',
        });
        candleSeries.setData(klineData);

        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: { type: 'volume' }, priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
        volumeSeries.setData(klineData.map(d => ({
            time: d.time, value: d.volume,
            color: d.close >= d.open ? '#ef535080' : '#26a69a80',
        })));

        chart.timeScale().fitContent();
        klineChartRef.current = chart;

        const handleResize = () => {
            if (klineContainerRef.current && klineChartRef.current) {
                klineChartRef.current.applyOptions({ width: klineContainerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
            if (klineChartRef.current) { klineChartRef.current.remove(); klineChartRef.current = null; }
        };
    }, [klineData]);

    const signalTab = {
        key: 'signal',
        label: '信号趋势',
        children: (
            <>
                <div style={{ marginBottom: 12 }}>
                    <Row gutter={[12, 8]} align="middle">
                        <Col flex="auto">
                            <Select
                                style={{ width: '100%' }}
                                placeholder="选择信号"
                                options={signals.map(s => ({ value: s, label: s }))}
                                value={detailSignal}
                                onChange={(value) => {
                                    setDetailSignal(value);
                                    loadSignalChart(vtSymbol, value, detailDateRange);
                                }}
                            />
                        </Col>
                        <Col>
                            <RangePicker
                                value={detailDateRange}
                                onChange={(dates) => {
                                    setDetailDateRange(dates);
                                    if (detailSignal && dates && dates[0] && dates[1]) {
                                        loadSignalChart(vtSymbol, detailSignal, dates);
                                    }
                                }}
                                format="YYYY-MM-DD"
                            />
                        </Col>
                    </Row>
                </div>
                <Spin spinning={detailChartLoading}>
                    {detailChartData.length > 0 ? (
                        <div style={{ height: 300, width: '100%' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={detailChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                                    <YAxis tick={{ fontSize: 11 }} />
                                    <Tooltip />
                                    <Legend />
                                    {detailChartSeries.map((s, idx) => (
                                        <Line
                                            key={s.name}
                                            type="monotone"
                                            dataKey={s.name}
                                            stroke={SIGNAL_COLORS[idx % SIGNAL_COLORS.length]}
                                            dot={false}
                                            connectNulls
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <Empty description={detailSignal ? "暂无数据" : "请选择信号查看趋势"} image={Empty.PRESENTED_IMAGE_SIMPLE} />
                    )}
                </Spin>
            </>
        ),
    };

    const klineTab = {
        key: 'kline',
        label: 'K线图',
        children: (
            <>
                <div style={{ marginBottom: 12 }}>
                    <Row gutter={[12, 8]} align="middle">
                        <Col flex="auto">
                            <RangePicker
                                style={{ width: '100%' }}
                                value={klineDateRange}
                                onChange={(dates) => {
                                    setKlineDateRange(dates);
                                    if (dates && dates[0] && dates[1]) {
                                        loadKlineData(vtSymbol, dates);
                                    }
                                }}
                                format="YYYY-MM-DD"
                            />
                        </Col>
                        <Col>
                            <Button
                                type="primary"
                                onClick={() => loadKlineData(vtSymbol, klineDateRange)}
                                loading={klineLoading}
                            >
                                查询
                            </Button>
                        </Col>
                    </Row>
                </div>
                <Spin spinning={klineLoading}>
                    {klineData.length > 0 ? (
                        <div ref={klineContainerRef} style={{ width: '100%', height: 400 }} />
                    ) : (
                        <Empty description="选择日期范围后点击查询" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                    )}
                </Spin>
            </>
        ),
    };

    const items = [...(extraTabs || []), signalTab, klineTab];

    return <Tabs defaultActiveKey={defaultTab || 'signal'} items={items} />;
};

export default StockDetailTabs;
