import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Row, Col, Spin, Empty, Select, Button, DatePicker, Tabs } from 'antd';
import { createChart, CandlestickSeries, HistogramSeries, LineSeries, createSeriesMarkers } from 'lightweight-charts';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;

const SIGNAL_COLORS = ['#8884d8', '#ff7300', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c', '#8dd1e1', '#83a6ed'];

const StockDetailTabs = ({ vtSymbol, signals: signalsProp, extraTabs, defaultTab, trades }) => {
    const [signals, setSignals] = useState(signalsProp || []);
    const [selectedSignal, setSelectedSignal] = useState(null);
    const [dateRange, setDateRange] = useState([dayjs().subtract(1, 'year'), dayjs()]);
    const [loading, setLoading] = useState(false);
    const [klineData, setKlineData] = useState([]);
    const [signalData, setSignalData] = useState([]);

    const chartContainerRef = useRef(null);
    const chartRef = useRef(null);
    const signalSeriesRef = useRef([]);
    const markersRef = useRef(null);

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

    useEffect(() => {
        if (signals.length > 0 && !selectedSignal) {
            setSelectedSignal(signals[signals.length - 1]);
        }
    }, [signals, selectedSignal]);

    useEffect(() => {
        setSelectedSignal(null);
        setKlineData([]);
        setSignalData([]);
        setDateRange([dayjs().subtract(1, 'year'), dayjs()]);
        if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
        }
        signalSeriesRef.current = [];
    }, [vtSymbol]);

    const loadData = useCallback(async (symbol, signalName, range) => {
        if (!symbol || !range || !range[0] || !range[1]) return;
        setLoading(true);
        try {
            const startDate = range[0].format("YYYYMMDD");
            const endDate = range[1].format("YYYYMMDD");

            const klineParams = new URLSearchParams({ vt_symbol: symbol, start_date: startDate, end_date: endDate });
            const klinePromise = fetch(`/api/kline?${klineParams}`).then(r => r.json());

            let signalPromise = Promise.resolve(null);
            if (signalName) {
                signalPromise = fetch('/api/signal_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ signal_name: signalName, start_date: startDate, end_date: endDate, vt_symbols: [symbol] })
                }).then(r => r.json());
            }

            const [klineResult, signalResult] = await Promise.all([klinePromise, signalPromise]);
            setKlineData(klineResult.data || []);

            if (signalResult && signalResult.dates && signalResult.series) {
                const seriesData = signalResult.series.map(s => ({
                    name: s.name,
                    data: signalResult.dates.map((date, idx) => ({
                        time: date,
                        value: s.data[idx],
                    })).filter(p => p.value != null),
                }));
                setSignalData(seriesData);
            } else {
                setSignalData([]);
            }
        } catch {
            setKlineData([]);
            setSignalData([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (!chartContainerRef.current || klineData.length === 0) {
            if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
            signalSeriesRef.current = [];
            return;
        }
        if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
        signalSeriesRef.current = [];

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 450,
            layout: { background: { color: '#ffffff' }, textColor: '#333' },
            grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
            timeScale: { timeVisible: false, borderColor: '#d1d4dc' },
            crosshair: { mode: 0 },
            rightPriceScale: { borderColor: '#d1d4dc' },
            leftPriceScale: { visible: true, borderColor: '#d1d4dc' },
        });

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#ef5350', downColor: '#26a69a',
            borderUpColor: '#ef5350', borderDownColor: '#26a69a',
            wickUpColor: '#ef5350', wickDownColor: '#26a69a',
            priceScaleId: 'right',
        });
        candleSeries.setData(klineData);

        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: { type: 'volume' }, priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
        volumeSeries.setData(klineData.map(d => ({
            time: d.time, value: d.volume,
            color: d.close >= d.open ? '#ef535080' : '#26a69a80',
        })));

        signalData.forEach((s, idx) => {
            const lineSeries = chart.addSeries(LineSeries, {
                color: SIGNAL_COLORS[idx % SIGNAL_COLORS.length],
                lineWidth: 2,
                priceScaleId: 'left',
                title: s.name,
            });
            lineSeries.setData(s.data);
            signalSeriesRef.current.push(lineSeries);
        });

        if (trades && trades.length > 0) {
            const klineTimes = new Set(klineData.map(d => d.time));
            const markers = trades
                .map(t => ({ ...t, _date: (t.date || '').slice(0, 10) }))
                .filter(t => klineTimes.has(t._date))
                .sort((a, b) => a._date.localeCompare(b._date))
                .map(t => {
                    const isBuy = t.direction && t.direction.includes('多');
                    return {
                        time: t._date,
                        position: isBuy ? 'belowBar' : 'aboveBar',
                        shape: isBuy ? 'arrowUp' : 'arrowDown',
                        color: isBuy ? '#52c41a' : '#ff4d4f',
                        text: `${isBuy ? '买' : '卖'} ${typeof t.price === 'number' ? t.price.toFixed(2) : t.price}`,
                    };
                });
            if (markers.length > 0) {
                markersRef.current = createSeriesMarkers(candleSeries, markers);
            }
        }

        chart.priceScale('right').applyOptions({ scaleMargins: { top: 0.05, bottom: 0.2 } });
        chart.priceScale('left').applyOptions({ scaleMargins: { top: 0.05, bottom: 0.2 } });

        chart.timeScale().fitContent();
        chartRef.current = chart;

        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
            if (markersRef.current) { markersRef.current.detach(); markersRef.current = null; }
            if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
            signalSeriesRef.current = [];
        };
    }, [klineData, signalData, trades]);

    const chartTab = {
        key: 'chart',
        label: 'K线 + 信号',
        children: (
            <>
                <div style={{ marginBottom: 12 }}>
                    <Row gutter={[12, 8]} align="middle">
                        <Col flex="auto">
                            <Select
                                style={{ width: '100%' }}
                                placeholder="选择信号（叠加在K线图上）"
                                options={signals.map(s => ({ value: s, label: s }))}
                                value={selectedSignal}
                                onChange={(value) => {
                                    setSelectedSignal(value);
                                    loadData(vtSymbol, value, dateRange);
                                }}
                                allowClear
                            />
                        </Col>
                        <Col>
                            <RangePicker
                                value={dateRange}
                                onChange={(dates) => {
                                    setDateRange(dates);
                                    if (dates && dates[0] && dates[1]) {
                                        loadData(vtSymbol, selectedSignal, dates);
                                    }
                                }}
                                format="YYYY-MM-DD"
                            />
                        </Col>
                        <Col>
                            <Button
                                type="primary"
                                onClick={() => loadData(vtSymbol, selectedSignal, dateRange)}
                                loading={loading}
                            >
                                查询
                            </Button>
                        </Col>
                    </Row>
                </div>
                <Spin spinning={loading}>
                    {klineData.length > 0 ? (
                        <div>
                            <div ref={chartContainerRef} style={{ width: '100%', height: 450 }} />
                            <div style={{ marginTop: 4, fontSize: 11, color: '#999' }}>
                                左轴：信号值 | 右轴：价格{trades && trades.length > 0 ? ' | ▲ 买入 ▼ 卖出' : ''}
                            </div>
                        </div>
                    ) : (
                        <Empty description="选择日期范围后点击查询" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                    )}
                </Spin>
            </>
        ),
    };

    const items = [...(extraTabs || []), chartTab];

    return <Tabs defaultActiveKey={defaultTab || 'chart'} items={items} />;
};

export default StockDetailTabs;
