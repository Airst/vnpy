import React, { useState, useEffect } from 'react';
import { Card, Select, DatePicker, Button, Space, message, Spin, Empty } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;

const SignalAnalysis = ({ factors = [], defaultStart, defaultEnd }) => {
    const [selectedSignal, setSelectedSignal] = useState(null);
    const [selectedSymbols, setSelectedSymbols] = useState([]);
    const [dateRange, setDateRange] = useState([defaultStart, defaultEnd]);
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState(null);
    const [chartData, setChartData] = useState([]);

    const colors = [
        '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#d0ed57', 
        '#a4de6c', '#8dd1e1', '#83a6ed', '#8e43e7', '#f44336'
    ];

    const handleAnalyze = async () => {
        if (!selectedSignal || !dateRange || !dateRange[0] || !dateRange[1]) {
            message.warning('Please select a signal and date range');
            return;
        }

        setLoading(true);
        setData(null);
        setChartData([]);

        try {
            const payload = {
                signal_name: selectedSignal,
                start_date: dateRange[0].format("YYYYMMDD"),
                end_date: dateRange[1].format("YYYYMMDD"),
                vt_symbols: selectedSymbols
            };

            const res = await fetch('/api/signal_data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const result = await res.json();

            if (result.error) {
                message.error(result.error);
                return;
            }

            setData(result);

            // Transform data for Recharts
            // result format: { dates: [], series: [{name: '', data: []}] }
            // Recharts needs: [{date: d1, sym1: v1, sym2: v2}, ...]
            if (result.dates && result.series) {
                const transformed = result.dates.map((date, idx) => {
                    const point = { date };
                    result.series.forEach(s => {
                        point[s.name] = s.data[idx];
                    });
                    return point;
                });
                setChartData(transformed);
            }

            message.success('Analysis completed');
        } catch (error) {
            message.error('Analysis failed');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
            <Card title="Signal Analysis Configuration" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                        <div>
                            <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Signal</label>
                            <Select
                                style={{ width: '100%' }}
                                placeholder="Select Signal"
                                options={factors}
                                onChange={setSelectedSignal}
                                value={selectedSignal}
                                showSearch
                            />
                        </div>
                        <div>
                            <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Symbols (Optional - Empty for Top 5)</label>
                            <Select
                                mode="tags"
                                style={{ width: '100%' }}
                                placeholder="Enter symbols e.g. 000001.SZ"
                                onChange={setSelectedSymbols}
                                value={selectedSymbols}
                                tokenSeparators={[',', ' ']}
                            />
                        </div>
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Date Range</label>
                        <RangePicker 
                            style={{ width: '100%' }}
                            value={dateRange}
                            onChange={setDateRange}
                            format="YYYY-MM-DD"
                        />
                    </div>
                    <Button type="primary" onClick={handleAnalyze} loading={loading} block size="large">
                        Analyze Signal
                    </Button>
                </Space>
            </Card>

            <Card title="Signal Visualization" bordered={false} style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <Spin spinning={loading}>
                    {chartData.length > 0 ? (
                        <div style={{ height: '500px', width: '100%' }}>
                             <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="date" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    {data.series.map((s, idx) => (
                                        <Line 
                                            key={s.name}
                                            type="monotone" 
                                            dataKey={s.name} 
                                            stroke={colors[idx % colors.length]} 
                                            activeDot={{ r: 8 }}
                                            connectNulls
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <Empty description="No data to display. Run analysis first." />
                    )}
                </Spin>
            </Card>
        </div>
    );
};

export default SignalAnalysis;
