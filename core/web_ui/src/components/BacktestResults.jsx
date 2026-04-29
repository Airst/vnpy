import React, { useState, useMemo } from 'react';
import { Tabs, Card, Table, Row, Col, Statistic, Empty, Alert } from 'antd';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BacktestResults = ({ result }) => {
    const stats = result.statistics || {};
    const dailyData = result.daily_data || [];
    const trades = result.trades || [];

    // Benchmark color palette
    const BENCHMARK_COLORS = {
        "上证指数": "#ff4d4f",
        "深证成指": "#52c41a",
        "沪深300": "#faad14",
    };

    // Merge benchmark values into dailyData for chart rendering
    const chartData = useMemo(() => {
        if (!dailyData || dailyData.length === 0) return [];
        const benchmarks = result.benchmarks || {};
        const names = Object.keys(benchmarks);
        const initialCapital = stats.capital || dailyData[0]?.balance || 1000000;
        if (names.length === 0) {
            const baseNav = dailyData[0]?.balance || initialCapital;
            return dailyData.map(entry => ({
                ...entry,
                portfolioNav: entry.balance / baseNav,
            }));
        }

        const lookup = {};
        names.forEach(name => {
            (benchmarks[name] || []).forEach(pt => {
                if (!lookup[pt.date]) lookup[pt.date] = {};
                lookup[pt.date][name] = pt.value;
            });
        });


        const baseNav = dailyData[0]?.balance || initialCapital;
        return dailyData.map(entry => {
            const extra = lookup[entry.date] || {};
            return { ...entry, portfolioNav: entry.balance / baseNav, ...extra };
        });
    }, [dailyData, result.benchmarks, stats.capital]);

    const hasBenchmarks = result.benchmarks && Object.keys(result.benchmarks).length > 0;
    
    // Format statistics data for display
    const statsData = Object.entries(stats).map(([key, value]) => ({
        key,
        metric: key.replace(/_/g, ' ').toUpperCase(),
        value: typeof value === 'number' ? value.toFixed(4) : value
    }));

    // Extract key metrics for KPI display
    const keyMetrics = {
        total_return: stats.total_return || 0,
        annual_return: stats.annual_return || 0,
        sharpe_ratio: stats.sharpe_ratio || 0,
        max_drawdown: stats.max_drawdown || 0,
        max_ddpercent: stats.max_ddpercent || 0,
        end_balance: stats.end_balance || 0,
        total_trades: stats.total_trade_count || 0,
        profit_days: stats.profit_days || 0,
        loss_days: stats.loss_days || 0
    };

    // Analyze data ranges
    const dailyDataRange = dailyData && dailyData.length > 0 ? {
        start: dailyData[0]?.date,
        end: dailyData[dailyData.length - 1]?.date
    } : null;

    const tradesDateRange = trades && trades.length > 0 ? {
        start: trades[0]?.date,
        end: trades[trades.length - 1]?.date,
        count: trades.length
    } : null;

    const tradeColumns = [
        { title: 'Date', dataIndex: 'date', key: 'date', width: 100 },
        { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', width: 80 },
        { title: 'Direction', dataIndex: 'direction', key: 'direction', width: 80 },
        { 
            title: 'Price', 
            dataIndex: 'price', 
            key: 'price',
            width: 80,
            render: (v) => typeof v === 'number' ? v.toFixed(2) : v
        },
        { title: 'Volume', dataIndex: 'volume', key: 'volume', width: 80 },
        { 
            title: 'PnL', 
            dataIndex: 'pnl', 
            key: 'pnl',
            width: 80,
            render: (v) => typeof v === 'number' ? v.toFixed(2) : v
        }
    ];

    const tradeData = (trades || []).map((t, idx) => ({ ...t, key: idx }));

    const tabs = [
        {
            key: 'metrics',
            label: 'Performance Metrics',
            children: (
                <Card title="Key Performance Indicators" bordered={false} style={{ marginBottom: 20 }}>
                    <Row gutter={[16, 16]}>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Total Return" 
                                value={keyMetrics.total_return} 
                                precision={2}
                                suffix="%" 
                                valueStyle={{ color: keyMetrics.total_return > 0 ? '#52c41a' : '#ff4d4f' }}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Annual Return" 
                                value={keyMetrics.annual_return} 
                                precision={2}
                                suffix="%" 
                                valueStyle={{ color: keyMetrics.annual_return > 0 ? '#52c41a' : '#ff4d4f' }}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Sharpe Ratio" 
                                value={keyMetrics.sharpe_ratio} 
                                precision={2}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Max Drawdown %" 
                                value={Math.abs(keyMetrics.max_ddpercent)} 
                                precision={2}
                                suffix="%" 
                                valueStyle={{ color: '#ff4d4f' }}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Final Balance" 
                                value={keyMetrics.end_balance} 
                                precision={0}
                                prefix="¥" 
                                valueStyle={{ color: '#1890ff' }}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Total Trades" 
                                value={keyMetrics.total_trades} 
                                precision={0}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Profit Days" 
                                value={keyMetrics.profit_days} 
                                precision={0}
                                valueStyle={{ color: '#52c41a' }}
                            />
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                            <Statistic 
                                title="Loss Days" 
                                value={keyMetrics.loss_days} 
                                precision={0}
                                valueStyle={{ color: '#ff4d4f' }}
                            />
                        </Col>
                    </Row>

                    <div style={{ marginTop: 30 }}>
                        <h3>All Metrics</h3>
                        <Table 
                            columns={[
                                { title: 'Metric', dataIndex: 'metric', key: 'metric', width: 200 },
                                { title: 'Value', dataIndex: 'value', key: 'value', width: 150 }
                            ]}
                            dataSource={statsData}
                            pagination={false}
                            size="small"
                            bordered
                            scroll={{ x: 350 }}
                        />
                    </div>
                </Card>
            )
        },
        {
            key: 'equity',
            label: 'Equity Curve',
            children: dailyData && dailyData.length > 0 ? (
                <Card title={hasBenchmarks ? "NAV曲线对比 (1.00 = 1元)" : "策略 NAV 曲线"} bordered={false}>
                    <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                                dataKey="date" 
                                tick={{ fontSize: 12 }}
                                angle={-45}
                                textAnchor="end"
                                height={80}
                            />
                            <YAxis 
                                tickFormatter={(value) => `${value.toFixed(2)}`}
                            />
                            <Tooltip 
                                formatter={(value, name) => {
                                    const formatted = typeof value === 'number' ? `${value.toFixed(3)}` : value;
                                    return [formatted, name];
                                }}
                                labelFormatter={(label) => `Date: ${label}`}
                            />
                            <Legend />
                            <Line
                                type="monotone"
                                dataKey="portfolioNav"
                                stroke="#1890ff"
                                name="策略 NAV"
                                dot={false}
                                isAnimationActive={false}
                            />
                            {hasBenchmarks && Object.keys(result.benchmarks).map(name => (
                                <Line 
                                    key={name}
                                    type="monotone"
                                    dataKey={name}
                                    stroke={BENCHMARK_COLORS[name] || "#999"}
                                    name={name}
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </Card>
            ) : (
                <Card title="Equity Curve" bordered={false}>
                    <Empty description="No equity curve data available" />
                </Card>
            )
        },
        {
            key: 'drawdown',
            label: 'Drawdown',
            children: dailyData && dailyData.length > 0 ? (
                <Card title="Drawdown Analysis" bordered={false}>
                    <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={dailyData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                                dataKey="date" 
                                tick={{ fontSize: 12 }}
                                angle={-45}
                                textAnchor="end"
                                height={80}
                            />
                            <YAxis />
                            <Tooltip formatter={(value) => value.toFixed(0)} />
                            <Legend />
                            <Bar dataKey="drawdown" fill="#ff4d4f" name="Drawdown" />
                        </BarChart>
                    </ResponsiveContainer>
                </Card>
            ) : (
                <Card title="Drawdown" bordered={false}>
                    <Empty description="No drawdown data available" />
                </Card>
            )
        },
        {
            key: 'trades',
            label: 'Trade Details',
            children: (
                <div>
                    {tradesDateRange && (
                        <div style={{ marginBottom: '16px', padding: '12px', background: '#f6f8fb', borderRadius: '4px' }}>
                            <strong>Trade Summary:</strong> {tradesDateRange.count} trades from {tradesDateRange.start} to {tradesDateRange.end}
                        </div>
                    )}
                    {tradeData && tradeData.length > 0 ? (
                        <Card title="All Trades" bordered={false}>
                            <Table 
                                columns={tradeColumns}
                                dataSource={tradeData}
                                pagination={{ pageSize: 20, showTotal: (total) => `Total ${total} trades` }}
                                size="small"
                                scroll={{ x: 600 }}
                            />
                        </Card>
                    ) : (
                        <Card title="Trade Details" bordered={false}>
                            <Empty description="No trade data available" />
                        </Card>
                    )}
                </div>
            )
        }
    ];

    return (
        <Tabs 
            items={tabs} 
            defaultActiveKey="metrics"
            style={{ background: 'white', padding: '20px', borderRadius: '4px' }}
        />
    );
};

export default BacktestResults;
