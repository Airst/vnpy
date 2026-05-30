import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Tabs, Card, Table, Row, Col, Statistic, Empty, Alert, Modal, Button, Tag, Divider, Typography, message } from 'antd';
import { EyeOutlined, ReloadOutlined, LoadingOutlined } from '@ant-design/icons';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import StockDetailTabs from './StockDetailTabs';

const { Text, Paragraph } = Typography;

const BacktestResults = ({ result }) => {
    const stats = result.statistics || {};
    const dailyData = result.daily_data || [];
    const trades = result.trades || [];

    const [detailModal, setDetailModal] = useState({ visible: false, record: null });
    const [llmRating, setLlmRating] = useState(null);
    const [llmLoading, setLlmLoading] = useState(false);
    const [taskStatus, setTaskStatus] = useState(null);
    const pollTimerRef = useRef(null);

    // Cleanup poll timer on unmount
    useEffect(() => {
        return () => { if (pollTimerRef.current) clearInterval(pollTimerRef.current); };
    }, []);

    // Fetch LLM rating when detail modal opens
    useEffect(() => {
        if (!detailModal.visible || !detailModal.record) {
            setLlmRating(null);
            setTaskStatus(null);
            if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }
            return;
        }
        setLlmLoading(true);
        setLlmRating(null);
        fetch(`/api/llm_ratings?vt_symbol=${detailModal.record.symbol}`)
            .then(res => res.json())
            .then(data => {
                const history = data.history || [];
                if (history.length > 0) {
                    setLlmRating(history[history.length - 1]);
                } else {
                    setLlmRating({ not_evaluated: true, reason: '该股票尚未被 LLM 评估' });
                }
            })
            .catch(() => setLlmRating(null))
            .finally(() => setLlmLoading(false));
    }, [detailModal.visible, detailModal.record?.symbol]);

    const startPolling = (vt_symbol) => {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        pollTimerRef.current = setInterval(async () => {
            try {
                const res = await fetch(`/api/llm_ratings/task_status/${vt_symbol}`);
                const data = await res.json();
                setTaskStatus(data);
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollTimerRef.current);
                    pollTimerRef.current = null;
                    if (data.status === 'completed') message.success(`${vt_symbol} 评估完成`);
                    // Refresh rating data
                    const ratingRes = await fetch(`/api/llm_ratings?vt_symbol=${vt_symbol}`);
                    const ratingData = await ratingRes.json();
                    const history = ratingData.history || [];
                    if (history.length > 0) {
                        setLlmRating(history[history.length - 1]);
                    }
                }
            } catch (error) {
                console.error('Failed to poll task status', error);
            }
        }, 3000);
    };

    const handleReevaluate = async (vt_symbol, score) => {
        setTaskStatus({ status: 'running', message: 'LLM 评估中...' });
        try {
            const url = `/api/llm_ratings/reevaluate?vt_symbol=${vt_symbol}&score=${score || 0}`;
            const res = await fetch(url, { method: 'POST' });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || '提交失败');
            }
            message.success(`${vt_symbol} 评估任务已提交，后台执行中...`);
            startPolling(vt_symbol);
        } catch (error) {
            setTaskStatus({ status: 'failed', message: error.message });
            message.error(`提交失败: ${error.message}`);
        }
    };

    const handleRefreshDetail = async (vt_symbol) => {
        setTaskStatus(null);
        try {
            const res = await fetch(`/api/llm_ratings?vt_symbol=${vt_symbol}`);
            if (res.ok) {
                const data = await res.json();
                const history = data.history || [];
                const latest = history.length > 0 ? history[history.length - 1] : null;
                if (latest) {
                    setLlmRating(latest);
                    message.success(`${vt_symbol} 数据已刷新`);
                } else {
                    setLlmRating({ not_evaluated: true, reason: '该股票尚未被 LLM 评估' });
                }
            }
        } catch (error) {
            console.error('Failed to refresh detail', error);
        }
    };

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
        },
        {
            title: '操作',
            key: 'operations',
            width: 80,
            render: (_, record) => (
                <Button
                    type="link"
                    icon={<EyeOutlined />}
                    onClick={() => setDetailModal({ visible: true, record })}
                >
                    详情
                </Button>
            ),
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
                            <style>{`.holding-row td { background-color: #fff1f0 !important; color: #cf1322; }`}</style>
                            <Table 
                                columns={tradeColumns}
                                dataSource={tradeData}
                                pagination={{ pageSize: 20, showTotal: (total) => `Total ${total} trades` }}
                                size="small"
                                scroll={{ x: 600 }}
                                rowClassName={(record) => record.holding ? 'holding-row' : ''}
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

    const getActionTag = (record) => {
        const action = record?.action?.toLowerCase();
        const rating = record?.rating?.toLowerCase();
        const finalAction = (action && ['buy_now', 'wait', 'avoid'].includes(action)) ? action
            : { good: 'buy_now', bad: 'avoid', neutral: 'wait' }[rating] || 'wait';
        const config = {
            buy_now: { color: 'green', text: '建议进场' },
            avoid: { color: 'red', text: '建议回避' },
            wait: { color: 'orange', text: '等待时机' },
        };
        const c = config[finalAction] || config.wait;
        return <Tag color={c.color}>{c.text}</Tag>;
    };

    const renderLlmEvaluation = () => {
        if (llmLoading) return <Empty description="加载中..." />;
        if (!llmRating) return <Empty description="无评估数据" />;
        if (llmRating.not_evaluated) return <Empty description={llmRating.reason} />;

        const dimensions = llmRating.analysis_dimensions || {};
        const keyFactors = llmRating.key_factors || [];
        const positiveFactors = keyFactors.filter(f => f.type === 'positive');
        const negativeFactors = keyFactors.filter(f => f.type === 'negative');

        return (
            <>
                {llmRating.date && (
                    <div style={{ marginBottom: 12 }}>
                        <Text type="secondary">评估日期：</Text>
                        <Text strong>{llmRating.date}</Text>
                    </div>
                )}
                <div style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 16]}>
                        <Col span={8}>
                            <Text type="secondary">进场建议：</Text>
                            {getActionTag(llmRating)}
                        </Col>
                        <Col span={8}>
                            <Text type="secondary">置信度：</Text>
                            <Text strong>{(llmRating.confidence * 100).toFixed(0)}%</Text>
                        </Col>
                        {llmRating.score != null && (
                            <Col span={8}>
                                <Text type="secondary">模型分数：</Text>
                                <Text code>{llmRating.score.toFixed(4)}</Text>
                            </Col>
                        )}
                    </Row>
                </div>

                <div style={{ marginBottom: 16 }}>
                    <Text type="secondary">评估理由：</Text>
                    <Paragraph style={{ marginTop: 4 }}>{llmRating.reason}</Paragraph>
                </div>

                {Object.keys(dimensions).length > 0 && (
                    <>
                        <Divider orientation="left">分析维度</Divider>
                        <Row gutter={[8, 12]}>
                            {Object.entries(dimensions).map(([key, val]) => (
                                <Col span={12} key={key}>
                                    <Text type="secondary">{key}：</Text>
                                    <div>{val}</div>
                                </Col>
                            ))}
                        </Row>
                    </>
                )}

                {(positiveFactors.length > 0 || negativeFactors.length > 0) && (
                    <>
                        <Divider orientation="left">关键因素</Divider>
                        <Row gutter={16}>
                            {positiveFactors.length > 0 && (
                                <Col span={12}>
                                    <Text type="success">正面因素：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {positiveFactors.map((f, i) => (
                                            <li key={i}><Text>{f.content}</Text></li>
                                        ))}
                                    </ul>
                                </Col>
                            )}
                            {negativeFactors.length > 0 && (
                                <Col span={12}>
                                    <Text type="danger">负面因素：</Text>
                                    <ul style={{ paddingLeft: 20, marginTop: 4 }}>
                                        {negativeFactors.map((f, i) => (
                                            <li key={i}><Text>{f.content}</Text></li>
                                        ))}
                                    </ul>
                                </Col>
                            )}
                        </Row>
                    </>
                )}

                {llmRating.error && (
                    <Alert message="LLM 调用错误" description={llmRating.error} type="error" showIcon style={{ marginTop: 12 }} />
                )}
            </>
        );
    };

    const renderDetailModal = () => {
        const { visible, record } = detailModal;
        if (!record) return null;

        const extraTabs = [{
            key: 'llm',
            label: 'LLM 评估',
            children: renderLlmEvaluation(),
        }];

        const reevaluateBtnText = taskStatus?.status === 'running' ? '评估中...'
            : taskStatus?.status === 'completed' ? '重新评估'
            : taskStatus?.status === 'failed' ? '重试'
            : '重新评估';

        return (
            <Modal
                title={`${record.symbol} - 详情`}
                open={visible}
                onCancel={() => { setDetailModal({ visible: false, record: null }); setTaskStatus(null); }}
                footer={[
                    <Button
                        key="reevaluate"
                        icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : <ReloadOutlined />}
                        onClick={() => handleReevaluate(record.symbol, llmRating?.score)}
                        loading={taskStatus?.status === 'running'}
                        disabled={taskStatus?.status === 'running'}
                    >
                        {reevaluateBtnText}
                    </Button>,
                    <Button
                        key="refresh"
                        icon={<ReloadOutlined />}
                        onClick={() => handleRefreshDetail(record.symbol)}
                    >
                        刷新
                    </Button>,
                    <Button key="close" type="primary" onClick={() => { setDetailModal({ visible: false, record: null }); setTaskStatus(null); }}>关闭</Button>,
                ]}
                width={900}
                destroyOnClose
            >
                {taskStatus && (
                    <Alert
                        message={taskStatus.status === 'running' ? '任务执行中' :
                                 taskStatus.status === 'completed' ? '评估完成' : '评估失败'}
                        description={taskStatus.message}
                        type={taskStatus.status === 'running' ? 'info' :
                              taskStatus.status === 'completed' ? 'success' : 'error'}
                        showIcon
                        icon={taskStatus?.status === 'running' ? <LoadingOutlined spin /> : undefined}
                        style={{ marginBottom: 16 }}
                    />
                )}
                <div style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 8]}>
                        <Col span={6}><span style={{ color: '#888' }}>日期：</span>{record.date}</Col>
                        <Col span={6}><span style={{ color: '#888' }}>方向：</span>{record.direction}</Col>
                        <Col span={6}><span style={{ color: '#888' }}>价格：</span>{typeof record.price === 'number' ? record.price.toFixed(2) : record.price}</Col>
                        <Col span={6}><span style={{ color: '#888' }}>PnL：</span><span style={{ color: (record.pnl || 0) >= 0 ? '#52c41a' : '#ff4d4f' }}>{typeof record.pnl === 'number' ? record.pnl.toFixed(2) : record.pnl}</span></Col>
                    </Row>
                </div>
                <StockDetailTabs
                    vtSymbol={record.symbol}
                    defaultTab="llm"
                    extraTabs={extraTabs}
                    trades={trades.filter(t => t.symbol === record.symbol)}
                />
            </Modal>
        );
    };

    return (
        <>
        <Tabs 
            items={tabs} 
            defaultActiveKey="metrics"
            style={{ background: 'white', padding: '20px', borderRadius: '4px' }}
        />
        {renderDetailModal()}
        </>
    );
};

export default BacktestResults;
