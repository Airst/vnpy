import React, { useState, useEffect } from 'react';
import {
    Layout, Menu, Button, Card, DatePicker, message, Select,
    Typography, Space, Spin, InputNumber
} from 'antd';
import { BarChartOutlined, LineChartOutlined, TransactionOutlined, FileSearchOutlined } from '@ant-design/icons';
import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom';
import dayjs from 'dayjs';
import BacktestResults from './components/BacktestResults';
import SignalAnalysis from './components/SignalAnalysis';
import TradeDashboard from './components/TradeDashboard';
import LlmEvaluation from './components/LlmEvaluation';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const App = () => {
    const navigate = useNavigate();
    const location = useLocation();
    
    const [strategies, setStrategies] = useState([]);
    const [loadingStrategies, setLoadingStrategies] = useState(false);
    const [factors, setFactors] = useState([]);
    
    // Backtest State
    const [btStrategy, setBtStrategy] = useState(null);
    const [btSignal, setBtSignal] = useState(null);
    const [btMaxHoldings, setBtMaxHoldings] = useState(1);
    const [btStart, setBtStart] = useState(null);
    const [btEnd, setBtEnd] = useState(null);
    const [btResult, setBtResult] = useState(null);
    const [btLoading, setBtLoading] = useState(false);
    const [btHistory, setBtHistory] = useState([]);
    const [loadingHistory, setLoadingHistory] = useState(false);

    useEffect(() => {
        loadStrategies();
        loadFactors();
        loadDataRange();
        loadBacktestHistory();
    }, []);

    const loadDataRange = async () => {
        try {
            const res = await fetch('/api/data_range');
            const data = await res.json();
            if (data.start && data.end) {
                setBtStart(dayjs(data.start, "YYYYMMDD"));
                setBtEnd(dayjs(data.end, "YYYYMMDD"));
            }
        } catch (error) {
            console.error("Failed to load data range", error);
        }
    };

    const loadStrategies = async () => {
        setLoadingStrategies(true);
        try {
            const res = await fetch('/strategies');
            const data = await res.json();
            setStrategies(data.strategies.map(s => ({ value: s, label: s })));
        } catch (error) {
            message.error('Failed to load strategies');
        } finally {
            setLoadingStrategies(false);
        }
    };

    const loadFactors = async () => {
        try {
            const res = await fetch('/factors');
            const data = await res.json();
            setFactors(data.factors.map(f => ({ value: f, label: f })));
        } catch (error) {
            message.error('Failed to load factors');
        }
    };

    const loadBacktestHistory = async () => {
        setLoadingHistory(true);
        try {
            const res = await fetch('/api/backtest/history');
            const data = await res.json();
            setBtHistory(data.history.map(h => ({
                value: h.filename,
                label: `${h.strategy} (${h.start_date}-${h.end_date}) - ${h.timestamp}`
            })));
        } catch (error) {
            console.error("Failed to load backtest history", error);
        } finally {
            setLoadingHistory(false);
        }
    };

    const handleLoadHistory = async (filename) => {
         if (!filename) return;
         setBtLoading(true);
         try {
             const res = await fetch(`/api/backtest/result/${filename}`);
             if (!res.ok) throw new Error(res.statusText);
             const data = await res.json();
             setBtResult(data);
             message.success('Loaded backtest result');
         } catch (error) {
             message.error('Failed to load backtest result');
             console.error(error);
         } finally {
             setBtLoading(false);
         }
    };

    const handleBacktest = async () => {
        if (!btStrategy || !btStart || !btEnd) {
            message.warning('Please fill in all backtest fields');
            return;
        }
        setBtLoading(true);
        setBtResult(null);
        try {
            const payload = {
                strategy_name: btStrategy,
                start_date: btStart.format("YYYYMMDD"),
                end_date: btEnd.format("YYYYMMDD"),
                setting: {
                    signal_name: btSignal,
                    max_holdings: btMaxHoldings
                }
            };
            const res = await fetch('/api/backtest', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            setBtResult(data);
            message.success('Backtest completed');
            loadBacktestHistory(); // Refresh history
        } catch (error) {
            message.error('Backtest failed');
            console.error(error);
        } finally {
            setBtLoading(false);
        }
    };

    const menuItems = [
        {
            key: '/trade',
            label: 'Trade',
            icon: <TransactionOutlined />
        },
        {
            key: '/backtest',
            label: 'Backtest',
            icon: <BarChartOutlined />
        },
        {
            key: '/signal',
            label: 'Signal Analysis',
            icon: <LineChartOutlined />
        },
        {
            key: '/llm',
            label: 'LLM 评估',
            icon: <FileSearchOutlined />
        },
    ];

    // Backtest Content
    const renderBacktest = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
            {/* 操作区 */}
            <Card title="Configuration" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Load History</label>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="Select Past Result"
                            options={btHistory}
                            onChange={handleLoadHistory}
                            onDropdownVisibleChange={(open) => { if (open) loadBacktestHistory(); }}
                            allowClear
                            loading={loadingHistory}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Strategy</label>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="Select Strategy"
                            options={strategies}
                            onChange={setBtStrategy}
                            value={btStrategy}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Signal</label>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="Select Signal (Optional)"
                            options={factors}
                            onChange={setBtSignal}
                            value={btSignal}
                            allowClear
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Max Holdings</label>
                        <InputNumber
                            style={{ width: '100%' }}
                            min={1}
                            max={100}
                            value={btMaxHoldings}
                            onChange={setBtMaxHoldings}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>Start Date</label>
                        <DatePicker 
                            style={{ width: '100%' }}
                            value={btStart} 
                            onChange={setBtStart} 
                            format="YYYYMMDD"
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '12px', color: '#666', display: 'block', marginBottom: '4px' }}>End Date</label>
                        <DatePicker 
                            style={{ width: '100%' }}
                            value={btEnd} 
                            onChange={setBtEnd} 
                            format="YYYYMMDD"
                        />
                    </div>
                    <Button type="primary" onClick={handleBacktest} loading={btLoading} block size="large">
                        Run Backtest
                    </Button>
                </Space>
            </Card>

            {/* 数据展示区 */}
            <Spin spinning={btLoading} style={{ flex: 1, overflow: 'hidden' }}>
                {btResult ? (
                    <div style={{ height: '100%', overflow: 'auto' }}>
                        <BacktestResults result={btResult} />
                    </div>
                ) : (
                    <Card 
                        title="Results" 
                        bordered={false}
                        style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    >
                        <Text type="secondary">Run backtest to see results</Text>
                    </Card>
                )}
            </Spin>
        </div>
    );

    return (
        <Layout style={{ minHeight: '100vh' }}>
            <Header style={{ 
                background: '#001529', 
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                padding: '0 24px',
                fontSize: '18px',
                fontWeight: 'bold'
            }}>
                A-Share Analysis System
            </Header>
            <Layout style={{ flex: 1 }}>
                <Sider width={200} style={{ background: '#fff' }} collapsible>
                    <Menu
                        mode="vertical"
                        selectedKeys={[location.pathname]}
                        onClick={(e) => navigate(e.key)}
                        items={menuItems}
                        style={{ border: 'none' }}
                    />
                </Sider>
                <Content style={{ 
                    padding: '24px', 
                    background: '#f0f2f5', 
                    overflow: 'auto',
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                    <Routes>
                        <Route path="/" element={<Navigate to="/backtest" replace />} />
                        <Route path="/trade" element={<TradeDashboard />} />
                        <Route path="/backtest" element={renderBacktest()} />
                        <Route path="/signal" element={
                            <SignalAnalysis 
                                factors={factors}
                                defaultStart={btStart}
                                defaultEnd={btEnd}
                            />
                        } />
                        <Route path="/llm" element={<LlmEvaluation />} />
                    </Routes>
                </Content>
            </Layout>
        </Layout>
    );
};

export default App;